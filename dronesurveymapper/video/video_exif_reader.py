# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:33:05 2025

@author: Labadmin
"""
import os
import subprocess
import cv2
import pandas as pd

from .utils import get_first_second_indices, elapsed_seconds_since_ref, filter_dict_by_keys, dms_to_epsg4326


class DJIVideoExifReader():
    def __init__(self, video_path, output_dir, txt_path=None):
        # Ensure the input video path exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f'Video file not found: {video_path}')
        self.video_path = video_path

        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        if txt_path is None:
            self.txt_path = self.exif_to_txt()
        else:
            self.txt_path = txt_path

    def exif_to_txt(self):
        txt_file = self.video_path.replace('.MP4', '.txt')
        txt_path = os.path.join(self.output_dir, txt_file)
        with open(txt_path, 'w') as f_out:
            _ = subprocess.call(
                ['exiftool', '-ee', self.video_path],
                stdout=f_out,
                stderr=subprocess.PIPE)

        return txt_path

    def extract_frames_from_video(self):
        """
        Extracts frames from an MP4 video at a specified interval and saves them as JPG images.

        Returns:
            None
        """
        interval_seconds = 1  # TODO: Add support to this class for different interval_seconds
        # Open the video file
        video_capture = cv2.VideoCapture(self.video_path)

        # Get frames per second (FPS) of the video
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('Invalid FPS value. Cannot extract frames.')

        # Calculate the number of frames to skip between extractions
        frame_interval = int(fps * interval_seconds)

        # Initialize frame counter
        frame_count = 0
        saved_frame_count = 0

        while True:
            # Read the next frame
            success, frame = video_capture.read()
            if not success:
                break  # Exit loop if no more frames

            # Save frame if it's at the correct interval
            if frame_count % frame_interval == 0:
                # Construct output frame filename
                frame_filename = os.path.join(self.output_dir, f'frame_{saved_frame_count:05d}.jpg')

                # Save frame as JPG
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

            frame_count += 1

        # Release the video capture object
        video_capture.release()

    def parse_exiftxt(self):
        # TODO: This function needs tidying
        df = pd.read_fwf(self.txt_path, header=None, usecols=[0, 2], names=['Variable', 'Value'])
        values_to_remove = ['Protocol', 'Serial Number', 'Model', 'Frame Width', 'Frame Height', 'Frame Rate']
        df = df[~df['Variable'].isin(values_to_remove)]  # Have to do some cleaning so the loop works

        # Get image_height, image_width, and aperture
        image_height = df[df['Variable'] == 'Image Height']['Value'].iloc[0]
        image_width = df[df['Variable'] == 'Image Width']['Value'].iloc[0]
        aperture = df[df['Variable'] == 'Aperture']['Value'].iloc[0]

        # Find index where frames start
        frame_starts = [i for i, val in enumerate(df['Variable'].tolist()) if val == 'Sample Time']
        frames = {}
        prev_drone_roll = 0
        prev_drone_pitch = 0

        # Init
        ISO_value = 0

        for index in frame_starts:
            frame_dict = {}
            frame_dict['Image Height'] = image_height
            frame_dict['Image Width'] = image_width
            frame_dict['Aperture'] = aperture

            # sample_time, not highly useful due to the constantly changing format
            sample_time_row = df.iloc[index]
            sample_time_variable = sample_time_row['Variable']
            assert sample_time_variable == 'Sample Time', f'Expected Sample Time but got {sample_time_variable} at {index}'
            sample_time_value = sample_time_row['Value']

            # sample_duration
            sample_duration_row = df.iloc[index + 1]
            sample_duration_variable = sample_duration_row['Variable']
            assert sample_duration_variable == 'Sample Duration', f'Expected Sample Duration but got {sample_duration_variable} at {index + 1}'
            sample_duration_value = sample_duration_row['Value']
            frame_dict['Sample Duration'] = sample_duration_value

            # ISO
            ISO_row = df.iloc[index + 2]
            ISO_variable = ISO_row['Variable']
            try:
                assert ISO_variable == 'ISO', f'Expected ISO but got {ISO_variable} at {index + 2}'
                ISO_value = ISO_row['Value']
            except AssertionError:
                index -= 1
                frame_starts[0] -= 1
            frame_dict['ISO'] = ISO_value

            # shutter_speed
            shutter_speed_row = df.iloc[index + 3]
            shutter_speed_variable = shutter_speed_row['Variable']
            assert shutter_speed_variable == 'Shutter Speed', f'Expected Shutter Speed but got {shutter_speed_variable} at {index + 3}'
            shutter_speed_value = shutter_speed_row['Value']
            frame_dict['Shutter Speed'] = shutter_speed_value

            # f_number
            f_number_row = df.iloc[index + 4]
            f_number_variable = f_number_row['Variable']
            assert f_number_variable == 'F Number', f'Expected F Number but got {f_number_variable} at {index + 4}'
            f_number_value = f_number_row['Value']
            frame_dict['F Number'] = f_number_value

            # digital_zoom
            digital_zoom_row = df.iloc[index + 5]
            digital_zoom_variable = digital_zoom_row['Variable']
            assert digital_zoom_variable == 'Digital Zoom', f'Expected Digital Zoom but got {digital_zoom_variable} at {index + 5}'
            digital_zoom_value = digital_zoom_row['Value']
            frame_dict['Digital Zoom'] = digital_zoom_value

            # drone_roll
            drone_roll_row = df.iloc[index + 6]
            drone_roll_variable = drone_roll_row['Variable']
            try:  # Many drone rolls are missing
                assert drone_roll_variable == 'Drone Roll', f'Expected Drone Roll but got {drone_roll_variable} at {index + 6}'
                drone_roll_value = drone_roll_row['Value']
                frame_dict['Drone Roll'] = drone_roll_value
                prev_drone_roll = drone_roll_value
            except AssertionError:
                # If drone roll is not found, interpolate
                next_drone_roll_row_index = df['Variable'].tolist().index('Drone Roll', index + 7)
                next_drone_roll_row = df.iloc[next_drone_roll_row_index]
                next_drone_roll_variable = next_drone_roll_row['Variable']
                assert next_drone_roll_variable == 'Drone Roll', f'Expected Drone Roll but got {next_drone_roll_variable} at {index + 6}'
                next_drone_roll_value = next_drone_roll_row['Value']
                avg_drone_roll_value = (float(next_drone_roll_value) + float(prev_drone_roll)) / 2
                frame_dict['Drone Roll'] = avg_drone_roll_value
                index -= 1  # Increment the index to account for the missing row

            # drone_pitch
            drone_pitch_row = df.iloc[index + 7]
            drone_pitch_variable = drone_pitch_row['Variable']
            try:
                assert drone_pitch_variable == 'Drone Pitch', f'Expected Drone Pitch but got {drone_pitch_variable} at {index + 7}'
                drone_pitch_value = drone_pitch_row['Value']
                frame_dict['Drone Pitch'] = drone_pitch_value
                prev_drone_pitch = drone_pitch_value
            except AssertionError:
                # If drone pitch is not found, interpolate
                next_drone_pitch_row_index = df['Variable'].tolist().index('Drone Pitch', index + 8)
                next_drone_pitch_row = df.iloc[next_drone_pitch_row_index]
                next_drone_pitch_variable = next_drone_pitch_row['Variable']
                assert next_drone_pitch_variable == 'Drone Pitch', f'Expected Drone Pitch but got {next_drone_pitch_variable} at {index + 7}'
                next_drone_pitch_value = next_drone_pitch_row['Value']
                avg_drone_pitch_value = (float(next_drone_pitch_value) + float(prev_drone_pitch)) / 2
                frame_dict['Drone Pitch'] = avg_drone_pitch_value
                index -= 1  # Increment the index to account for the missing row

            # drone_yaw
            drone_yaw_row = df.iloc[index + 8]
            drone_yaw_variable = drone_yaw_row['Variable']
            try:
                assert drone_yaw_variable == 'Drone Yaw', f'Expected Drone Yaw but got {drone_yaw_variable} at {index + 8}'
                drone_yaw_value = drone_yaw_row['Value']
                frame_dict['Drone Yaw'] = drone_yaw_value
                prev_drone_yaw = drone_yaw_value
            except AssertionError:
                # If drone yaw is not found, interpolate
                next_drone_yaw_row_index = df['Variable'].tolist().index('Drone Yaw', index + 9)
                next_drone_yaw_row = df.iloc[next_drone_yaw_row_index]
                next_drone_yaw_variable = next_drone_yaw_row['Variable']
                assert next_drone_yaw_variable == 'Drone Yaw', f'Expected Drone Yaw but got {next_drone_yaw_variable} at {index + 8}'
                next_drone_yaw_value = next_drone_yaw_row['Value']
                avg_drone_yaw_value = (float(next_drone_yaw_value) + float(prev_drone_yaw)) / 2
                frame_dict['Drone Yaw'] = avg_drone_yaw_value
                index -= 1  # Increment the index to account for the missing row

            # gps_latitude
            gps_latitude_row = df.iloc[index + 9]
            gps_latitude_variable = gps_latitude_row['Variable']
            assert gps_latitude_variable == 'GPS Latitude', f'Expected GPS Latitude but got {gps_latitude_variable} at {index + 9}'
            gps_latitude_value = gps_latitude_row['Value']
            frame_dict['GPS Latitude'] = gps_latitude_value

            # gps_longitude
            gps_longitude_row = df.iloc[index + 10]
            gps_longitude_variable = gps_longitude_row['Variable']
            assert gps_longitude_variable == 'GPS Longitude', f'Expected GPS Longitude but got {gps_longitude_variable} at {index + 10}'
            gps_longitude_value = gps_longitude_row['Value']
            frame_dict['GPS Longitude'] = gps_longitude_value

            # absolute_altitude
            absolute_altitude_row = df.iloc[index + 11]
            absolute_altitude_variable = absolute_altitude_row['Variable']
            assert absolute_altitude_variable == 'Absolute Altitude', f'Expected Absolute Altitude but got {absolute_altitude_variable} at {index + 11}'
            absolute_altitude_value = absolute_altitude_row['Value']
            frame_dict['Absolute Altitude'] = absolute_altitude_value

            # relative_altitude
            relative_altitude_row = df.iloc[index + 12]
            relative_altitude_variable = relative_altitude_row['Variable']
            try:
                assert relative_altitude_variable == 'Relative Altitude', f'Expected Relative Altitude but got {relative_altitude_variable} at {index + 12}'
                relative_altitude_value = relative_altitude_row['Value']
                frame_dict['Relative Altitude'] = relative_altitude_value
                prev_relative_altitude = relative_altitude_value
            except AssertionError:
                # If gimbal pitch is not found, interpolate
                next_relative_altitude_row_index = df['Variable'].tolist().index('Relative Altitude', index + 13)
                next_relative_altitude_row = df.iloc[next_relative_altitude_row_index]
                next_relative_altitude_variable = next_relative_altitude_row['Variable']
                assert next_relative_altitude_variable == 'Relative Altitude', f'Expected Relative Altitude but got {next_relative_altitude_variable} at {index + 12}'
                next_relative_altitude_value = next_relative_altitude_row['Value']
                avg_relative_altitude_value = (float(next_relative_altitude_value) + float(prev_relative_altitude)) / 2
                frame_dict['Relative Altitude'] = avg_relative_altitude_value
                index -= 1  # Increment the index to account for the missing row

            # gimbal_pitch
            gimbal_pitch_row = df.iloc[index + 13]
            gimbal_pitch_variable = gimbal_pitch_row['Variable']
            try:
                assert gimbal_pitch_variable == 'Gimbal Pitch', f'Expected Gimbal Pitch but got {gimbal_pitch_variable} at {index + 13}'
                gimbal_pitch_value = gimbal_pitch_row['Value']
                frame_dict['Gimbal Pitch'] = gimbal_pitch_value
                prev_gimbal_pitch = gimbal_pitch_value
            except AssertionError:
                # If gimbal pitch is not found, interpolate
                next_gimbal_pitch_row_index = df['Variable'].tolist().index('Gimbal Pitch', index + 14)
                next_gimbal_pitch_row = df.iloc[next_gimbal_pitch_row_index]
                next_gimbal_pitch_variable = next_gimbal_pitch_row['Variable']
                assert next_gimbal_pitch_variable == 'Gimbal Pitch', f'Expected Gimbal Pitch but got {next_gimbal_pitch_variable} at {index + 13}'
                next_gimbal_pitch_value = next_gimbal_pitch_row['Value']
                avg_gimbal_pitch_value = (float(next_gimbal_pitch_value) + float(prev_gimbal_pitch)) / 2
                frame_dict['Gimbal Pitch'] = avg_gimbal_pitch_value
                index -= 1  # Increment the index to account for the missing row

            # gimbal_yaw
            gimbal_yaw_row = df.iloc[index + 14]
            gimbal_yaw_variable = gimbal_yaw_row['Variable']
            if gimbal_yaw_variable == 'Gimbal Roll':  # Gimbal Roll is very rarely recorded
                index += 1
                gimbal_yaw_row = df.iloc[index + 14]
                gimbal_yaw_variable = gimbal_yaw_row['Variable']
            try:
                assert gimbal_yaw_variable == 'Gimbal Yaw', f'Expected Gimbal Yaw but got {gimbal_yaw_variable} at {index + 14}'
                gimbal_yaw_value = gimbal_yaw_row['Value']
                frame_dict['Gimbal Yaw'] = gimbal_yaw_value
                prev_gimbal_yaw = gimbal_yaw_value
            except AssertionError:
                # If gimbal yaw is not found, interpolate
                next_gimbal_yaw_row_index = df['Variable'].tolist().index('Gimbal Yaw', index + 15)
                next_gimbal_yaw_row = df.iloc[next_gimbal_yaw_row_index]
                next_gimbal_yaw_variable = next_gimbal_yaw_row['Variable']
                assert next_gimbal_yaw_variable == 'Gimbal Yaw', f'Expected Gimbal Yaw but got {next_gimbal_yaw_variable} at {index + 14}'
                next_gimbal_yaw_value = next_gimbal_yaw_row['Value']
                avg_gimbal_yaw_value = (float(next_gimbal_yaw_value) + float(prev_gimbal_yaw)) / 2
                frame_dict['Gimbal Yaw'] = avg_gimbal_yaw_value
                index -= 1  # Increment the index to account for the missing row

            # gimbal_roll
            frame_dict['Gimbal Roll'] = 0  # Gimbal Roll is very rarely recorded

            # gps_date_time
            gps_date_time_row = df.iloc[index + 15]
            gps_date_time_variable = gps_date_time_row['Variable']
            if gps_date_time_variable == 'Warning':  # Clunky handling of Warning row for now
                gps_date_time_row = df.iloc[index + 16]
                gps_date_time_variable = gps_date_time_row['Variable']
            assert gps_date_time_variable == 'GPS Date/Time', f'Expected GPS Date/Time but got {gps_date_time_variable} at {index + 16}'
            gps_date_time_value = gps_date_time_row['Value']
            frame_dict['GPS Date/Time'] = gps_date_time_value

            if index == frame_starts[0]:
                ref_time = gps_date_time_value

            elapsed = elapsed_seconds_since_ref(ref_time, gps_date_time_value)

            frames[elapsed] = frame_dict
        _, frame_keys = get_first_second_indices(frames)
        return frames, frame_keys

    def save_frame_csv(self, frames, frame_keys):
        # Filter frames
        frames = filter_dict_by_keys(frames, frame_keys)

        # Init dataframe
        df = pd.DataFrame(columns=['Image Name', 'Sample Time', 'Sample Duration',
                                   'ISO', 'Shutter Speed', 'F Number',
                                   'Image Height', 'Image Width', 'Aperture',
                                   'Digital Zoom', 'Drone Roll', 'Drone Pitch',
                                   'Drone Yaw', 'GPS Latitude', 'GPS Longitude',
                                   'Absolute Altitude', 'Relative Altitude',
                                   'Gimbal Roll', 'Gimbal Pitch', 'Gimbal Yaw',
                                   'GPS Date/Time'])

        # Iter over frames
        for i, key in enumerate(frames):
            img_name = f'frame_{i:05d}.jpg'
            lat = frames[key]['GPS Latitude']
            lon = frames[key]['GPS Longitude']
            lat, lon = dms_to_epsg4326(lat, lon)

            row = [img_name,
                   key,
                   frames[key]['Sample Duration'],
                   frames[key]['ISO'],
                   frames[key]['Shutter Speed'],
                   frames[key]['F Number'],
                   frames[key]['Image Height'],
                   frames[key]['Image Width'],
                   frames[key]['Aperture'],
                   frames[key]['Digital Zoom'],
                   frames[key]['Drone Roll'],
                   frames[key]['Drone Pitch'],
                   frames[key]['Drone Yaw'],
                   lat,
                   lon,
                   frames[key]['Absolute Altitude'],
                   frames[key]['Relative Altitude'],
                   frames[key]['Gimbal Roll'],
                   frames[key]['Gimbal Pitch'],
                   frames[key]['Gimbal Yaw'],
                   frames[key]['GPS Date/Time']]

            df.loc[i] = row
        df.to_csv(os.path.join(self.output_dir, 'frames.csv'))
