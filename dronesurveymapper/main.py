# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:26:10 2025

@author: Labadmin
"""

from dronesurveymapper.image.get_metadata import SurveyImagesToSpatial
from dronesurveymapper.video.video_exif_reader import DJIVideoExifReader

if __name__ == '__main__':
    import os
    os.chdir(r'D:\!Research\01 - Python\DroneSurveyMapper')
    # # Imagery
    # mapper = SurveyImagesToSpatial(survey_dir='test',
    #                                out_epsg='EPSG:32615')
    # mapper.img_to_geojson(geojson_path='test.geojson')

    # Video
    video_path = 'DJI_20250426143054_0001_T.MP4'
    output_dir = './DJI_20250426143054_0001_T/'
    VidReader = DJIVideoExifReader(video_path, output_dir)
    VidReader.extract_frames_from_video()
    frames, frame_keys = VidReader.parse_exiftxt()
    VidReader.save_frame_csv(frames, frame_keys)
