# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:19:35 2025

@author: Labadmin
"""
import os
from collections import defaultdict
from ast import literal_eval

import numpy as np
from pyproj import Proj
import urllib.request
import pandas as pd
import shapefile
import rasterio
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point


def getWKT_PRJ(epsg_code):
    """
    Retrieves the Well-Known Text (WKT) projection string for a given EPSG code.

    Args:
        epsg_code (int): EPSG code representing the spatial reference system.

    Returns:
        output (str): WKT string with no spaces or newlines, suitable for .prj files.
    """
    # Access projection information from spatialreference.org using the EPSG code
    with urllib.request.urlopen(f'http://spatialreference.org/ref/epsg/{epsg_code}/prettywkt/') as wkt:
        # Read the response and decode it to UTF-8
        content = wkt.read().decode('utf-8')

        # Remove all spaces from the WKT string
        remove_spaces = content.replace(' ', '')

        # Remove newlines to create a single-line WKT string
        output = remove_spaces.replace('\n', '')

    return output


def rot_matrix(kappa, omega, phi):
    """
    Computes the rotation matrix from orientation angles using ZYX (Yaw-Pitch-Roll) order.
    Args:
        kappa (float): Yaw (degrees)
        omega (float): Pitch (degrees)
        phi (float): Roll (degrees)
    Returns:
        R (np.ndarray): 3x3 rotation matrix
    """
    # Convert degrees to radians
    k = np.radians(kappa)
    o = np.radians(omega)
    p = np.radians(phi)

    # Rotation around Z (Yaw)
    Rz = np.array([
        [np.cos(k), -np.sin(k), 0],
        [np.sin(k),  np.cos(k), 0],
        [0,          0,         1]
    ])

    # Rotation around Y (Pitch)
    Ry = np.array([
        [ np.cos(o), 0, np.sin(o)],
        [ 0,         1, 0],
        [-np.sin(o), 0, np.cos(o)]
    ])

    # Rotation around X (Roll)
    Rx = np.array([
        [1, 0,         0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p),  np.cos(p)]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def load_exif(exif_csv, img_path):
    # Load EXIF csv (created by DJIVideoExifReader())
    exif_df = pd.read_csv(exif_csv, index_col=0)

    # Get image name
    img_name = os.path.basename(f'{img_path}')

    # Get the corresponding row
    exif_row = exif_df[exif_df['Image Name'] == img_name]

    # Make EXIF dict
    exif_dict = {}
    for key, value in exif_row.items():
        exif_dict[key] = value.iloc[0]

    return exif_dict


def img2gis(img_path: str,
            points: np.array([np.uint16]),
            dsm_path: str,
            cfg: dict,
            full_image: bool = False
            ) -> [np.array([np.float32]), tuple([int, str])]:
    """
    Projects image-space coordinates to georeferenced UTM coordinates using camera calibration
    and DSM data.

    Args:
        img_path (str): Path to the image file.
        points (np.array): Array of shape (N, 2) with pixel coordinates. First point is the centroid.
        dsm_path (str): Path to DSM
        cfg (dict): Configuration dictionary containing calibration and image alignment info.
            calibration_dict (dict): Camera intrinsics and optional distortion parameters.
            csv_path (str, optional): Path to image alignment metadata file.
            show_box (bool, optional): If True, displays the image bounding box.
        full_image (bool, optional): If True, use the full image extent as input instead of 'points'.

    Returns:
        curr_poly (np.array): Projected polygon vertices in UTM coordinates.
        tuple (int, str): UTM zone number and letter of the image center.
    """
    # Extract path to aligned cameras CSV file, if available
    cameras_csv_path = cfg['paths'].get('aligned_cameras_csv_path', None)

    # Extract configuration settings related to projection parameters
    coarse_del_Z = cfg['img2gis'].get('coarse_height_increment', 0.5)    # Initial height step for coarse ray tracing
    fine_del_H = cfg['img2gis'].get('fine_height_increment', 0.01)       # Step size for fine ray tracing refinement
    coarse_tolerance = cfg['img2gis'].get('coarse_tolerance', 2)         # Tolerance to switch from coarse to fine ray tracing
    fine_tolerance = cfg['img2gis'].get('fine_tolerance', 0.2)           # Acceptable tolerance for final intersection with DSM
    bound = cfg['img2gis'].get('boundary', 5)                            # Spatial window size (b×b) for local DSM median filtering

    # Extract camera calibration parameters from configuration
    calibration_dict = cfg['camera_calibration']

    # Focal length
    f = calibration_dict['f']  # e.g., 11017.3

    # Principal point offsets in pixels
    cx = calibration_dict['cx']  # e.g., -6.35398
    cy = calibration_dict['cy']  # e.g., 22.0852

    # Sensor dimensions in meters
    sw = calibration_dict['sw']  # e.g., 36.0448 / 1000
    sh = calibration_dict['sh']  # e.g., 24.024 / 1000

    # Radial distortion coefficients (optional)
    k1 = calibration_dict.get('k1', 0)  # e.g., -0.00761618
    k2 = calibration_dict.get('k2', 0)  # e.g., -0.245679
    k3 = calibration_dict.get('k3', 0)  # e.g., -0.520738
    k4 = calibration_dict.get('k4', 0)  # e.g., 0.0

    # Tangential distortion coefficients (optional)
    p1 = calibration_dict.get('p1', 0)  # e.g., -0.000425435
    p2 = calibration_dict.get('p2', 0)  # e.g., 0.000371409

    # Skew parameters (optional)
    b1 = calibration_dict.get('b1', 0)  # e.g., -8.81372
    b2 = calibration_dict.get('b2', 0)  # e.g., -0.708353

    # Read DSM
    with rasterio.open(dsm_path) as src:
        dsm_arr = src.read(1)
        dsm_epsg = int(32611)  # TODO: fix hard-coded EPSG code

    # Use aligned cameras CSV if provided for more accurate pose and orientation
    if cameras_csv_path is not None:
        # Define expected CSV column headers
        columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa',
                   'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

        # Load CSV using tab separator
        df = pd.read_csv(cameras_csv_path, sep='\t', lineterminator='\n', skiprows=0, names=columns)

        # Extract image name without extension from image path (Windows-style paths assumed)
        img_name = img_path.split('\\')[-1].split('.')[0]

        # Retrieve row matching the current image
        row = df.loc[df['PhotoID'] == img_name]

        # If no match is found, log and skip this image
        if len(row) == 0:
            print(f'Error! Image |{img_name}| not found in csv file.')
            return None, None

        # Extract orientation angles (in degrees)
        omega = float(row['Omega'])
        phi = float(row['Phi'])
        kappa = float(row['Kappa'])

        # Compute rotation matrix from orientation angles
        m = rot_matrix(kappa, omega, phi)

        # Extract camera center coordinates from CSV
        Z0 = float(row['Z'])
        X0 = float(row['X'])
        Y0 = float(row['Y'])

    # Fallback: use EXIF metadata if camera CSV is not provided
    else:
        # Extract EXIF data dictionary from image
        h, _ = os.path.split(img_path)
        exif_csv = os.path.join(h, 'frames.csv')
        exif_dict = load_exif(exif_csv, img_path)

        # Extract altitude info (absolute and relative)
        Z0 = float(exif_dict['Absolute Altitude'])
        altitude = float(exif_dict['Relative Altitude'])  # affects box size

        # Image dimensions
        image_width = float(exif_dict['Image Width'])
        image_height = float(exif_dict['Image Height'])

        # Focal length in meters (convert mm to meters)
        # Focal length = aperture * F_Number
        aperture = float(exif_dict['Aperture'])
        f_number = float(exif_dict['F Number'])
        focal = float(9.1) / 1000  # TODO: Fix hard-coded focal length

        # Get GPS latitude and longitude, convert to the DSM's CRS
        gps_lat = exif_dict['GPS Latitude']
        gps_lon = exif_dict['GPS Longitude']
        p = Proj(dsm_epsg, preserve_units=True)
        X0, Y0 = p(gps_lon, gps_lat)

        # Extract gimbal rotation angles (yaw, roll, pitch)
        yaw = float(exif_dict['Gimbal Yaw'])
        roll = float(exif_dict['Gimbal Roll'])
        pitch = float(exif_dict['Gimbal Pitch'])

        drone_yaw = float(exif_dict['Drone Yaw'])
        drone_roll = float(exif_dict['Drone Roll'])
        drone_pitch = float(exif_dict['Drone Pitch'])
        print(pitch, drone_pitch)
        # pitch -= 9.5
        # print(roll, drone_roll)
        # print(yaw, drone_yaw)

        # Compute rotation matrix with adjusted angles for DJI convention
        kappa, omega, phi = 90 - yaw, pitch, roll
        m = rot_matrix(kappa, omega, phi)

    # Replace points with full image corners if requested
    if full_image:
        points = np.array([
            [image_width // 2 + cx, image_height // 2 + cy],  # image center with offsets
            [0, 0],                                           # top-left corner
            [image_width, 0],                                 # top-right corner
            [image_width, image_height],                       # bottom-right corner
            [0, image_height]                                  # bottom-left corner
        ])

    # -------- Iterative 'Reverse' Projection with Undistortion ---------- #
    # Converts UV pixel coordinates to undistorted XYZ local image coordinates
    # Construct intrinsic camera matrix K and compute its inverse
    u0 = image_width // 2 + cx
    v0 = image_height // 2 + cy
    k = np.array([
        [f + b1,  b2,  u0],
        [0,      f,    v0],
        [0,      0,     1]
    ])
    k_inv = np.linalg.inv(k)

    # Define full image extent corners if projecting entire image rather than a box
    full_extent = np.array([
        [image_width // 2 + cx, image_height // 2 + cy],  # image center with offsets
        [0, 0],                                           # top-left
        [image_width, 0],                                 # top-right
        [image_width, image_height],                       # bottom-right
        [0, image_height]                                  # bottom-left
    ])

    def iterate_undistort(points):
        """
        Iteratively corrects for radial and tangential distortion to estimate undistorted image coordinates.

        Args:
            points (np.ndarray): Array of distorted pixel points, shape (N, 2).

        Returns:
            x (np.ndarray): Undistorted x-coordinates in normalized image space.
            y (np.ndarray): Undistorted y-coordinates in normalized image space.
        """
        points = points.reshape(-1, 2)
        # Convert to homogeneous coordinates for intrinsic matrix inversion
        points_hom = np.hstack([points, np.ones((len(points), 1))])

        # Apply inverse intrinsic matrix
        xy_prime = k_inv.dot(points_hom.T)[:2, :]

        # Initial radius from principal point
        r = np.sqrt(xy_prime[0]**2 + xy_prime[1]**2)

        # Initial undistorted coordinates estimate (x_0, y_0)
        x = (xy_prime[0] - xy_prime[0] * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p1 * (r**2 + 2 * xy_prime[0]**2) - 2 * p2 * xy_prime[0] * xy_prime[1])
        y = (xy_prime[1] - xy_prime[1] * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p2 * (r**2 + 2 * xy_prime[1]**2) - 2 * p1 * xy_prime[0] * xy_prime[1])

        # Iterate to refine undistorted coordinates
        for _ in range(100):
            r = np.sqrt(x**2 + y**2)
            x = (xy_prime[0] - x * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p1 * (r**2 + 2 * x**2) - 2 * p2 * x * y)
            y = (xy_prime[1] - y * (k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) - p2 * (r**2 + 2 * y**2) - 2 * p1 * x * y)

        return x, y

    # Undistort input points and full image corners
    x, y = iterate_undistort(points)
    xf, yf = iterate_undistort(full_extent)

    # Compute min and max of full extent undistorted coords
    xmin, xmax = min(xf), max(xf)
    ymin, ymax = min(yf), max(yf)

    # Scale undistorted full extent coords to sensor size in meters (sw, sh)
    xf_m = (xf - xmin) / (xmax - xmin) * sw  # full extent scaled x coords in meters
    yf_m = (yf - ymin) / (ymax - ymin) * sh  # full extent scaled y coords in meters

    # Scale undistorted target points to sensor size in meters
    x_m = (x - xmin) / (xmax - xmin) * sw    # target points scaled x coords in meters
    y_m = (y - ymin) / (ymax - ymin) * sh    # target points scaled y coords in meters

    # ---------- Depth estimation through ray tracing ------------ #
    # Trace a ray from sensor location through centroid to DSM to estimate depth.
    # Method:
    # 1. Compute d' from known focal length (f) and xy' distance (distance from centroid to principal point in sensor space).
    # 2. Increment curr_Zc (height), compute d by similarity, project centroid to 3D image coords.
    # 3. Convert 3D image coords to UTM coords, use these to index DSM for actual height.
    # 4. If difference between estimated height and DSM height is within tolerance, stop; else, repeat.

    # +-xy'-+                    [sensor plane]
    #  \    |
    # d'\   | f
    #    \  |
    #     \ |
    #      \|
    # ------+------              [lens]
    #       |\
    #       | \
    #       |  \
    #       |   \
    #curr_Zc|    \  d (depth)     NOTE: curr_Zc (depth of principal point) roughly equal to curr_H (height above point in 3D world space)
    #       |     \                     for nadir images. For oblique, need to compute curr_H from curr d and 3d world distance b/n camera center and polygon centroid
    #       |      \
    #       |       \
    #       +---xy---+           [plane at Z=curr_H in image local coords]

    # Variables reference in diagram:
    #   - d' is distance in sensor plane to lens.
    #   - curr_Zc approximates height above ground (curr_H) for nadir images.
    #   - For oblique images, curr_H must be computed differently.

    # Compute xy' (distance from principal point in meters) and d' (hypotenuse in meters)
    undist_lengths_m = np.sqrt((x_m - xf_m[0])**2 + (y_m - yf_m[0])**2)
    d_dash_m = np.sqrt(undist_lengths_m**2 + focal**2)

    # Initialize variables
    del_Z = coarse_del_Z  # coarse height increment (meters)
    curr_Zc = 1  # starting estimate for camera height above ground (meters)

    dsm = rasterio.open(dsm_path)
    while True:
        # Increment height estimate
        curr_Zc += del_Z
        # Compute current ray length d using similarity ratio (meters)
        curr_d = curr_Zc * d_dash_m[0] / focal  # [0] for centroid

        # Compute 3D local image coordinates of centroid (meters)
        Xc = -x[0] * curr_d  # negative x due to coordinate system
        Yc = y[0] * curr_d

        # Convert local 3D coords to global UTM coords using rotation matrix m
        points = np.vstack([Xc, Yc, curr_Zc]).T  # shape (1, 3)
        curr_poly = points.dot(m.T)  # rotate points
        curr_poly = curr_poly[:, :2] / curr_poly[:, 2:3]  # perspective division
        curr_poly = -curr_d * curr_poly + np.array([X0, Y0])  # translate to global coords

        # Calculate horizontal squared distance in UTM between camera center and projected point
        dist_m_sq = (X0 - curr_poly[0][0])**2 + (Y0 - curr_poly[0][1])**2
        # Compute vertical height above ground from Pythagoras theorem
        curr_H = np.sqrt(curr_d**2 - dist_m_sq)

        # Convert UTM coordinates to pixel indices in DSM raster
        py, px = dsm.index(curr_poly[0][0], curr_poly[0][1])

        # Extract DSM values in a small window around pixel indices (bounded by 'bound')
        slice_ = dsm_arr[py - bound:py + bound, px - bound:px + bound]
        # Compute median DSM elevation ignoring zero or invalid values
        dsm_value = np.median(slice_[slice_ > 0])

        if np.isnan(dsm_value):
            # No valid DSM data found at projected location
            print('NAN in DSM!')
            return None, None

        # Calculate vertical distance from drone altitude to DSM surface
        computed_del_H = Z0 - dsm_value  # Z0 and DSM elevations are absolute altitudes (meters)
        diff = computed_del_H - curr_H  # difference between DSM height and current height estimate

        if abs(diff) < fine_tolerance:
            # Depth estimate converged within tolerance, stop loop
            break

        elif diff < 0:
            # Ray overshot DSM (inside terrain), raise error and abort
            print(f"Computed H => {computed_del_H}, Actual => {curr_H}")
            print("Error! Too coarse. Reduce height increment del_H or increase tolerance, and try again")
            return None, None

        elif diff < coarse_tolerance:
            # Difference small but not within fine tolerance; reduce height increment for finer approximation
            del_Z = fine_del_H

    # -------------- Actual Projection -------------- #
    # Compute depths for all points using the estimated depth of centroid (curr_Zc).
    # Using relation: corrected_point_depths = curr_Zc * d_dash_m / focal
    # Project points from local image coordinates to global UTM coordinates.
    # Calculate corrected depths for all points (array)
    corrected_point_depths = curr_Zc * d_dash_m / focal

    # Compute local 3D coordinates in image space
    X = x * corrected_point_depths  # negative sign based on coordinate system
    Y = y * corrected_point_depths

    # Create array of focal lengths (curr_Zc) for all points (depth coordinate)
    f_col = np.full(len(X), curr_Zc)

    # Stack local coordinates into Nx3 array (X, Y, Z)
    points = np.vstack([X, Y, f_col]).T

    # Convert local coordinates to global UTM coordinates
    curr_poly = points.dot(m.T)  # rotate points
    curr_poly = curr_poly[:, :2] / curr_poly[:, 2:3]  # perspective division
    curr_poly = -corrected_point_depths.reshape(-1, 1) * curr_poly + np.array([X0, Y0])  # translate to global coords

    # Calculate horizontal distance from image center to first projected point (for shapefile batching)
    dist = np.sqrt((X0 - curr_poly[0][0])**2 + (Y0 - curr_poly[0][1])**2)

    return curr_poly, dist


def batch_frame2gis(cfg):
    # Get parameters from config yml
    img_dir = cfg['paths']['batch_cfg']['img_dir']
    dsm_path = cfg['paths']['dsm_path']
    points_csv = cfg['paths']['batch_cfg']['img2gis']['points_csv']

    # Read points_csv
    points_df = pd.read_csv(points_csv)

    # Read DSM
    with rasterio.open(dsm_path) as src:
        dsm_arr = src.read(1)
        try:
            dsm_epsg = int(str(src.crs)[5:])
        except:
            exit("Error: DSM does not seem to have CRS information. Check and try again")

    # Make output dir if it does not exist
    output_file = f"{cfg['paths']['batch_cfg']['img2gis']['out_shp']}"
    print(f'Outputting results to {output_file}...')
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exit_ok=True)

    # Define default field types and widths for specific column names
    defaults = {
        'id': ('N', '20'),                   # 'id' as numeric with width 20
        'img_path': ('C', '160'),           # 'img_path' as character with width 160
        'box_id': ('C', '20'),              # 'box_id' as character with width 20
        'prediction': ('C', '20'),          # 'prediction' as character with width 20
        'confidence': ('F', '20', 4)        # 'confidence' as float with width 20 and 4 decimal places
    }

    # Initialize list to store formatted column definitions
    cols = []

    # Iterate through DataFrame columns and assign appropriate type definitions
    for c in points_df.columns:
        if c in defaults:
            # Use predefined type and width if available
            cols.append((c, *defaults[c]))
        else:
            # Default to character type with width 80 if not specified
            cols.append((c, 'C', '80'))

    # Create a shapefile writer for POLYGON geometry
    w = shapefile.Writer(output_file, shapefile.POLYGON)

    # Add fields to the shapefile based on column definitions
    for col in cols:
        if col[0] == 'geometry':
            # Rename 'geometry' to avoid reserved field name conflict
            w.field('img_coords', 'C', '80')
        else:
            # Add all other fields as defined
            w.field(*col)

    # Add an extra field for distance to image center with 3 decimal places
    w.field('distance_to_img_center', 'F', decimal=3)

    # Open a log file for writing skipped entries
    open('output/log_skipped.txt', 'w')

    # Initialize a dictionary to store skipped box_ids per image
    skipped_dict = defaultdict(list)

    # Initialize counters for projected and skipped detections
    projected = 0
    skipped = 0

    # Loop over each row in the DataFrame (each row corresponds to a box in an image)
    for _, row in tqdm(points_df.iterrows(), total=len(points_df)):

        # Convert 'geometry' field from string to NumPy array of coordinates
        points = np.array(literal_eval(row['geometry']), dtype=np.uint16)

        # Compute bounding box limits
        xlim = (np.amin(points[:, 0]), np.amax(points[:, 0]))
        ylim = (np.amin(points[:, 1]), np.amax(points[:, 1]))

        # Compute centroid of the bounding box
        center = np.array([(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2]).reshape(-1, 2)

        # Prepend centroid to the list of polygon points
        points = np.vstack([center, points])

        # Resolve full path to image
        if os.path.exists(row['img_path']):
            img_path = f'{row["img_path"]}'
        else:
            img_path = f'{img_dir}/{row["img_path"]}'

        # Project image-space polygon to geographic coordinates
        curr_poly, dist = img2gis(img_path, points, dsm_path, dsm_arr, cfg)

        if curr_poly is not None:
            # Write the projected polygon to the shapefile
            w.poly([curr_poly.tolist()[1:]])

            # Write associated attributes including distance to center and a placeholder 'Polygon' type
            w.record(*row, dist, 'Polygon')
            projected += 1
        else:
            # Log skipped projections by image path and box ID
            skipped_dict[row['img_path']].append(row['box_id'])
            skipped += 1

    # Close the shapefile writer
    w.close()

    # Create the .prj file with the appropriate WKT projection string
    prj = open(f'{output_file}.prj', 'w')
    wkt = getWKT_PRJ(dsm_epsg)
    prj.write(wkt)
    prj.close()

    # Write log of skipped boxes to file
    with open('output/skipped_log.txt', 'w') as f:
        f.write('img_name, total_skipped, boxIDs\n')
        for k, v in skipped_dict.items():
            f.write(f'{k}, {len(v)}, {v}\n')
        f.write(f'\n(Total - Skipped {skipped} boxes out of {projected + skipped})\n')

    # Print summary of the projection process
    print('\n=================================')
    print('Batch projection complete!')
    print(f'Projected: {projected}')
    print(f'Skipped: {skipped}')
    print(f'(Total - {projected + skipped} boxes in {points_df["img_path"].nunique()} images)')
    print(f'\nSaved to {output_file}.shp')
    print('=================================\n')


def build_m30t_wide_cfg_dict():
    cfg = {}
    # # Set img2gis params
    img2gis_dict = {}
    img2gis_dict['aligned_cameras_csv_path'] = None
    img2gis_dict['coarse_height_increment'] = 0.25
    img2gis_dict['fine_height_increment'] = 0.01
    img2gis_dict['coarse_tolerance'] = 1
    img2gis_dict['fine_tolerance'] = 0.1
    img2gis_dict['boundary'] = 10
    cfg['img2gis'] = img2gis_dict

    # # Set calibration params
    calibration_dict = {}

    # %% Intrinsic
    # Focal lengths
    # pixel_size_microns = sensor_width_mm / image_width_px
    # Note that this should come from the IMAGE (camera) parameters!
    calibration_dict['f'] = 2100  # Focal length (pixels), 1000 * focal_length_mm / pixel_size_microns

    # Principal offset points in pixels
    calibration_dict['cx'] = 0  # Pixels
    calibration_dict['cy'] = 0  # Pixels

    # Sensor dimensions (in metres)
    calibration_dict['sw'] = 8 / 1000
    calibration_dict['sh'] = 4.8 / 1000

    # %% Survey-dependent
    # Radial distortion coefficients (optional)
    calibration_dict['k1'] = 0.129238
    calibration_dict['k2'] = -0.34149
    calibration_dict['k3'] = 0.207202
    calibration_dict['k4'] = 0

    # Tangential distortion coefficients (optional)
    calibration_dict['p1'] = 0.000384231
    calibration_dict['p2'] = 0.0204863

    # Skew coefficiants (optional)
    calibration_dict['b1'] = 675
    calibration_dict['b2'] = 0
    cfg['camera_calibration'] = calibration_dict

    # # Set cameras params
    cameras_dict = {'aligned_cameras_csv_path': None}
    cfg['paths'] = cameras_dict
    return cfg


def build_m30t_thermal_cfg_dict():
    cfg = {}
    # # Set img2gis params
    img2gis_dict = {}
    img2gis_dict['aligned_cameras_csv_path'] = None
    img2gis_dict['coarse_height_increment'] = 0.25
    img2gis_dict['fine_height_increment'] = 0.01
    img2gis_dict['coarse_tolerance'] = 1
    img2gis_dict['fine_tolerance'] = 0.1
    img2gis_dict['boundary'] = 10
    cfg['img2gis'] = img2gis_dict

    # # Set calibration params
    calibration_dict = {}

    # %% Intrinsic
    # Focal lengths
    # pixel_size_microns = sensor_width_mm / image_width_px
    # Note that this should come from the IMAGE (camera) parameters!
    calibration_dict['f'] = 5228.54052  # Focal length (pixels), 1000 * focal_length_mm / pixel_size_microns

    # Principal offset points in pixels
    calibration_dict['cx'] = 0  # Pixels
    calibration_dict['cy'] = 0  # Pixels

    # Sensor dimensions (in metres)
    calibration_dict['sw'] = 7.68 / 1000
    calibration_dict['sh'] = 6.144 / 1000

    # %% Survey-dependent
    # Radial distortion coefficients (optional)
    calibration_dict['k1'] = -0.247124
    calibration_dict['k2'] = 0
    calibration_dict['k3'] = 0
    calibration_dict['k4'] = 0

    # Tangential distortion coefficients (optional)
    calibration_dict['p1'] = 0.00665011
    calibration_dict['p2'] = -0.0328982

    # Skew coefficiants (optional)
    calibration_dict['b1'] = 0
    calibration_dict['b2'] = 0
    cfg['camera_calibration'] = calibration_dict

    # # Set cameras params
    cameras_dict = {'aligned_cameras_csv_path': None}
    cfg['paths'] = cameras_dict
    return cfg


def save_coords_as_geojson(coords_array, output_path, crs_epsg=32611):
    """
    Saves a list of coordinates as a GeoJSON file.

    Args:
        coords_array (np.ndarray): Array of shape (N, 2) with [easting, northing] coordinates.
        output_path (str): File path to save the GeoJSON.
        crs_epsg (int): EPSG code for the coordinate reference system. Defaults to 32612 (UTM Zone 12N).

    Returns:
        None
    """
    # Convert each coordinate pair to a Shapely Point object
    point_geometries = [Point(xy) for xy in coords_array]

    # Create a GeoDataFrame with the point geometries and CRS
    gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=f'EPSG:{crs_epsg}')

    # Save the GeoDataFrame to a GeoJSON file
    gdf.to_file(output_path, driver='GeoJSON')

# TODOs:
    # Try better DSM
    # Try thermal -- ok
    ## Fix pitch adjustment?
    # Improve camera calib

if __name__ == '__main__':
    os.chdir(r'D:\!Research\01 - Python\DroneSurveyMapper\DJI_20250426143054_0001_T')
    img_path = r'D:\!Research\01 - Python\DroneSurveyMapper\DJI_20250426143054_0001_T\frame_00705.jpg'
    points = None
    dsm_path = r"D:\!Research\01 - Python\DroneSurveyMapper\test_dsm_beaumont.tif"
    cfg = build_m30t_thermal_cfg_dict()
    curr_poly, dist = img2gis(img_path, points, dsm_path, cfg, full_image=True)
    save_coords_as_geojson(curr_poly, 'frame_00705.geojson')
