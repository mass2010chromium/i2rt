import argparse
import json
import pickle

import cv2
import cv2.aruco as aruco
# Used for 3d calculations.
import mujoco
from motionlib import so3, se3
import numpy as np
import scipy.optimize as opt

from calibration_utils import (
    unflatten_pose7,
    unflatten_pose6,
    detect_aruco_corner_pixels,
    read_intrinsics
)

parser = argparse.ArgumentParser(description="Calibrate the extrinsics of a "
    + "camera using the chess board poster")
parser.add_argument("path", type=str, help="directory containing the "
    + "calibration images")
parser.add_argument("--skip", action="store_true", help="skip image "
    + "extraction, use cached data")
args = parser.parse_args()

cache_name = "tmp_data2.p"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

DEPTH_RESOLUTION = 1000
wrist_matrix = read_intrinsics("wrist_intrinsics.json")
static_matrix = read_intrinsics("static_intrinsics.json")
wrist_unproject = np.linalg.inv(wrist_matrix)
with open("wrist_calibration.json", "r") as calib_file:
    wrist_to_ee = np.array(json.load(calib_file)['camera_offset'])

# Arrays to store object points and image points from all the images.
if not args.skip:
    img_points = [] # 2d points in image plane.
    world_points = []
    img_indices = []

    import sys
    sys.path.insert(0, "..")
    from synchronize_data import SyncData
    data_loader = SyncData(args.path, action_mode='ee')
    sync_data = data_loader.get_synchronized_data()
    ee_poses = sync_data['state']

    first_episode_start = sync_data['event_times'][sync_data['first_episode']]
    start_index = np.searchsorted(sync_data['image_timestamps'], first_episode_start)

    for i in range(start_index, len(ee_poses)):
        static_image_data = data_loader.get_images(i, key="cam1")
        wrist_image_data = data_loader.get_images(i, key="cam2")
        static_img = static_image_data['rgb']
        wrist_img = wrist_image_data['rgb']
        wrist_depth = wrist_image_data['depth']
        if static_img is None or wrist_img is None or wrist_depth is None:
            print("Read error?", i)
            continue

        # Find the chess board corners
        wrist_corners, _ = detect_aruco_corner_pixels(wrist_img, aruco.DICT_5X5_100)
        static_corners, _ = detect_aruco_corner_pixels(static_img, aruco.DICT_5X5_100)
        if wrist_corners is not None and static_corners is not None:
            # Strip out gripper pos
            ee_pose = unflatten_pose7(ee_poses[i][:-1])
            world_point_set = []
            ok = True
            depth_vals = []
            for corner in wrist_corners[0][0]:
                depth_value = wrist_depth[int(corner[1]), int(corner[0])] / DEPTH_RESOLUTION
                if depth_value == 0:
                    ok = False
                    break
                point_3d_local = [*(wrist_unproject @ [*(corner * depth_value), depth_value]), 1]
                world_point = ee_pose @ (wrist_to_ee @ point_3d_local)
                world_point_set.append(world_point)
                depth_vals.append(depth_value)
            if not ok:
                continue
            if max(depth_vals) - min(depth_vals) > 0.2:
                # Impossible depths
                continue
            print(np.array(world_point_set))
            print(depth_vals)

            # Make homogenous coordinates
            world_points.append(np.array(world_point_set).T)
            img_points.append(np.array(static_corners[0][0]))
            img_indices.append(i)
    print(f"Found {len(img_points)} good images")
    with open(cache_name, "wb") as tmp_file:
        data = (world_points, img_points, img_indices)
        pickle.dump(data, tmp_file)
else:
    with open(cache_name, "rb") as tmp_file:
        world_points, img_points, img_indices = pickle.load(tmp_file)

def camera_objective(cam_pose6):
    err = 0
    camera_transform_inv = unflatten_pose6(cam_pose6)
    for point_3d, corners in zip(world_points, img_points):
        corner_local = camera_transform_inv @ point_3d
        projected_points = static_matrix[:2, :2] @ corner_local[:2]
        projected_pixels = (projected_points / corner_local[2:3, :]) + static_matrix[:2, 2:3]
        pixel_error = corners - projected_pixels.T

        err += np.sum(pixel_error.flatten() ** 2)
        #err += np.sum(np.abs(pixel_error.flatten()))
    return err / len(img_points)

#init = np.array([
#    0, 0, 0, 0, 0, 0,
#])
#init = np.random.random(6)
init = np.array(
#[0.8207672320662143, 0.10301531511304654, -0.38909392017161504, -0.29850071790340943, 0.5128450798178066, 0.4005285709213732]
#[*np.random.random(3), 0, -1.0, 0.4]
[1.9563640803448528, -0.3391761465201263, 0.34956585830004594, -0.4373495116941391, -0.00974611775268271, 0.970540709048741]
)
print("init:", init.tolist())
res = opt.minimize(camera_objective, init, method='Nelder-Mead')
#res = opt.minimize(camera_objective, init, method='BFGS')
print("result:", res.x.tolist())
print("result mse:", res.fun)
camera_transform = np.linalg.inv(unflatten_pose6(res.x))
print("camera pose estimate:")
print(camera_transform)
with open("static_calibration.json", "w") as outfile:
    json.dump({
        'camera_offset': camera_transform.tolist(),
    }, outfile, indent=2)
