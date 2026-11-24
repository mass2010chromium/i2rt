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
    read_intrinsics,
    get_aruco_marker_grid
)

parser = argparse.ArgumentParser(description="Calibrate the extrinsics of a "
    + "camera using the chess board poster")
parser.add_argument("path", type=str, help="directory containing the "
    + "calibration images")
parser.add_argument("--skip", action="store_true", help="skip image "
    + "extraction, use cached data")
args = parser.parse_args()

cache_name = "tmp_data.p"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
aruco_points = get_aruco_marker_grid(0.1)

# Arrays to store object points and image points from all the images.
if not args.skip:
    calib_poses = []
    imgpoints = [] # 2d points in image plane.
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
        image_data = data_loader.get_images(i, key="cam2")
        img = image_data['rgb']
        if img is None:
            print("Read error?", i)
            continue

        # Find the chess board corners
        corners, ids = detect_aruco_corner_pixels(img, aruco.DICT_5X5_100)
        if corners is not None:
            imgpoints.append(np.array(corners[0][0]))
            # Strip out gripper pose
            calib_poses.append(np.linalg.inv(unflatten_pose7(ee_poses[i][:-1])))
            img_indices.append(i)
    print(f"Found {len(imgpoints)} good images")
    with open(cache_name, "wb") as tmp_file:
        data = (calib_poses, imgpoints, img_indices)
        pickle.dump(data, tmp_file)
else:
    with open(cache_name, "rb") as tmp_file:
        calib_poses, imgpoints, img_indices = pickle.load(tmp_file)
        #calib_poses = calib_poses[::5]
        #imgpoints = imgpoints[::5]
        #img_indices = img_indices[::5]

camera_matrix = read_intrinsics("wrist_intrinsics.json")

def camera_objective(x):
    err = 0
    cam_pose6 = x[:6]
    marker_pose6 = x[6:]
    camera_transform_inv = unflatten_pose6(cam_pose6)
    marker_transform = unflatten_pose6(marker_pose6)
    corner_world = marker_transform @ aruco_points
    for pose_inv, corners in zip(calib_poses, imgpoints):
        full_inverse = camera_transform_inv @ pose_inv
        corner_local = full_inverse @ corner_world
        projected_points = camera_matrix[:2, :2] @ corner_local[:2]
        projected_pixels = (projected_points / corner_local[2:3, :]) + camera_matrix[:2, 2:3]
        pixel_error = corners - projected_pixels.T

        #err += np.sum(pixel_error.flatten() ** 2)
        err += np.sum(np.abs(pixel_error.flatten()))
    return err / len(calib_poses)

#init = np.array([
#    0, 0, 0, 0, 0, 0,
#    0, 0, 0, 0, 0, 0
#])
#init = np.random.random(12)
init = np.array(
#[0.13975432, 0.45844857, 0.86702325, 0.69064999, 0.92876567, 0.57363849, 0.79978165, 0.29575385, 0.54335649, 0.9913072, 0.42685584, 0.26231438]
#[*np.random.rand(3), -0.4295147927002241, -0.16765394881929507, 0.23450951204122775, *np.random.rand(3), 0.5135482004956717, -0.065461144871344, 0.022320151182080118]
#[0.1438057799723814, -0.03230043838722763, 1.5733110314714933, 0.04854074860233541, -0.03025804526595758, 0.10343952753209548, -0.09862043251500009, 0.010013866184116918, -0.11001464341389185, 0.49356909145711353, -0.057834328170495414, 0.059426430461776036]
#[0.3580238388454602, -0.3612338730569752, 1.5451919544518251, 0.04315524895174136, 0.05592051103575907, 0.09865109044577045, 0.009642905504183286, -0.016169068080906762, 0.08863373443901645, 0.5005949276227606, -0.09146226396073291, 0.012387498586328184]
#[0.3551977551256492, -0.3764742824790426, 1.5455325766138905, 0.046208548344893366, 0.057865950424843465, 0.09644041701522466, 0.011938837480053166, 0.026595665270252637, 0.0920827714283896, 0.5000109741583588, -0.09169623616764924, 0.013702452803171866]

# These are from lab
#[0.05347030023934203, -0.5645547846628576, 1.2373876598660245, 0.13818323985113815, -0.13945604057945193, 0.326078328320531, 0.08776531682752677, 0.053365495970279384, 1.2514858521157897, 0.5546333733898847, -0.08482809766825504, -0.19954994195321185]
#[0.37403610720046476, -0.3263250025036125, 1.4995023103328005, 0.030080598752324175, 0.04574646549099784, 0.0962738229093745, 0.16982012075056097, 0.0076736913751554725, 2.0702941327291216, 0.6449737474751749, -0.013085690598788592, 0.0025300678440738525]
#[0.3492301187596345, -0.3230551403616714, 1.5206218875631334, 0.030937531347646624, 0.039489790708406636, 0.09265507591397365, 0.02837200681061306, 0.008031702872215187, 2.0907551871531576, 0.6525101482316862, -0.016336529749292082, 0.004551108638853668]
#[0.3487867805466487, -0.3154826639032744, 1.535131664665961, 0.02950591378538974, 0.040706357248236236, 0.09222859236743727, 0.013341990845687794, 0.044425976847871224, 2.094123809121327, 0.6544649382486646, -0.01622030189182641, 0.008720210955375337]
[0.34217761562855736, -0.33548644944079886, 1.53388909545589, 0.035625729334049595, 0.04214539291132961, 0.09299029159668006, 0.01071257873724803, 0.07991588394945418, 2.0730719829981767, 0.6522239619059218, -0.01808912699801051, 0.007039398893692926]
)
#init[0:6] = np.random.rand(6)
#init[6:9] = np.random.rand(3)
print("init:", init.tolist())
res = opt.minimize(camera_objective, init, method='Nelder-Mead')
#res = opt.minimize(camera_objective, init, method='BFGS')
print("result:", res.x.tolist())
print("result mse:", res.fun)
camera_transform = np.linalg.inv(unflatten_pose6(res.x[:6]))
marker_transform = unflatten_pose6(res.x[6:])
print("marker pose estimate:")
print(marker_transform)
print("camera pose estimate:")
print(camera_transform)
with open("wrist_calibration.json", "w") as outfile:
    json.dump({
        'camera_offset': camera_transform.tolist(),
        'marker_pose': marker_transform.tolist()
    }, outfile, indent=2)
