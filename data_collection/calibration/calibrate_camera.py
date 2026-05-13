import argparse
import json

import cv2
import cv2.aruco as aruco
# Used for 3d calculations.
import mujoco
from motionlib import so3, se3
import numpy as np
import scipy.optimize as opt

from calibration_utils import detect_aruco_corner_pixels

parser = argparse.ArgumentParser(description="Calibrate the extrinsics of a "
    + "camera using the chess board poster")
parser.add_argument("path", type=str, help="directory containing the "
    + "calibration images")
parser.add_argument("--width", type=int, default=8, help="width of the chess "
    + "board, number of lattice points inside the border of outer squares")
parser.add_argument("--height", type=int, default=5, help="height of the chess "
    + "board, counted same way as width")
parser.add_argument("--skip", action="store_true", help="skip image "
    + "extraction, use cached data")
args = parser.parse_args()

def unflatten_pose7(pose_7d):
    pose = np.eye(4, dtype=np.float32)
    rot_flat = np.empty(9, dtype=np.float32)
    mujoco.mju_quat2Mat(rot_flat_ pose_7d[:4])
    pose[:3, :3] = rot_flat.reshape((3, 3))
    pose[:3, 3] = pose_7d[4:]
    return pose

def unflatten_pose6(pose_6d):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = so3.matrix(so3.from_moment(pose_6d[:3]))
    pose[:3, 3] = pose_6d[3:]
    return pose

cache_name = "tmp_data.p"
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
width = args.width
height = args.height

# Arrays to store object points and image points from all the images.
calib_poses = []
imgpoints = [] # 2d points in image plane.
images = sorted(glob.glob(f'{args.path}/*.png'))
print(images)
size_img = cv.imread(images[0])
h, w, _ = size_img.shape
if not args.skip:
    from synchronize_data import SyncData
    data_loader = SyncData(out_dir, action_mode=ACTION_MODE)
    sync_data = data_loader.get_synchronized_data()
    ee_poses = sync_data['state']

    for i in range(len(ee_poses)):
        image_data = data_loader.get_images(i, key="cam2")
        img = image_data['rgb']
        depth = image_data['depth']

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        corners, ids = detect_aruco_corner_pixels(img, aruco.DICT_5X5_100)
        if corners is not None:
            imgpoints.append(np.array(corners))
            calib_poses.append(np.linalg.inv(unflatten_pose7(ee_poses[i])))
    print(f"Found {len(objpoints)} good images")
    with open(cache_name, "wb") as tmp_file:
        data = (calib_poses, imgpoints)
        pickle.dump(data, tmp_file)
else:
    with open(cache_name, "rb") as tmp_file:
        calib_poses, imgpoints = pickle.load(tmp_file)

with open("intrinsics.json", "r") as intrinsics_file:
    intr_data = json.load(intrinsics_file)
    camera_matrix = np.array([
        [intr_data['fx'], 0, 0, intr_data['cx']],
        [0, intr_data['fy'], 0, intr_data['cy']],
        [0, 0, 1, 0]
    ])

# Aruco points, homogenous, column vectors. clockwise
aruco_points = np.array([
    [0, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0.1, 0.1, 0, 1],
    [0.1, 0, 0, 1]
]).T
def camera_objective(x):
    err = 0
    cam_pose6 = x[:6]
    marker_pose6 = x[6:]
    camera_transform_inv = unflatten_pose6(x)
    marker_transform = unflatten_pose6(x)
    corner_world = marker_transform @ aruco_points
    for pose_inv, corners in zip(calib_poses, imgpoints):
        full_inverse = camera_transform_inv @ pose_inv
        corner_local = full_inverse @ corner_world
        projected_points = camera_matrix @ corner_local
        projected_points /= projected_points[2:3, :]
        pixel_error = corners - projected_points[:2].T

        err += np.sum(pixel_error.flatten() ** 2)
    return err / len(calib_poses)

res = opt.minimize(camera_objective, np.zeros(12), method='Nelder-Mead')
print("result mse:", res.fun)
camera_transform = np.linalg.inv(unflatten_pose6(res.x[:6]))
marker_transform = unflatten_pose6(res.x[6:])
print("marker pose estimate:")
print(marker_transform)
with open("calibration.json", "w") as outfile:
    json.dump({
        'camera_offset': camera_transform.tolist()
    }, outfile)
