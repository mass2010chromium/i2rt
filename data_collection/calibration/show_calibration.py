import argparse
import json
import pickle

import cv2
import cv2.aruco as aruco
# Used for 3d calculations.
import numpy as np

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
args = parser.parse_args()

cache_name = "tmp_data.p"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
aruco_points = get_aruco_marker_grid(0.1)

import sys
sys.path.insert(0, "..")
from synchronize_data import SyncData
data_loader = SyncData(args.path, action_mode='ee')
# Arrays to store object points and image points from all the images.
with open(cache_name, "rb") as tmp_file:
    calib_poses, imgpoints, img_indices = pickle.load(tmp_file)

camera_matrix = read_intrinsics("wrist_intrinsics.json")

import matplotlib.pyplot as plt
import mediapy
x = \
[0.3512978655989486, -0.3840093118461824, 1.5479907273094384, 0.04785065740431616, 0.061766666427187686, 0.09881165220844043, 0.012278261643244286, 0.021125952967118895, 0.09961014423885398, 0.5040460194006895, -0.09343809626642355, 0.012996279646771452]
#[0.3551977551256492, -0.3764742824790426, 1.5455325766138905, 0.046208548344893366, 0.057865950424843465, 0.09644041701522466, 0.011938837480053166, 0.026595665270252637, 0.0920827714283896, 0.5000109741583588, -0.09169623616764924, 0.013702452803171866]
def show_calibration(x):

    video_frames = []
    err = 0
    cam_pose6 = x[:6]
    marker_pose6 = x[6:]
    camera_transform_inv = unflatten_pose6(cam_pose6)
    marker_transform = unflatten_pose6(marker_pose6)
    corner_world = marker_transform @ aruco_points
    for pose_inv, corners, i in zip(calib_poses, imgpoints, img_indices):
        full_inverse = camera_transform_inv @ pose_inv
        corner_local = full_inverse @ corner_world
        projected_points = camera_matrix[:2, :2] @ corner_local[:2]
        projected_pixels = (projected_points / corner_local[2:3, :]) + camera_matrix[:2, 2:3]
        pixel_error = corners - projected_pixels.T

        cur_err = np.sum(pixel_error.flatten() ** 2)
        err += cur_err

        if cur_err > 0:
            image_data = data_loader.get_images(i, key="cam2")
            img = cv2.polylines(image_data['rgb'], [np.int32(corners)], isClosed=False, color=(255, 0, 0), thickness=2)
            img = cv2.circle(img, center=np.int32(corners[0]), radius=4, color=(255, 0, 0), thickness=-1)
            img = cv2.polylines(img, [np.int32(projected_pixels.T)], isClosed=False, color=(0, 0, 255), thickness=2)
            img = cv2.circle(img, center=np.int32(projected_pixels.T[0]), radius=4, color=(0, 0, 255), thickness=-1)
            img = cv2.putText(img, f"{i}", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            video_frames.append(img)
            #print(np.linalg.inv(pose_inv))
            #print(np.linalg.inv(full_inverse))
            #input(i)

    mediapy.write_video("out.mp4", video_frames)
    return err / len(calib_poses)

show_calibration(x)
