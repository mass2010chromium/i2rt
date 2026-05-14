import json

import cv2
import cv2.aruco as aruco
import numpy as np
from motionlib import so3, se3
import mujoco

def detect_aruco_corner_pixels(pic: np.ndarray, dictionary: int) -> tuple[list[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    # parameters.adaptiveThreshWinSizeMin = 3  # default 3
    # parameters.adaptiveThreshWinSizeMax = 700  # default 23
    # parameters.adaptiveThreshWinSizeStep = 5  # default 10
    # parameters.adaptiveThreshConstant = 10      # default 7
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, detected_ids, _ = detector.detectMarkers(gray)
    if detected_ids is None:
        return None, None
    return corners, detected_ids.flatten()

def read_intrinsics(fname):
    with open(fname, "r") as intrinsics_file:
        intr_data = json.load(intrinsics_file)
        return np.array([
            [intr_data['fx'], 0, intr_data['cx']],
            [0, intr_data['fy'], intr_data['cy']],
            [0, 0, 1]
        ])

def unflatten_pose7(pose_7d):
    pose = np.eye(4, dtype=np.float32)
    rot_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_flat, pose_7d[:4])
    pose[:3, :3] = rot_flat.reshape((3, 3))
    pose[:3, 3] = pose_7d[4:]
    return pose

def unflatten_pose6(pose_6d):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = so3.matrix(so3.from_moment(pose_6d[:3]))
    pose[:3, 3] = pose_6d[3:]
    return pose

# Aruco points, homogenous, column vectors. clockwise
def get_aruco_marker_grid(size):
    return np.array([
        [0, 0, 0, 1],
        [0, size, 0, 1],
        [size, size, 0, 1],
        [size, 0, 0, 1]
    ]).T
