import sys
import time

import numpy as np
import cv2
import einops
import mediapy

# SE3 maths
import mujoco
from motionlib import so3

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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

def get_6d_error(target_4x4, current_4x4):
    """
    Computes target - current, as a 6d difference vector (rot, pos).
    Assumes both matrices are in a shared (world) frame.
    Returns the result in the world frame. (involves rotating by current,
        since the difference computed is in the "current" frame.)
    """
    # Translation error
    pos_err = target_4x4[:3, 3] - current_4x4[:3, 3]

    # Rotation error
    q_curr = np.zeros(4)
    q_targ = np.zeros(4)
    mujoco.mju_mat2Quat(q_curr, current_4x4[:3, :3].flatten())
    mujoco.mju_mat2Quat(q_targ, target_4x4[:3, :3].flatten())

    rot_err = np.zeros(3)
    # Computes the 3D rotation vector that maps q_curr to q_targ
    mujoco.mju_subQuat(rot_err, q_targ, q_curr)

    return np.concatenate([current_4x4[:3, :3] @ rot_err, pos_err])

def write_text_to_frame(frame, text):
    frame = np.array(frame * 255.0, dtype=np.uint8, order='c')

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    margin = 8
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = max(margin, frame.shape[1] - text_w - margin)
    y = margin + text_h

    cv2.rectangle(
        frame,
        (max(0, x - 4), max(0, y - text_h - 4)),
        (min(frame.shape[1] - 1, x + text_w + 4), min(frame.shape[0] - 1, y + baseline + 4)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

ds_name = sys.argv[1]
repo_id = f"local/{ds_name}"

dataset = LeRobotDataset(repo_id, root=f"hf/{ds_name}")

out_dir = f"{ds_name}_fix_transforms"
out_dataset = LeRobotDataset.create(
    repo_id=f"local/{out_dir}",
    root=f"hf/{out_dir}",
    fps=30,
    robot_type="yam",
    image_writer_threads=4,
    image_writer_processes=4,
    features={
        "image": {
            "dtype": "video",
            "shape": (3, 480, 640),  # (Channels, Height, Width)
            "names": ["color"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (3, 480, 640),  # (Channels, Height, Width)
            "names": ["color"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8, ),
            #"names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            "names": ["q_1", "q_2", "q_3", "q_4", "x", "y", "z", "gripper"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7, ),
            "names": ["q_1", "q_2", "q_3", "q_4", "x", "y", "z", "gripper"],
        },
    },
)

i = 0
N = 30
t = time.monotonic()
episodes_counted = set()
task_dist = {}
gripper_min = 100
gripper_max = -100
while True:
    if i >= len(dataset):
        break
    img = dataset[i]['image']
    episode_idx = int(dataset[i]['episode_index'])
    gripper_min = min(gripper_min, dataset[i]['actions'][-1])
    gripper_max = max(gripper_max, dataset[i]['actions'][-1])
    episode_idx = int(dataset[i]['episode_index'])
    if episode_idx not in episodes_counted:
        task_idx = int(dataset[i]['task_index'])
        if task_idx not in task_dist:
            task_dist[task_idx] = 0
        task_dist[task_idx] += 1
        if len(episodes_counted) != 0:
            out_dataset.save_episode()
        print(f"Episode {len(episodes_counted)}:")
        episodes_counted.add(episode_idx)

    target_reconstruct = (dataset[i]['state'] + dataset[i]['actions'])[:-1]
    current_pose = unflatten_pose7(dataset[i]['state'][:-1])
    target_pose = unflatten_pose7(target_reconstruct)
    error_6d = get_6d_error(target_pose, current_pose)
    #delta_pose = unflatten_pose6(error_6d)
    #reconstruct_pose = np.eye(4)
    #reconstruct_pose[:3, :3] = delta_pose[:3, :3] @ current_pose[:3, :3]
    #reconstruct_pose[:3, 3] = delta_pose[:3, 3] + current_pose[:3, 3]
    #print(target_pose)
    #print(reconstruct_pose)
    #input()
    out_dataset.add_frame({
        "image": img,
        "wrist_image": dataset[i]['wrist_image'],
        "state": dataset[i]['state'],
        "actions": np.concatenate([error_6d, [dataset[i]['actions'][-1]]], dtype=np.float32),
        "task": dataset[i]['task']
    })

    i += 1
    if i % N == 0:
        t1 = time.monotonic()
        print(f"{N / (t1 - t)} fps")
        print(f"gripper limits: [{gripper_min}, {gripper_max}]")
        t = time.time()
out_dataset.save_episode()
