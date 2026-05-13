import gc
import glob
import json
import sys

import cv2
import numpy as np
import scipy
import torch
torch.set_grad_enabled(False)

import tqdm

out_dir = sys.argv[1]

from synchronize_data import SyncData

ACTION_MODE = "ee" #"joint"
data_loader = SyncData(out_dir, action_mode=ACTION_MODE)
sync_data = data_loader.get_synchronized_data()
image_timestamps = sync_data['image_timestamps']
states = sync_data['state']
targets = sync_data['target']
event_timestamps = sync_data['event_timestamps']
event_text = sync_data['events']
first_episode = sync_data['first_episoe']
n_episodes = sync_data['num_episodes']

print("Parsed metadata files")
print(f"{len(image_timestamps)} images, {n_episodes} episodes")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

#help(LeRobotDataset.create)
# 1. Define the dataset metadata and features
# 'observation.image' is for the camera feed
# 'observation.state' is for the vector (e.g., joint angles)
dataset = LeRobotDataset.create(
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
        "depth": {
            "dtype": "uint16",
            "shape": (480, 640),
            "names": ["raw"],
        },
        "wrist_depth": {
            "dtype": "uint16",
            "shape": (480, 640),
            "names": ["raw"],
        },
        "state": {
            "dtype": "float32",
            "shape": state_observation[0].shape,
            #"names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
            "names": ["q_1", "q_2", "q_3", "q_4", "x", "y", "z", "gripper"],
        },
        "actions": {
            "dtype": "float32",
            "shape": state_target[0].shape,
            "names": ["q_1", "q_2", "q_3", "q_4", "x", "y", "z", "gripper"],
        },
    },
)

first_episode_start = event_timestamps[first_episode]
video_index = np.searchsorted(image_timestamps, first_episode_start)
event_idx = first_episode
cur_time = image_timestamps[video_index]

import torchvision
resize_transform = torchvision.transforms.Resize((480, 640),
    interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
def read_image_torch(image_path):
    img = torchvision.io.decode_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
    return resize_transform(img)

episode_idx = 0
while True:
    episode_task = event_text[event_idx]

    # Scroll to the first `start_episode`
    while event_text[event_idx] != 'start_episode':
        # In case of spurious activations
        event_idx += 1
    # And also scroll the camera images forward
    event_start_time = event_timestamps[event_idx]
    while cur_time < event_start_time:
        video_index += 1
        cur_time = image_timestamps[video_index]

    next_event_time = event_timestamps[event_idx + 1]

    print(f"Writing episode {episode_idx}...")
    time_delta = next_event_time - cur_time
    start_time = cur_time
    approx_frames = int(np.ceil(time_delta * 30 / 1e9))
    pbar = tqdm.tqdm(total=approx_frames)
    refresh_n = 0
    while cur_time < next_event_time:
        cur = states[video_index].astype(np.float32)
        target = targets[video_index].astype(np.float32)
        # Assume gripper is last
        action = np.array([*(target[:-1] - cur[:-1]), target[7]], dtype=np.float32)
        dataset.add_frame({
            "image": read_image_torch(f"{out_dir}/images/cam1_{video_index}.png"),
            "wrist_image": read_image_torch(f"{out_dir}/images/cam2_{video_index}.png"),
            "depth": torch.tensor(cv2.imread(f"{out_dir}/images/cam1_depth_{video_index}.tiff", cv2.IMREAD_UNCHANGED), dtype=torch.uint16),
            "wrist_depth": torch.tensor(cv2.imread(f"{out_dir}/images/cam1_depth_{video_index}.tiff", cv2.IMREAD_UNCHANGED), dtype=torch.uint16),
            "state": cur,
            "actions": action,
            "task": episode_task
        })
        video_index += 1
        cur_time = image_timestamps[video_index]
        refresh_n += 1
        if refresh_n == 10:
            pbar.n = int(np.ceil((cur_time - start_time) * 30 / 1e9))
            pbar.refresh()
            refresh_n = 0
    dataset.save_episode()

    # If done with episodes, don't scroll
    episode_idx += 1
    if episode_idx >= n_episodes:
        break

    # Scroll to next episode start event...
    event_idx += 1
    while event_text[event_idx] in control_messages:
        # In case of spurious activations
        event_idx += 1
    # And also scroll the camera images forward
    event_start_time = event_timestamps[event_idx]
    while cur_time < event_start_time:
        video_index += 1
        cur_time = image_timestamps[video_index]

print("Done writing.")
