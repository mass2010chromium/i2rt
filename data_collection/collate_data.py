import glob
import json
import sys

import numpy as np
import scipy
import tqdm

out_dir = sys.argv[1]
image_timestamps = np.array([int(x.strip()) for x in open(f"{out_dir}/image_timestamps", 'r').readlines()])

# Parsing robot log outputs. (more annoying)
joint_pos_timestamps = []
joint_pos = []
n_episodes = 0
first_episode = None
previous_episode = None
event_timestamps = []
event_text = []

control_messages = [
    "start_episode",
    "end_episode",
    "cancel"
]

# Grab the timestamp. Strip out the .log
time_from_fname = lambda s: int(s.rsplit('_', 1)[1][:-4])
robot_data_files = sorted(glob.glob(f"{out_dir}/robot_joint*"), key=time_from_fname)
for fname in robot_data_files:
    with open(fname, 'r') as robot_joint_data:
        for line in robot_joint_data:
            try:
                data = json.loads(line)
                joint_pos_timestamps.append(data['time'])
                joint_pos.append(data['joint.pos'])
            except:
                print(f"{fname}: Could not parse line: '{line}'")
                continue
    # Replace last occurrence.
    event_fname = fname[::-1].replace("robot_joint"[::-1], "teleop_event"[::-1])[::-1]
    with open(event_fname, 'r') as event_data:
        for line in event_data:
            try:
                data = json.loads(line)
                if data['msg'] == 'cancel':
                    if previous_episode is not None:
                        del event_timestamps[previous_episode]
                        del event_text[previous_episode]
                        previous_episode = None
                elif data['msg'] not in control_messages:
                    if first_episode is None:
                        first_episode = len(event_timestamps)
                    previous_episode = len(event_timestamps)
                    n_episodes += 1
                event_timestamps.append(data['time'])
                event_text.append(data['msg'])
            except:
                print(f"{event_fname}: Could not parse line: '{line}'")
                continue

# Joint positions synchronized with image timestamps by linear interpolation.
joint_pos_interpolated = scipy.interpolate.make_interp_spline(np.array(joint_pos_timestamps), np.array(joint_pos), k=1)(image_timestamps)
event_timestamps = np.array(event_timestamps)

print("Parsed metadata files")
print(f"{len(image_timestamps)} images, {n_episodes} episodes")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. Define the dataset metadata and features
# 'observation.image' is for the camera feed
# 'observation.state' is for the vector (e.g., joint angles)
dataset = LeRobotDataset.create(
    repo_id=f"local/{out_dir}",
    root=f"hf/{out_dir}",
    fps=30,
    robot_type="yam",
    features={
        "image": {
            "dtype": "video",
            "shape": (3, 224, 224),  # (Channels, Height, Width)
            "names": ["color"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (3, 224, 224),  # (Channels, Height, Width)
            "names": ["color"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["target_1", "target_2", "target_3", "target_4", "target_5", "target_6", "gripper"],
        },
    },
)

first_episode_start = event_timestamps[first_episode]
video_index = np.searchsorted(image_timestamps, first_episode_start)
event_idx = first_episode

import torchvision
resize_transform = torchvision.transforms.Resize((224, 224),
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
    cur_time = image_timestamps[video_index]

    print(f"Writing episode {episode_idx}...")
    time_delta = next_event_time - cur_time
    start_time = cur_time
    approx_frames = int(np.ceil(time_delta * 30 / 1e9))
    pbar = tqdm.tqdm(total=approx_frames)
    refresh_n = 0
    while cur_time < next_event_time:
        dataset.add_frame({
            "image": read_image_torch(f"{out_dir}/images/cam1_{video_index}.png"),
            "wrist_image": read_image_torch(f"{out_dir}/images/cam2_{video_index}.png"),
            "state": joint_pos_interpolated[video_index].astype(np.float32),
            "action": np.zeros(7).astype(np.float32),
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
