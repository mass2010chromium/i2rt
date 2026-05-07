import time

import numpy as np
import cv2
import einops
import mediapy

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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

repo_id = "local/simple_tasks"

dataset = LeRobotDataset(repo_id, root="hf/simple_tasks")
i = 0
N = 30
images = []
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
    gripper_min = min(gripper_min, dataset[i]['actions'][6])
    gripper_max = max(gripper_max, dataset[i]['actions'][6])
    episode_idx = int(dataset[i]['episode_index'])
    if episode_idx not in episodes_counted:
        task_idx = int(dataset[i]['task_index'])
        if task_idx not in task_dist:
            task_dist[task_idx] = 0
        task_dist[task_idx] += 1
        episodes_counted.add(episode_idx)

    images.append(write_text_to_frame(einops.rearrange(img, 'c h w -> h w c').cpu().numpy(), f"episode {episode_idx}"))
    i += 1
    if i % N == 0:
        t1 = time.monotonic()
        print(f"{N / (t1 - t)} fps")
        print(f"gripper limits: [{gripper_min}, {gripper_max}]")
        t = t1

print(task_dist)
mediapy.write_video("episode.mp4", images)
