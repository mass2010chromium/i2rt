import time

import einops
import mediapy

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_id = "local/outputs"

dataset = LeRobotDataset(repo_id, root="hf/outputs")
i = 0
N = 30
images = []
t = time.monotonic()
while True:
    if dataset[i]['task_index'] > 0:
        break
    img = dataset[i]['image']
    images.append(einops.rearrange(img, 'c h w -> h w c').cpu().numpy())
    i += 1
    if i % N == 0:
        t1 = time.monotonic()
        print(f"{N / (t1 - t)} fps")
        t = t1

mediapy.write_video("episode.mp4", images)
