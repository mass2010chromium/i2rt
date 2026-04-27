import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import cv2
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(10)

# 1. Define the dataset metadata and features
# 'observation.image' is for the camera feed
# 'observation.state' is for the vector (e.g., joint angles)
dataset = LeRobotDataset.create(
    repo_id=None,
    fps=10,
    robot_type="yam",
    features={
        "observation.image": {
            "dtype": "video",
            "shape": (3, 224, 224),  # (Channels, Height, Width)
            "names": ["color"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["target_1", "target_2", "target_3", "target_4", "target_5", "target_6"],
        },
    },
)

# 2. Simulate gathering data for one episode
num_frames = 100
for i in range(num_frames):
    # Dummy data: random image and vectors
    image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    state = torch.randn(6)
    action = torch.randn(6)

    # Add a single frame to the dataset
    dataset.add_frame({
        "observation.image": image,
        "observation.state": state,
        "action": action,
    })

# 3. Mark the end of the episode
dataset.save_episode(task="Reach the red cube")

# 4. Finalize and consolidate the dataset (creates metadata files)
dataset.consolidate()
