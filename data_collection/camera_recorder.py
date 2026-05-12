import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import tqdm

out_dir = sys.argv[1]
out_path = os.path.join(out_dir, 'images')
os.makedirs(out_path, exist_ok=True)

try:
    with open(f"{out_dir}/image_timestamps", 'r') as image_stamps:
        n_lines = len(image_stamps.readlines())
except:
    print("Could not open timestamps file, creating new")
    n_lines = 0
meta_file = open(f"{out_dir}/image_timestamps", 'a')

print("Resetting realsense devices to be safe...", end="")
ctx = rs.context()
devices = ctx.query_devices()
for dev in tqdm.tqdm(devices):
    dev.hardware_reset()
print("Done.")

pipeline_1 = rs.pipeline()  # Overhead
config_1 = rs.config()
config_1.enable_device('346522076629')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_2 = rs.pipeline()  # Wrist
config_2 = rs.config()
config_2.enable_device('348122070940')
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_1.start(config_1)
pipeline_2.start(config_2)

i = 0
j = n_lines
t = time.time_ns()
while True:
    frames_1 = pipeline_1.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    color_frame_1 = frames_1.get_color_frame()
    if not depth_frame_1 or not color_frame_1:
        continue
    frames_2 = pipeline_2.wait_for_frames()
    depth_frame_2 = frames_2.get_depth_frame()
    color_frame_2 = frames_2.get_color_frame()
    if not depth_frame_2 or not color_frame_2:
        continue

    # Close enough....
    cur_time = time.time_ns()
    cv2.imwrite(f"{out_path}/cam1_{j}.png", np.asanyarray(color_frame_1.get_data()))
    cv2.imwrite(f"{out_path}/cam2_{j}.png", np.asanyarray(color_frame_2.get_data()))
    depth1 = np.asanyarray(depth_frame_1.get_data())
    cv2.imwrite(f"{out_path}/cam1_depth_{j}.tiff", depth1)
    cv2.imwrite(f"{out_path}/cam2_depth_{j}.tiff", np.asanyarray(depth_frame_2.get_data()))
    #readback = cv2.imread(f"{out_path}/cam1_depth_{j}.tiff", cv2.IMREAD_UNCHANGED)
    print(cur_time, file=meta_file, flush=True)
    i += 1
    j += 1
    if i == 120:
        t2 = time.time_ns()
        print(f"{i * 1e9/(t2 - t)} FPS")
        t = t2
        with open(f"{out_dir}/latest_image", 'w') as latest:
            print(j, file=latest, flush=True)
        i = 0
