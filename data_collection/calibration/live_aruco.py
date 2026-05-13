import json

import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import tqdm

from calibration_utils import detect_aruco_corner_pixels

print("Resetting realsense devices to be safe...", end="")
ctx = rs.context()
devices = ctx.query_devices()
for dev in tqdm.tqdm(devices):
    dev.hardware_reset()
print("Done.")

pipeline = rs.pipeline()  # Overhead
config = rs.config()
config.enable_device('346522076629')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg = pipeline.start(config)

profile = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics() 
with open("intrinsics.json", "w") as outfile:
    json.dump({
        "cx": intr.ppx,
        "cy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy
    }, outfile)

#pipeline = rs.pipeline()  # Wrist
#config = rs.config()
#config.enable_device('348122070940')
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    corners, ids = detect_aruco_corner_pixels(img, aruco.DICT_5X5_100)
    print(corners, ids)
