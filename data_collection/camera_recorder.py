import os
import sys
import time

import cv2

out_dir = sys.argv[1]
out_path = os.path.join(out_dir, 'images')
os.makedirs(out_path, exist_ok=True)

meta_file = open(f"{out_dir}/image_timestamps", 'a')
cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(10)

i = 0
j = 0
t = time.time_ns()
while True:
    res1, img1 = cap1.read()
    res2, img2 = cap2.read()
    cur_time = time.time_ns()
    cv2.imwrite(f"{out_path}/cam1_{j}.png", img1)
    cv2.imwrite(f"{out_path}/cam2_{j}.png", img2)
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
