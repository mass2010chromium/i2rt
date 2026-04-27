import os
import time
import sys

out_dir = sys.argv[1]
while True:
    try:
        latest_idx = open(f"{out_dir}/latest_image", 'r').read().strip()
        try:
            os.remove("cam1_latest.png")
        except:
            pass
        try:
            os.remove("cam2_latest.png")
        except:
            pass
        os.symlink(f"{out_dir}/images/cam1_{latest_idx}.png", "cam1_latest.png")
        os.symlink(f"{out_dir}/images/cam2_{latest_idx}.png", "cam2_latest.png")
        print(f"Showing frame {latest_idx}")
    except Exception as e:
        print(e)
        print("Could not open output file")
    time.sleep(4)
