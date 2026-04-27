"""
Utility file to map between usb device IDs and files on computer.
Thin wrapper around `usb_info` script that I (Jing-Chen) found somewhere online.
"""
import os
import subprocess
from threading import Lock

"""
Mapping from device_id to path on the computer.

Usage:

```
    device_id = 'FTDI_USB__-__Serial_Converter_FT4XXXXX'
    usb_fname = usb_util.DEVPATH_MAP[device_id]
```
"""
DEVPATH_MAP=dict()
_devpath_lock = Lock()

def refresh_usb_map():
    global DEVPATH_MAP
    with _devpath_lock:
        DEVPATH_MAP = dict()
        res = subprocess.run(["bash", os.path.dirname(os.path.realpath(__file__))+"/usb_info.bash"], capture_output=True, text=True)
        for line in res.stdout.split("\n"):
            if line.strip():
                devpath, device_id = line.strip().split(" - ", 1)
                DEVPATH_MAP[device_id] = devpath

refresh_usb_map()
print(f"initialize usb map, keys={list(DEVPATH_MAP.keys())}")
if __name__ == "__main__":
    hand_id = "FTDI_USB__-__Serial_Converter_FT4NQ5KC"
    head_id = "FTDI_USB__-__Serial_Converter_FT4TFNFF"
    head2_id = "FTDI_USB_TO_RS-485_DA6ACSMC"
    print(f"Hand is on {DEVPATH_MAP.get(hand_id, '---')}")
    print(f"Head is on {DEVPATH_MAP.get(head_id, '---')}")
    print(f"Head RS485 is on {DEVPATH_MAP.get(head2_id, '---')}")

