# Data collection scripts readme

## Robot basics

`ifconfig` can check if robot CAN comms are set up correctly (you should see `can0` and `can1`)

## Starting data collection
For the following instructions, `<log_dir>` should be the same for all command invocations. Pick one and it will be where data is saved
1. Starting the follower robot
  - STAND CLEAR OF BOTH ROBOTS WHEN EXECUTING THIS SCRIPT! ROBOTS MAY MOVE ON STARTUP!
  - `python robot_recorder.py --log-dir <log_dir>`
  - Optional args: `--can-channel <can_channel>`
  - Starts the follower robot server, logs joint angles to a file in the log directory.
  - Also responsible for logging events (start of episode, end of episode, etc)
  - This will enable the follower robot's motors.
2. Start the cameras
  - `python camera_recorder.py <log_dir>`
  - Reads from the two cameras, and dumps the raw images into a subfolder of the logs folder.
  - Also writes a metadata file with image timestamps
3. Start teleop
  - STAND CLEAR OF BOTH ROBOTS WHEN EXECUTING THIS SCRIPT! ROBOTS MAY MOVE ON STARTUP!
  - Before running the script, try to make sure the robots' configurations are similar.
  - `python data_collect_leader.py`
  - Optional args: `--can-channel <can_channel>`
  - Starts the leader robot, which connects to the follower robot server.
  - This will enable the leader robot's motors.
  - To enter teleop:
    - Press the yellow button on the leader robot's handle.
    - Wait for three(ish) seconds for the robots to synchronize. Make sure the leader robot does not move during this time. The follower robot should move to match the leader's position.
    - After this time has elapsed, the follower robot will "follow" the leader's motion. You can check this by moving the leader robot around.
  - It may help to have two hands when teleoperating: One on the handle and one on the robot's elbow
    - Maybe we can fix this with good OSC?
  - Press the white button to freeze/unfreeze the leader arm.
    - Will toggle between "locked in place" and "movable"
    - Leave it locked in place, and move on to the next step
4. Start task generator
  - `python data_collect_sender.py`
  - This opens a terminal window that accepts keyboard input.

## Running data collection

To collect one episode, while all the scripts are running:
1. Move the arm to the desired start position via teleop.
2. Lock the arm by pressing the white button.
3. Enter a task description by typing into the `data_collect_sender.py` terminal window.
4. Unlock the arm by pressing the white button, and execute the task.
  - This signals the start of the episode.
  - Try your best to do it in one smooth motion.
5. Lock the arm by pressing the white button to signal the end of the episode.

If the robots ever crash, it is OK to restart the robot scripts into the same output folder.
It should also be fine to restart the camera scripts... some work is required (TODO) to make sure the cameras bind consistently always.

## Collating data at the end

Run `python collate_data.py <log_dir>`
This will dump everything into a local lerobot dataset in `hf/<log_dir>`.
Images that are not a part of an episode will be stripped out.
