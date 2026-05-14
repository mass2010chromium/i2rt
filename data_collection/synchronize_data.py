import glob
import json
import re

import cv2
import numpy as np
import scipy

class SyncData:
    def __init__(self, out_dir: str, action_mode: str = "ee"):
        self.out_dir = out_dir
        self.action_mode = action_mode

    CONTROL_MESSAGES = [
        "start_episode",
        "end_episode",
        "cancel"
    ]

    def get_synchronized_data(self):
        """
        Read and synchronize the joint and control data according to image frame timestamps.
        """
        out_dir = self.out_dir
        action_mode = self.action_mode

        image_timestamps = np.array([int(x.strip()) for x in open(f"{out_dir}/image_timestamps", 'r').readlines()])

        # Parsing robot log outputs. (more annoying)
        state_observation_timestamps = []
        state_observation = []
        state_target = []
        n_episodes = 0
        first_episode = None
        previous_episode = None
        event_timestamps = []
        event_text = []

        control_messages = SyncData.CONTROL_MESSAGES

        observation_keys = {
            "joint": ('joint.pos', 'joint.target'),
            "ee": ('ee.pos', 'ee.target')
        }

        observation_key, target_key = observation_keys[action_mode]

        # Grab the timestamp. Strip out the .log
        time_from_fname = lambda s: int(s.rsplit('_', 1)[1][:-4])
        robot_data_files = sorted(glob.glob(f"{out_dir}/robot_joint*"), key=time_from_fname)
        for fname in robot_data_files:
            with open(fname, 'r') as robot_joint_data:
                for line in robot_joint_data:
                    try:
                        data = json.loads(line)
                        state_observation_timestamps.append(data['time'])
                        state_observation.append(np.array(data[observation_key]))
                        state_target.append(np.array(data.get(target_key, np.zeros(8))))
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
                                if first_episode == previous_episode:
                                    first_episode = None
                                del event_timestamps[previous_episode]
                                del event_text[previous_episode]
                                previous_episode = None
                                n_episodes -= 1
                        elif data['msg'] not in control_messages:
                            previous_episode = len(event_timestamps)
                            if first_episode is None:
                                first_episode = previous_episode
                            n_episodes += 1
                        event_timestamps.append(data['time'])
                        event_text.append(data['msg'])
                    except:
                        print(f"{event_fname}: Could not parse line: '{line}'")
                        continue

        # Joint positions synchronized with image timestamps by linear interpolation.
        state_observation_interpolated = scipy.interpolate.make_interp_spline(np.array(state_observation_timestamps), np.array(state_observation), k=1)(image_timestamps)
        state_target_interpolated = scipy.interpolate.make_interp_spline(np.array(state_observation_timestamps), np.array(state_target), k=1)(image_timestamps)
        event_timestamps = np.array(event_timestamps)
        return {
            "image_timestamps": image_timestamps,
            "state": state_observation_interpolated,
            "target": state_target_interpolated,
            "event_times": event_timestamps,
            "events": event_text,
            "first_episode": first_episode,
            "num_episodes": n_episodes
        }

    def get_images(self, index: int, key: str = "cam1"):
        return {
            "rgb": cv2.imread(f"{self.out_dir}/images/{key}_{index}.png", cv2.IMREAD_UNCHANGED),
            "depth": cv2.imread(f"{self.out_dir}/images/{key}_depth_{index}.tiff", cv2.IMREAD_UNCHANGED)
        }
