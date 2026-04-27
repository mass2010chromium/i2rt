import json
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import threading

import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperType

DEFAULT_ROBOT_PORT = 11333


class ServerRobot:
    """A simple server for a leader robot."""

    def __init__(self, robot: Robot, port: str, event_logfile: str):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Robot Sever Binding to {port}, Robot: {robot}")

        self.event_logfile = open(event_logfile, 'w')

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)
        self._server.bind("log_message", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the leader robot."""
        self._server.start()

    def log_message(self, message: str) -> None:
        print(message)
        print(json.dumps({"time": time.time_ns(), "msg": message}), file=self.event_logfile)

    def close(self):
        self._robot.close()
        self.event_logfile.close()


@dataclass
class Args:
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = (
        "yam_teaching_handle"
    )
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    ee_mass: Optional[float] = None
    log_dir: str = "outputs"
    """Override end-effector (link_6) mass in kg for gravity compensation. Defaults to the value in the XML."""


def main(args: Args) -> None:
    gripper_type = GripperType.from_string_name(args.gripper)

    t = time.time_ns()
    robot_log_filename = f"{args.log_dir}/robot_joint_{t}.log"
    event_log_filename = f"{args.log_dir}/teleop_event_{t}.log"

    # TODO: do calibration in the proper order
    robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type, ee_mass=args.ee_mass, start_thread=False
        limit_gripper_force=15.0, logfile=log_filename)
    robot.start_server(separate_thread=False)
    server_robot = ServerRobot(robot, args.server_port, event_logfile=event_log_filename)
    try:
        server_robot.serve()
    except KeyboardInterrupt:
        server_robot.close()

if __name__ == "__main__":
    main(tyro.cli(Args))
