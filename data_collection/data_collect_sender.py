import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import threading

import numpy as np
import portal
import tyro

DEFAULT_ROBOT_PORT = 11333

class Client:
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def log_message(self, message: str):
        self._client.log_message(message)


@dataclass
class Args:
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT


def main(args: Args) -> None:
    client_robot = Client(args.server_port, host=args.server_host)
    while True:
        s = input()
        client_robot.log_message(s)

if __name__ == "__main__":
    main(tyro.cli(Args))
