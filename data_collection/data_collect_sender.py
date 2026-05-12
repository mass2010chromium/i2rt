import copy
from dataclasses import dataclass
import random
import threading
import time
from typing import Dict, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import portal
import tyro

DEFAULT_ROBOT_PORT = 11333

class Client:
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def log_message(self, message: str):
        self._client.log_message(message)



class Generate:
    def __init__(self):
        self.movables = []
        self.containers = []
        self.placed_items = []
        self.place_items = []
        self.init_state = []
        self.task = []
        self.container_occupancy = 0.3
        self.reuse_dest = 0.8
        #self.basket_occupancy = 0.5

    def setup(self):
        f = input("Enter filename or empty to populate manually: ").strip()
        if f == "":
            print("Enter your containers:")
            self.containers = self.populate_list()
            print("Enter your other items")
            self.movables = self.populate_list() + self.containers
        else:
            try:
                file = open(f)
                self.containers = []
                self.movables = []
                dest = self.containers
                for line in file:
                    line = line.strip()
                    if line == '':
                        continue
                    if line.startswith('='):
                        if dest is not self.movables:
                            self.movables += self.containers
                            dest = self.movables
                        continue
                    dest.append(line)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return
        print(f"{self.movables=}")
        print(f"{self.containers=}")
        self.make_init_state()

    def make_hl_task(self):
        num_steps = random.randint(1, 3)
        container_states = {i: 3 for i in range(len(self.containers))}
        for i, container in enumerate(self.containers):
            if 'basket' in container:
                container_states[i] = 5
        for (i, j) in self.init_state:
            if i < len(self.containers):
                container_states[j] -= 3
            else:
                container_states[j] -= 1
        moved_items = set()
        placed_items = copy.copy(self.placed_items)
        self.task = []
        prev_dest = None
        for i in range(num_steps):
            ok = False
            for j in range(200):
                selection = random.randrange(0, len(self.movables))
                if selection in moved_items:
                    continue
                move_item = self.movables[selection]
                if move_item in self.containers:
                    size = 3
                else:
                    size = 1
                if prev_dest is not None and random.random() < self.reuse_dest:
                    if container_states[prev_dest] < size:
                        prev_dest = None
                    else:
                        if prev_dest == selection:
                            continue
                        ok = True
                        dest = prev_dest
                        break
                if move_item in placed_items:
                    dest = random.randrange(-1, len(self.containers))
                    if dest == -1:
                        ok = True
                        break
                    if dest == selection:
                        continue
                    if container_states[dest] < size:
                        continue
                    prev_dest = dest
                    placed_items.remove(move_item)
                else:
                    dest = random.randrange(0, len(self.containers))
                    if dest == selection:
                        continue
                    if container_states[dest] < size:
                        continue
                    prev_dest = dest
                ok = True
                break
            if not ok:
                print("Failed to select move {i}...")
                continue

            moved_items.add(selection)
            if dest == -1:
                dest_name = "table"
            else:
                container_states[dest] -= size
                dest_name = self.containers[dest]
            self.task.append(self.gen_command(move_item, dest_name))
        return ' and '.join(self.task)

    def gen_command(self, move, dest):
        middle = "on"
        if "basket" in dest and move not in self.containers:
            middle = "in"
        starts = [
            "place", "put", "move"
        ]
        ends = [
            " the",
            "to the"
        ]
        pattern = "{start} the {move} {middle}{end} {dest}"
        return pattern.format(start=random.choice(starts), move=move, middle=middle, end=random.choice(ends), dest=dest)

    def make_init_state(self):
        self.init_state = []
        placed = []
        for i in random.sample(list(range(len(self.containers))), k=len(self.containers)):
            if i in placed:
                continue
#             if "basket" in container:
#                 if random.random() < self.basket_occupancy:
#                     for item in random.sample(self.containers, k=len(self.containers)):
#                         if item in items:
#                             continue
#                         placed.add(item)
#                         self.init_state.append(f"place {place_item} on {container}")
#                         break
#             else:
            if i < (len(self.containers) - 1) and random.random() < self.container_occupancy:
                idx = random.randrange(i, len(self.movables))
                placed.append(idx)
                self.init_state.append((idx, i))

        items = []
        for i in range(len(self.movables)):
            if i not in placed:
                items.append(i)
        self.placed_items = placed
        self.place_items = items
        self.print_init_state()

    def print_init_state(self):
        print("initialization:")
        print('-'*40)
        print('\n'.join(f"place {self.movables[i]} on {self.movables[j]}" for i, j in self.init_state))
        print('-'*40)


    def populate_list(self):
        res = []
        while True:
            s = input("> ").strip()
            if s == "":
                break
            res.append(s)
        return res

    def make_shuffle(self, n=1):
        placements = []
        for item in self.place_items:
            while True:
                coord = np.random.normal(size=(2,))
                failed = False
                for prev_coord in placements:
                    if np.linalg.norm(coord - prev_coord) < 0.5:
                        failed = True
                        break
                if not failed:
                    break
            placements.append(coord)
        placements = np.array(placements)
        plt.figure(0)
        plt.clf()
        plt.scatter(placements[:, 0], placements[:, 1])
        ax = plt.gca()
        for coord, item in zip(placements, self.place_items):
            ax.annotate(self.movables[item], coord, textcoords='data')
        ax.grid()
        plt.savefig("shuffle.png")
        print("Refer to shuffle.png for initial arrangement")


@dataclass
class Args:
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT

def main(args: Args) -> None:
    client_robot = Client(args.server_port, host=args.server_host)
    save_task = '<enter task>'
    # Jank: just wait for the other messages to go through
    gen = Generate()
    time.sleep(1)
    while True:
        s = input(save_task + " | ").strip()
        if s == "setup":
            gen.setup()
            save_task = gen.make_hl_task()
        elif s == "task":
            save_task = gen.make_hl_task()
        elif s == "shuffle":
            gen.make_shuffle()
        elif s == "cancel":
            client_robot.log_message("cancel")
        else:
            if s:
                save_task = s
            client_robot.log_message(save_task)

if __name__ == "__main__":
    main(tyro.cli(Args))
