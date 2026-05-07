import os
import threading
from typing import Optional

import mujoco
import numpy as np


class MuJoCoKDL:
    """A simple class for computing inverse dynamics using MuJoCo."""

    def __init__(self, path: str, site_name: str = "grasp_site"):
        self.model = mujoco.MjModel.from_xml_path(os.path.expanduser(path))
        self.data = mujoco.MjData(self.model)
        self.set_gravity(np.array([0, 0, -9.81]))

        # Disable all collisions
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        # Disable all joint limit
        self.model.jnt_limited[:] = 0
        self._site_name = site_name
        self.mj_lock = threading.Lock()

    @property
    def joint_limits(self) -> np.ndarray:
        return self.model.jnt_range

    def fk(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray:
        """Compute the forward kinematics for the given joint configuration.

        Args:
            q (np.ndarray): The joint configuration.
            site_name (Optional[str]): Name of the site for which to compute the forward kinematics.
                       If not provided, the default site name is used.

        Returns:
            (np.ndarray): Site frame in world frame. Shape: (4, 4)
        """
        length = len(q)
        with self.mj_lock:
            self.data.qpos[:length] = q
            self.data.qpos[length:] = 0
            self.data.qvel[:] = 0
            site_name = site_name or self._site_name
            mujoco.mj_forward(self.model, self.data)

            # Partly copied from gemini
            pose = np.eye(4)
            site = self.data.site(site_name)
            # site_xmat is stored as a flattened array of 9 elements
            pose[:3, :3] = site.xmat.reshape((3, 3))
            pose[:3, 3] = site.xpos
            return pose

    # Adapted from gemini
    def compute_jacobian(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray:
        """
        Computes the 6xnv Jacobian for a specific site.
        Returns:
            Jac: 6xnv translation Jacobian, first three rows are rotation, last 3 are position.
        """
        length = len(q)
        with self.mj_lock:
            site_name = site_name or self._site_name
            site = self.data.site(site_name)
            self.data.qpos[:length] = q
            self.data.qpos[length:] = 0

            jac = np.zeros((6, self.model.nv))

            # Argument order: pos, rot
            mujoco.mj_jacSite(self.model, self.data, jac[3:, :], jac[:3, :], site.id)

        return jac[:, :length]

    def compute_inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qdotdot: np.ndarray) -> np.ndarray:
        assert len(q) == len(qdot) == len(qdotdot)
        length = len(q)
        with self.mj_lock:
            self.data.qpos[:length] = q
            self.data.qpos[length:] = 0
            self.data.qvel[length:] = 0
            self.data.qvel[:length] = qdot
            self.data.qvel[length:] = 0
            self.data.qacc[:length] = qdotdot
            self.data.qacc[length:] = 0
            mujoco.mj_inverse(self.model, self.data)
            return np.copy(self.data.qfrc_inverse[:length])

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Sets the gravity vector for the robot.

        Args:
            gravity (np.ndarray): The gravity vector as a NumPy array.

        """
        assert gravity.shape == (3,)
        self.model.opt.gravity = gravity
