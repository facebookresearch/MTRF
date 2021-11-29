# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

from gym.spaces import Box
from r3l.utils.quatmath import quat2euler, euler2quat, mat2quat
import numpy as np
from r3l.r3l_envs.base_env.mujoco_env import MujocoEnv
from typing import Tuple, Sequence, Optional

# Physical constraint from the Sawyer base
PHYSICAL_XY_LIMITATION = 0.945
PHYSICAL_Z_LIMITATION = 0.72

# add mocap to specs (can be called multiple times to add mutiple mocaps)
class MocapActor:
    def __init__(
            self,
            env: MujocoEnv,
            mocap_name: str,
            mocap_range: Tuple[np.ndarray, np.ndarray],
            mocap_velocity_lim: np.ndarray,
            control_mode: str = "deltapos",
    ):
        self.env = env

        # Update env configurations to add mocap [xyz, euler] action dimensions
        new_ac_low = np.concatenate([env.action_space.low, -np.ones((6,))])
        new_ac_high = np.concatenate([env.action_space.high, np.ones((6,))])
        env.action_space = Box(new_ac_low, new_ac_high)
        self.configure(
            mocap_name, mocap_range, mocap_velocity_lim, control_mode)

    def configure(
            self,
            mocap_name: str,
            mocap_range: Tuple[np.ndarray, np.ndarray],
            mocap_velocity_lim: np.ndarray,
            control_mode: str = "deltapos",
    ):
        self.mocap_name = mocap_name
        self.mocap_range = np.array(mocap_range.copy())
        self.mocap_pos_mean = np.mean(self.mocap_range[:3, :], axis=1)
        self.mocap_euler_mean = np.mean(self.mocap_range[3:, :], axis=1)
        self.mocap_quat_mean = euler2quat(self.mocap_euler_mean)
        self.mocap_velocity_lim = mocap_velocity_lim.copy()
        self.control_mode = control_mode

        self.mocap_mid = np.concatenate([self.mocap_pos_mean, self.mocap_euler_mean])
        # TODO: How to handle asymmetric ranges where one side should be allowed
        # to move farther? Ex: z direction, limit direction toward the ground
        # more than the up direction.
        self.mocap_rng = 0.5 * (self.mocap_range[:, 1] - self.mocap_range[:, 0])

    def update(self, act):
        # TODO(justinvyu): This code should be all moved outside into SawyerRobot
        # Get current position
        mocap_pos = self.env.data.get_mocap_pos(self.mocap_name).copy()
        mocap_quat = self.env.data.get_mocap_quat(self.mocap_name).copy()
        a = np.clip(act, -1., 1.)
        mocap_euler = quat2euler(mocap_quat)
        for i in range(3):
            if abs(mocap_euler[i]-self.mocap_mid[3+i])>np.pi/2:
                # if bottom flipped to +ve side
                if(mocap_euler[i]>self.mocap_mid[3+i] + self.mocap_rng[3+i]):
                    mocap_euler[i] -= 2*np.pi
                # if top flipped to -ve side
                elif(mocap_euler[i]<self.mocap_mid[3+i] - self.mocap_rng[3+i]):
                    mocap_euler[i] += 2*np.pi

        mocap_curr = np.concatenate([mocap_pos, mocap_euler])
        velocity_lim = self.mocap_velocity_lim.copy()
        if self.control_mode == "deltapos":
            # Switched into a delta controlled mocap system
            a = mocap_curr + velocity_lim * a
            a = np.clip(
                a,
                self.mocap_mid - self.mocap_rng,
                self.mocap_mid + self.mocap_rng
            )
        elif self.control_mode == "pos":
            a = self.mocap_mid + a * self.mocap_rng
        else:
            raise NotImplementedError

        # After clipping to the user provided range, also clip to physical
        # by not taking the action in the respective direction
        if np.linalg.norm(a[:2]) > PHYSICAL_XY_LIMITATION:
            a[:2] = mocap_curr[:2]
        if a[2] < PHYSICAL_Z_LIMITATION:
            a[2] = mocap_curr[2]

        pos = a[:3]
        euler = a[3:]
        quat = euler2quat(euler)

        self.env.data.set_mocap_pos(self.mocap_name, pos)
        self.env.data.set_mocap_quat(self.mocap_name, quat)
