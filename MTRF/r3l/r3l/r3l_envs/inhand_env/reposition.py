# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
from gym import utils
import numpy as np
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple
import time

from r3l.r3l_envs.inhand_env.base import SawyerDhandInHandObjectBaseEnv
from r3l.robot.object import ObjectState
from r3l.utils.quatmath import quat2euler, euler2quat
from r3l.utils.range import get_range_from_params
from r3l.robot.default_configs import (
    DEFAULT_DHAND_ROBOT_CONFIG,
    REPOSITION_SAWYER_ROBOT_CONFIG,
    DEFAULT_OBJECT_CONFIG,
)


class SawyerDhandInHandObjectRepositionFixed(SawyerDhandInHandObjectBaseEnv):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_hand_xyz_distance_reward": 1.0,
        "object_to_target_xy_distance_reward": 1.0,
        "span_dist": 1.0,
    }
    def __init__(
            self,
            sim=None,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            init_xyz_range_params={},
            init_euler_range_params={},
            target_xyz_range_params={},
            target_euler_range_params={},
            pose_range_params={},
            **kwargs
    ):
        # Default params (which will be updated if you pass in new ones)
        env_params = dict(
            sim=sim,
            task_name="Reposition",
            readjust_to_object_in_reset=True,
            readjust_hand_euler=False,
            dhand_config=DEFAULT_DHAND_ROBOT_CONFIG,
            sawyer_config=REPOSITION_SAWYER_ROBOT_CONFIG,
            object_config=DEFAULT_OBJECT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
        )
        env_params.update(kwargs)
        all_params = env_params.copy()
        all_params.update(dict(
            init_xyz_range_params=init_xyz_range_params,
            init_euler_range_params=init_euler_range_params,
            target_xyz_range_params=target_xyz_range_params,
            target_euler_range_params=target_euler_range_params,
            pose_range_params=pose_range_params,
            # mujoco_render_frames=self.mujoco_render_frames
        ))
        utils.EzPickle.__init__(self, **all_params)
        super().__init__(**env_params)

        # TODO: Move these into base.py
        init_xyz_range_params = init_xyz_range_params or {
            "type": "DiscreteRange",
            "values": [self.init_qpos[-7:-4].copy()],
        }
        self._init_xyz_range = get_range_from_params(init_xyz_range_params, self.np_random)
        init_euler_range_params = init_euler_range_params or {
            "type": "DiscreteRange",
            "values": [quat2euler(self.init_qpos[-4:].copy())],
        }
        self._init_euler_range = get_range_from_params(init_euler_range_params, self.np_random)
        target_xyz_range_params = target_xyz_range_params or {
            "type": "DiscreteRange",
            # "values": [self.init_qpos[-7:-4].copy()],
            "values": [
                np.array([0.72, 0.15, 0.76]),
            ],
        }
        self._target_xyz_range = get_range_from_params(target_xyz_range_params, self.np_random)
        target_euler_range_params = target_euler_range_params or {
            "type": "DiscreteRange",
            "values": [np.array([0, 0, 0])],
        }
        self._target_euler_range = get_range_from_params(target_euler_range_params, self.np_random)
        pose_range_params = pose_range_params or {
            "type": "DiscreteRange",
            "values": [np.zeros(16)],
        }
        self._pose_range_params = get_range_from_params(pose_range_params, self.np_random)

        # Need to initialize the _last_target_xyz and _last_target_euler
        self.reset_target()

    def reset_target(self):
        target_xyz = next(self._target_xyz_range)
        target_euler = next(self._target_euler_range)
        self._pose_target = next(self._pose_range_params)

        self.sim.model.body_pos[self.target_bid] = target_xyz
        self.sim.model.body_quat[self.target_bid] = euler2quat(target_euler)
        self._last_target_xyz = target_xyz
        self._last_target_euler = target_euler

    def _reset(self):
        if self._should_reset():
            if self._verbose:
                print("Forcing full reset of object")
            # Reset object based on init parameters
            obj_xyz = next(self._init_xyz_range)
            obj_euler = next(self._init_euler_range)
            self.object.set_state(ObjectState(
                qpos=np.concatenate([obj_xyz, euler2quat(obj_euler)]),
                # qvel=np.zeros(6),
            ))

        # Perform loaded policy reset if applicable
        self.run_reset_policy()

        # Reset target
        self.reset_target()
        self.sim.forward()

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = reward_dict["object_to_target_xy_distance_reward"]
        solved = bool(obs_dict["object_to_target_xy_distance"] < 0.15)
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))


class SawyerDhandInHandObjectRepositionRandomInit(SawyerDhandInHandObjectRepositionFixed):
    def __init__(
            self,
            init_xyz_range_params={},
            init_euler_range_params={},
            random_init_angle: bool = True,
            x_range: float = 0.2,
            y_range: float = 0.2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._x_range = x_range
        self._y_range = y_range
        box_center = self.init_qpos[-7:-4]
        random_init_xyz_range = (
            box_center + np.array([-x_range, -y_range, 0]),
            box_center + np.array([x_range, y_range, 0])
        )
        xyz_params = {
            "type": "UniformRange",
            "values": random_init_xyz_range,
        }
        xyz_params.update(init_xyz_range_params)
        if random_init_angle:
            random_init_euler_range = (
                np.array([0, 0, -np.pi]),
                np.array([0, 0, np.pi])
            )
            euler_params = {
                "type": "UniformRange",
                "values": random_init_euler_range,
            }
        else:
            fixed_init_euler_range = [np.array([0, 0, 0])]
            euler_params = {
                "type": "DiscreteRange",
                "values": fixed_init_euler_range,
            }
        euler_params.update(init_euler_range_params)
        self._init_xyz_range = get_range_from_params(xyz_params, self.np_random)
        self._init_euler_range = get_range_from_params(euler_params, self.np_random)
