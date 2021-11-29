# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple

from r3l.r3l_envs.inhand_env.base import ObjectType
from r3l.r3l_envs.inhand_env.reposition import SawyerDhandInHandObjectRepositionFixed
from r3l.robot.object import ObjectState
from r3l.utils.quatmath import quat2euler, euler2quat
from r3l.utils.range import get_range_from_params
from r3l.utils.circle_math import circle_distance, circle_distance_mod
from r3l.robot.default_configs import REORIENT_SAWYER_ROBOT_CONFIG


class SawyerDhandInHandObjectReorientFixed(SawyerDhandInHandObjectRepositionFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_target_circle_distance_reward": 1.0,
        "object_to_target_xy_distance_reward": 1.0,
        "object_to_hand_xyz_distance_reward": 1.0,
        # "span_dist": 1.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
    }

    def __init__(
            self,
            object_type: ObjectType = ObjectType.Rod,
            random_init_angle: bool = False,
            random_target_angle: bool = False,
            symmetric_task: bool = True,
            reorient_only: bool = False,
            **kwargs
    ):
        if random_init_angle:
            init_euler_params = {
                "type": "UniformRange",
                "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
            }
        else:
            init_euler_params = {
                "type": "DiscreteRange",
                "values": [np.array([0, 0, 0])],
            }
        target_xyz_params = {
            "type": "DiscreteRange",
            "values": [np.array([0.72, 0.15, 0.76])],
        }
        if random_target_angle:
            target_euler_params = {
                "type": "UniformRange",
                "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
            }
        else:
            target_euler_params = {
                "type": "DiscreteRange",
                "values": [np.array([0, 0, np.pi])],
            }

        if object_type == ObjectType.Dodecahedron:
            target_euler_params = {
                "type": "DiscreteRange",
                "values": [np.array([2.034, 0, 0])],
            }
        self._symmetric_task = symmetric_task

        # Replace the reward keys with symmetric ones if we want rotational symmetry
        reward_keys_and_weights = self.DEFAULT_REWARD_KEYS_AND_WEIGHTS.copy()
        if symmetric_task:
            reward_keys_and_weights.pop("object_to_target_circle_distance_reward", None)
            if object_type == ObjectType.Valve3:
                reward_keys_and_weights.update({
                    "object_to_target_mod_120_circle_distance_reward": 2.0,
                })
            elif object_type in (ObjectType.Rod, ObjectType.Pipe):
                reward_keys_and_weights.update({
                    "object_to_target_mod_180_circle_distance_reward": 2.0,
                })

        self._reorient_only = reorient_only
        if reorient_only:
            # Remove XY distance reward if we only care about reorienting
            reward_keys_and_weights.pop("object_to_target_xy_distance_reward", None)

        env_params = dict(
            task_name="Reorient",
            object_type=object_type,
            init_euler_range_params=init_euler_params,
            target_xyz_range_params=target_xyz_params,
            target_euler_range_params=target_euler_params,
            reward_keys_and_weights=reward_keys_and_weights,
            sawyer_config=REORIENT_SAWYER_ROBOT_CONFIG,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()
        if self._symmetric_task and self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            object_euler_z = np.array([np.mod(obs_dict["object_z_orientation"], np.pi)])
            target_euler_z = np.array([np.mod(obs_dict["target_z_orientation"], np.pi)])
            obs_dict.update(
                object_z_orientation=object_euler_z,
                target_z_orientation=target_euler_z,
                object_z_orientation_cos=np.cos(object_euler_z),
                object_z_orientation_sin=np.sin(object_euler_z),
                target_z_orientation_cos=np.cos(target_euler_z),
                target_z_orientation_sin=np.sin(target_euler_z),
            )
        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        reward_dict = super().get_reward_dict(action, obs_dict)
        xy_dist = obs_dict["object_to_target_xy_distance"].squeeze()

        if self._symmetric_task:
            if self.object_type == ObjectType.Valve3:
                circle_dist = obs_dict["object_to_target_mod_120_circle_distance"].squeeze()
            elif self.object_type in (ObjectType.Rod, ObjectType.Pipe):
                circle_dist = obs_dict["object_to_target_mod_180_circle_distance"].squeeze()
            else:
                circle_dist = obs_dict["object_to_target_circle_distance"].squeeze()
        else:
            circle_dist = obs_dict["object_to_target_circle_distance"].squeeze()

        small_bonus = np.zeros(1)
        big_bonus = np.zeros(1)

        if self._reorient_only:
            small_bonus = 10. * (circle_dist < 0.1)
            big_bonus = 50. * (circle_dist < 0.05)
        else:
            small_bonus = 10. * (xy_dist < 0.1 and circle_dist < 0.1)
            big_bonus = 50. * (xy_dist < 0.075 and circle_dist < 0.05)

        reward_dict["small_bonus"] = small_bonus
        reward_dict["big_bonus"] = big_bonus
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = reward_dict["object_to_target_circle_distance_reward"]
        solved = bool(obs_dict["object_to_target_circle_distance"] < 0.1)
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))