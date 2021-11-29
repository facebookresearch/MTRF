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
from r3l.robot.default_configs import POSE_SAWYER_ROBOT_CONFIG


class SawyerDhandInHandObjectPoseFixed(SawyerDhandInHandObjectRepositionFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "pose_distance_reward": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            **kwargs
    ):
        env_params = dict(
            sawyer_config=POSE_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        reward_dict = super().get_reward_dict(action, obs_dict)
        reward_dict.update({
            "pose_distance_reward": -obs_dict["pose_dist"],
        })
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = reward_dict["pose_distance_reward"]
        solved = bool(obs_dict["pose_dist"] < 0.05)
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))
