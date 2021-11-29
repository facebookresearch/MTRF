# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
import os
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple
from pathlib import Path
import pickle

from r3l import PROJECT_PATH, RESET_STATES_PATH
from r3l.r3l_envs.inhand_env.base import ObjectType
from r3l.r3l_envs.inhand_env.reposition import SawyerDhandInHandObjectRepositionFixed
from r3l.robot.default_configs import FLIPUP_SAWYER_ROBOT_CONFIG, ARM_QPOS_PALMDOWN, MOCAP_EULER_PALMUP, MOCAP_EULER_PALMDOWN
from r3l.utils.circle_math import circle_distance
from r3l.utils.quatmath import quat2euler, euler2quat, mat2quat


class SawyerDhandInHandObjectFlipUpFixed(SawyerDhandInHandObjectRepositionFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "sawyer_to_target_x_circle_distance_reward": 5.0,
        "sawyer_to_target_z_circle_distance_reward": 5.0,
        # NOTE: This xyz reward is needed to make sure the flip up policy
        # settles at a reasonable height
        "object_to_target_xyz_distance_reward": 2.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "drop_penalty": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            **kwargs
    ):
        if kwargs.get("object_type", None) == ObjectType.Valve3:
            reset_policy_dirs = [
                # (Path(PROJECT_PATH)
                # / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3PickupFixed-v0/pickup_raised_valve")
            ]
            reset_state_pkl_path = None # TODO
        elif kwargs.get("object_type", None) in (
                ObjectType.Dodecahedron, ObjectType.DodecahedronBasket, ObjectType.DodecahedronBulb):
            reset_policy_dirs = [(
                Path(PROJECT_PATH)
                / "r3l/r3l_agents/softlearning/SawyerDhandInHandDodecahedronPickupFixed-v0/pickup_trained_with_resets"
            )]
            reset_state_pkl_path = str(Path(RESET_STATES_PATH) / "dodecahedron/picked_up.pkl")
        else:
            print("Object type doesn't have a reset policy")
            reset_policy_dirs = []
            reset_state_pkl_path = None

        env_params = dict(
            task_name="Flip Up",
            sawyer_config=FLIPUP_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
            # Set a default init and target
            init_xyz_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0.72, 0.15, 0.75])]
            },
            init_euler_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0, 0, np.pi])],
            },
            target_xyz_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0.72, 0.15, 1.0])]
            },
            target_euler_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0, 0, 0])],
            },
            reset_policy_directories=reset_policy_dirs,
            reset_state_pkl_path=reset_state_pkl_path,
            # NOTE: `reset_robot=False` prevents the environment from doing a
            # hard reset after the object has been picked up by the loaded policy
            reset_robot=False,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

        self.init_qpos[:7] = ARM_QPOS_PALMDOWN
        self.act_mid = 0.5 * self.model.actuator_ctrlrange[:, 0] + 0.5 * self.model.actuator_ctrlrange[:, 1]
        self.init_qpos[7:23] = self.act_mid + 0.35 * self.act_rng

        # pkl_path = (
        #     Path(PROJECT_PATH)
        #     / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3PickupFixed-v0"
        #     / "pickup_raised_valve/pickup_reset.pkl"
        # )
        # with open(pkl_path, "rb") as f:
        #     reset_data = pickle.load(f)
        # self.reset_qpos = reset_data["qpos"]
        # self.reset_qvel = reset_data["qvel"]

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        reward_dict = super().get_reward_dict(action, obs_dict)

        # Distance to turn the palm up
        mocap_desired_euler = MOCAP_EULER_PALMUP
        curr_mocap_euler = obs_dict['mocap_euler'].squeeze()
        circle_dist = circle_distance(curr_mocap_euler, mocap_desired_euler)
        # Penalize both errors in the x and z angles when flipping up
        # This prevents just the wrist from flipping but having the Z angle
        # of the Sawyer get very far from the initial angle
        circle_dist_flipup_x = np.abs(circle_dist[0])
        circle_dist_flipup_z = np.abs(circle_dist[2])
        reward_dict["sawyer_to_target_x_circle_distance_reward"] = -circle_dist_flipup_x
        reward_dict["sawyer_to_target_z_circle_distance_reward"] = -circle_dist_flipup_z
        object_xyz = obs_dict['object_xyz'].squeeze()

        # bonus for being close to desired orientation with wrist and object being not dropped
        reward_dict["small_bonus"] = (
            10. * (circle_dist_flipup_x + circle_dist_flipup_z < 0.2)
            + 1. * (object_xyz[2] > 0.85)
        )

        reward_dict["big_bonus"] = (
            50. * (circle_dist_flipup_x + circle_dist_flipup_z < 0.2
                   and obs_dict["object_to_target_xyz_distance"] < 0.075)
        )

        # penalty for dropping the object
        # z_dist = obs_dict["object_to_target_z_distance"]
        reward_dict["drop_penalty"] = -5 * (object_xyz[2] < 0.85)

        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = (
            reward_dict["sawyer_to_target_x_circle_distance_reward"]
            + reward_dict["sawyer_to_target_z_circle_distance_reward"]
            + reward_dict["small_bonus"]
            + reward_dict["drop_penalty"]
        )
        solved = reward_dict["small_bonus"] > 0 and reward_dict["drop_penalty"] >= 0
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))
