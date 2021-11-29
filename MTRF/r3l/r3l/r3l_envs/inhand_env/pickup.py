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
from r3l.robot.default_configs import PICKUP_SAWYER_ROBOT_CONFIG


class SawyerDhandInHandObjectPickupFixed(SawyerDhandInHandObjectRepositionFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_hand_xyz_distance_reward": 1.0,
        "object_to_target_xy_distance_reward": 2.0,
        "object_to_target_z_distance_reward": 2.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "table_clear": 2.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            init_xyz_range_params={},
            init_euler_range_params={},
            target_xyz_range_params={},
            random_init_angle: bool = False,
            pickup_height_m: float = 0.2,
            **kwargs
    ):
        self.script_motion = False # TODO(Tony) Hardcoded as an option for now
        self.step_counter = 0
        # When script_motion is True:
        # - the arm motion will be scripted as defined in _step
        # - the counter will be incremented and added to observation dict
        # When script_motion is False:
        # - the counter will always be zero and added to observation dict

        self._pickup_height_m = pickup_height_m
        env_params = dict(
            sawyer_config=PICKUP_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

        # Initialize ranges for init and target positions
        box_center = np.array([0.72, 0.15, 0.72])
        xyz_params = {
            "type": "DiscreteRange",
            "values": [box_center],
        }
        xyz_params.update(init_xyz_range_params)
        euler_params = {
            "type": "DiscreteRange",
            "values": [np.array([0, 0, 0])],
        }
        if random_init_angle:
            euler_params = {
                "type": "UniformRange",
                "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
            }
        euler_params.update(init_euler_range_params)
        target_xyz_params = {
            "type": "DiscreteRange",
            "values": [box_center + np.array([0, 0, pickup_height_m])]
        }
        target_xyz_params.update(target_xyz_range_params)
        self._init_xyz_range = get_range_from_params(xyz_params, self.np_random)
        self._init_euler_range = get_range_from_params(euler_params, self.np_random)
        self._target_xyz_range = get_range_from_params(target_xyz_params, self.np_random)

        # Need to initialize the _last_target_xyz and _last_target_euler
        self.reset_target()


    def _step(self, action):
        """Task-specific step for the environment."""
        dhand_act, sawyer_act = action[:16], action[16:]
        if not self.initializing:
            if self.script_motion:
                # override sawyer_act with simple PD controller of hand xyz
                curr_xyz = self.sim.data.get_mocap_pos('mocap').copy()
                if self.step_counter < 20:
                    target_xyz = np.array([0.57, 0.15, 0.92])
                else:
                    target_xyz = np.array([0.57, 0.15, 1.52])
                diff = target_xyz - curr_xyz
                alpha = 1.5
                action_xyz = diff * alpha
                sawyer_act[:3] = action_xyz # override only xyz, not euler

            # NOTE: Order matters here, because dhand_robot wait controls freq
            self.sawyer_robot.step(sawyer_act)
            self.dhand_robot.step(dhand_act)

        if self.object.is_hardware:
            raise NotImplementedError
            # NOTE: Match hardware to sim by adding an offset to the object z
            object_qpos = self.last_obs_dict["object_qpos"] + np.array([0, 0, 0.8, 0, 0, 0, 0])
            self.sim.data.qpos[self.object.config.qpos_indices] = object_qpos
            self.sim.forward()


    def step(self, action):
        obs, rew, done, info = super().step(action)
        if self.script_motion:
            obs['step_counter'] = np.array([self.step_counter]) # override step_counter to be the correct value
        self.step_counter += 1
        return obs, rew, done, info

    def reset(self):
        self.step_counter = 0
        return super().reset()


    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_obs_dict()
        object_xyz, target_xyz = obs_dict["object_xyz"].squeeze(), obs_dict["target_xyz"].squeeze()
        object_to_target_z_distance = np.abs(target_xyz[2] - object_xyz[2])
        obs_dict["object_to_target_z_distance"] = object_to_target_z_distance

        if not self.initializing and self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            obs_dict['reach_err_left'] = (
                self.sim.data.site_xpos[self.target_left]
                - self.sim.data.site_xpos[self.rod_left]
            )
            obs_dict['reach_err_right'] = (
                self.sim.data.site_xpos[self.target_right]
                - self.sim.data.site_xpos[self.rod_right]
            )
            obs_dict['grasp_err_left'] = (
                self.sim.data.site_xpos[self.grasp_left]
                - self.sim.data.site_xpos[self.rod_left]
                - np.array([0, 0, .010])
            )
            obs_dict['grasp_err_right'] = (
                self.sim.data.site_xpos[self.grasp_right]
                - self.sim.data.site_xpos[self.rod_right]
                - np.array([0, 0, .010])
            )
            object_to_hand_dist = 0.5 * (
                np.linalg.norm(obs_dict['grasp_err_left'])
                + np.linalg.norm(obs_dict['grasp_err_right'])
            )
            object_to_target_dist = 0.5 * (
                np.linalg.norm(obs_dict['reach_err_left'])
                + np.linalg.norm(obs_dict['reach_err_right'])
            )
            obs_dict.update({
                "object_sites_to_hand_sites_distance": np.array([object_to_hand_dist]),
                "object_sites_to_target_sites_distance": np.array([object_to_target_dist]),
            })

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        reward_dict = super().get_reward_dict(action, obs_dict)
        object_xyz = obs_dict["object_xyz"].squeeze()
        target_dist = obs_dict["object_to_target_xyz_distance"].squeeze()
        object_to_target_z_distance = obs_dict["object_to_target_z_distance"].squeeze()

        reward_dict["object_to_target_z_distance_reward"] = -object_to_target_z_distance
        reward_dict["table_clear"] = -1 * (obs_dict["grasp_xyz"].squeeze()[2] < 0.75)

        small_bonus = np.zeros(1)
        big_bonus = np.zeros(1)
        if self.object_type in (
                ObjectType.Dodecahedron,
                ObjectType.DodecahedronBasket,
                ObjectType.DodecahedronBulb):
            small_bonus_threshold = 0.82
            big_bonus_threshold = 0.88
        elif self.object_type == ObjectType.Valve3:
            # With the peg in the middle to raise the object
            # small_bonus_threshold = 0.79
            # big_bonus_threshold = 0.88
            # Without the peg
            small_bonus_threshold = 0.77
            big_bonus_threshold = 0.85
        elif self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            small_bonus_threshold = 0.77
            big_bonus_threshold = 0.85

        if self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            reward_dict["object_sites_to_hand_sites_distance_reward"] = (
                -obs_dict["object_sites_to_hand_sites_distance"])
            reward_dict["object_sites_to_target_sites_distance_reward"] = (
                -obs_dict["object_sites_to_target_sites_distance"])
            rod_left = self.sim.data.site_xpos[self.rod_left]
            rod_right = self.sim.data.site_xpos[self.rod_right]

            # Make sure both sites are above the threshold
            if rod_left[2] > small_bonus_threshold and rod_right[2] > small_bonus_threshold:
                small_bonus = 20 + 20 * (target_dist < 0.2)
            if rod_left[2] > big_bonus_threshold and rod_right[2] > big_bonus_threshold:
                big_bonus = 50. + 50. * (target_dist < 0.1)
        elif self.object_type == ObjectType.Valve3:
            reward_dict["object_sites_to_target_sites_distance_reward"] = (
                -obs_dict["object_sites_to_target_sites_distance"])
            valve_1 = self.sim.data.site_xpos[self.valve_1]
            valve_2 = self.sim.data.site_xpos[self.valve_2]
            valve_3 = self.sim.data.site_xpos[self.valve_3]

            if (valve_1[2] > small_bonus_threshold
                    and valve_2[2] > small_bonus_threshold
                    and valve_3[2] > small_bonus_threshold):
                # small_bonus += 20 + 20 * (target_dist < 0.2)
                small_bonus = 20 + 20 * (target_dist < 0.2)
            if (valve_1[2] > big_bonus_threshold
                    and valve_2[2] > big_bonus_threshold
                    and valve_3[2] > big_bonus_threshold):
                big_bonus = 50. + 50. * (target_dist < 0.15)
        else:
            if object_xyz[2] > small_bonus_threshold:
                small_bonus = 20 + 20 * (target_dist < 0.2)
            if object_xyz[2] > big_bonus_threshold:
                big_bonus = 50. + 50. * (target_dist < 0.1)

        reward_dict["small_bonus"] = small_bonus
        reward_dict["big_bonus"] = big_bonus
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = reward_dict["big_bonus"]
        solved = bool(obs_dict["object_to_target_xyz_distance"] < 0.1)
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))

