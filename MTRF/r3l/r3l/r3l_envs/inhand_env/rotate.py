# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple

from r3l import PROJECT_PATH
from r3l.r3l_envs.inhand_env.base import ObjectType
from r3l.r3l_envs.inhand_env.reorient import SawyerDhandInHandObjectReorientFixed
from r3l.robot.default_configs import MIDAIR_SAWYER_ROBOT_CONFIG, ARM_QPOS_PALMUP
from r3l.utils.circle_math import circle_distance, circle_distance_mod
from r3l.utils.quatmath import quat2euler, euler2quat, mat2quat


class SawyerDhandInHandObjectReorientMidairFixed(SawyerDhandInHandObjectReorientFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_target_xyz_distance_reward": 5.0,
        "object_to_target_mod_120_circle_distance_reward": 1.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "drop_penalty": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            sawyer_goal_offset: np.ndarray = np.array([0.15, 0, 0.05]),
            use_grasp_xy_target: bool = False,
            use_sawyer_offset_target: bool = False,
            reposition_only: bool = False,
            **kwargs
    ):
        self._use_sawyer_offset_target = use_sawyer_offset_target
        self._use_grasp_xy_target = use_grasp_xy_target
        self._reposition_only = reposition_only
        if kwargs.get("object_type", None) == ObjectType.Valve3:
            reset_policy_dirs = [
                (Path(PROJECT_PATH)
                / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3PickupFixed-v0/pickup_raised_valve"),
                (Path(PROJECT_PATH)
                / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3FlipUpFixed-v0/flipup_raised_valve_resetfree"),
            ]
        else:
            print("Object type doesn't have reset policy")
            reset_policy_dirs = []

        env_params = dict(
            task_name="Reorient Midair",
            sawyer_config=MIDAIR_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
            target_xyz_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0.72, 0.15, 1.0])]
            },
            target_euler_range_params={
                "type": "UniformRange",
                "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
            },
            reset_policy_directories=reset_policy_dirs,
            # NOTE: `reset_robot=False` prevents the environment from doing a
            # hard reset after the object has been picked up by the loaded policy
            reset_robot=False,
        )
        env_params.update(kwargs)

        super().__init__(**env_params)
        self.init_qpos[:7] = ARM_QPOS_PALMUP
        if sawyer_goal_offset is not None:
            self._sawyer_goal_offset = sawyer_goal_offset
        else:
            # Wrist to hand offset of ~0.15, and a little bit of Z offset
            self._sawyer_goal_offset = np.array([0.15, 0, 0.05])

    def run_reset_policy(self):
        # Do not run the reset policy if we're still above the z threshold
        if self.get_obs_dict()["object_xyz"].squeeze()[2] < 0.85:
            super().run_reset_policy()

    def reset_target(self):
        obs_dict = self.get_obs_dict()
        if self._use_sawyer_offset_target:
            # Target is an offset from the mocap position
            target_xyz = obs_dict['mocap_pos'] + self._sawyer_goal_offset
        elif self._reposition_only:
            target_xyz = next(self._target_xyz_range)
        else:
            # Fixed z, but change the target to match the grasp xy
            target_xyz = next(self._target_xyz_range)
            if self._use_grasp_xy_target:
                target_xyz[:2] = obs_dict["grasp_xyz"][:2]

        target_euler = next(self._target_euler_range)
        self.sim.model.body_pos[self.target_bid] = target_xyz
        self.sim.model.body_quat[self.target_bid] = euler2quat(target_euler)
        self._last_target_xyz = target_xyz
        self._last_target_euler = target_euler

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        reward_dict = super().get_reward_dict(action, obs_dict)
        xyz_dist = obs_dict["object_to_target_xyz_distance"]
        circle_dist = obs_dict["object_to_target_mod_120_circle_distance"]

        # Check if it's kinda close in xyz and close in mod circle distance
        small_bonus = 10 * (xyz_dist < 0.1 and circle_dist < 0.25)
        # Check if it's really close in those 2
        big_bonus = 50 * (xyz_dist < 0.075 and circle_dist < 0.1)

        reward_dict["small_bonus"] = small_bonus
        reward_dict["big_bonus"] = big_bonus

        # penalty for dropping the valve
        reward_dict["drop_penalty"] = np.array([-5 * (obs_dict["object_xyz"].squeeze()[2] < 0.85)])

        return reward_dict


class SawyerDhandInHandObjectRepositionMidairFixed(SawyerDhandInHandObjectReorientMidairFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        # Same as above, but don't include the angle reward for reorienting.
        "object_to_target_xyz_distance_reward": 5.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "drop_penalty": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            target_range: np.ndarray = np.array([0.1, 0.1, 0]),
            target_four_corners: bool = True,
            **kwargs
    ):
        center = np.array([0.72, 0.15, 1.0])
        self._target_range = target_range
        if target_four_corners:
            target_xyz_params = {
                "type": "DiscreteRange",
                "values": [
                    center + np.array([-0.1, -0.1, 0]),
                    center + np.array([-0.1, 0.1, 0]),
                    center + np.array([0.1, 0.1, 0]),
                    center + np.array([0.1, -0.1, 0]),
                ],
                "choosing_strategy": "cycle",
            }
        else:
            target_xyz_params = {
                "type": "UniformRange",
                "values": [
                    center - target_range,
                    center + target_range,
                ]
            }
        env_params = dict(
            task_name="Reposition Midair",
            target_xyz_range_params=target_xyz_params,
            target_euler_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0, 0, 0])],
            },
            reposition_only=True,  # TODO: this flag is a little confusing
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
        xyz_dist = obs_dict["object_to_target_xyz_distance"]

        small_bonus = 10 * (xyz_dist < 0.1)
        big_bonus = 50 * (xyz_dist < 0.05)

        reward_dict["small_bonus"] = small_bonus
        reward_dict["big_bonus"] = big_bonus

        return reward_dict


# Baseline
from r3l.robot.default_configs import MIDAIR_RESETFREE_SAWYER_ROBOT_CONFIG
class SawyerDhandInHandObjectReorientMidairResetFree(SawyerDhandInHandObjectReorientMidairFixed):
    def __init__(self, sawyer_config=MIDAIR_RESETFREE_SAWYER_ROBOT_CONFIG, **kwargs):
        env_params = dict(
            reset_every_n_episodes=np.inf, sawyer_config=sawyer_config,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

    def run_reset_policy(self):
        # Only run the reset policy on the first iteration
        if self._should_reset():
            super().run_reset_policy()


try:
    from softlearning.policies.utils import get_policy_from_variant
    from softlearning.environments.utils import get_environment_from_params
    from softlearning.environments.adapters.gym_adapter import GymAdapter
    from softlearning.models.utils import flatten_input_structure
except:
    print('Error: softlearning package not found.')
from r3l.robot.default_configs import MOCAP_EULER_PALMDOWN

# Requires a few mocap ranges to be changed:
# REPOSITION_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
# REPOSITION_SAWYER_ROBOT_CONFIG.update({
#     "mocap_range": (
#         (0.705 - 0.4, 0.705 + 0.4),
#         (0.185 - 0.4, 0.185 + 0.4),
#         (1.0 - 0.1, 1.0 + 0.1),
#         (-np.pi, -np.pi),
#         (0, 0),
#         (0 - np.pi / 3, 0 + np.pi / 3),
#     ),
# })
# PICKUP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
# PICKUP_SAWYER_ROBOT_CONFIG.update({
#     "mocap_range": (
#         # 0.135 is the wrist to hand offset
#         (0.705 - 0.15 - 0.135, 0.705 + 0.15 - 0.135),
#         (0.185 - 0.15, 0.185 + 0.15),
#         (1.0 - 0.175, 1.0 + 0.15),
#         (-np.pi, -np.pi),
#         (0, 0),
#         (0 - np.pi / 6, 0 + np.pi / 6),
#     ),
# })
class SawyerDhandInHandObjectReorientMidairSlottedResetFree(SawyerDhandInHandObjectReorientMidairFixed):
    MAX_TRIES = 5

    def __init__(
            self,
            reset_policy_directories=[],
            reset_robot=True,
            **kwargs
    ):
        super().__init__(
            reset_policy_directories=[],
            reset_robot=False,
            target_xyz_range_params={
                "type": "UniformRange",
                "values": [np.array([0.72 - 0.1, 0.15 - 0.1, 1.0]), np.array([0.72 + 0.1, 0.15 + 0.1, 1.0])]
            },
            target_euler_range_params={
                "type": "DiscreteRange",
                "values": [np.array([0, 0, np.pi])],
            },
            **kwargs
        )

        # Load in trained reset controllers for reposition, reorient, pickup
        # Policy is on deepthought
        path = Path("/home/justinvyu/ray_results/gym/SawyerDhandInHandValve3/RepositionReorientPickupPerturbResetFree-v0/2020-10-28T19-43-42-4phase_fixedsawyerxzrange_newobskeys_repos_to_middle/id=8f3e609b-seed=470_2020-10-28_19-43-43pj_w9_hx")
        ckpt_path = path / "checkpoint_695"
        policy_params_path = ckpt_path / "policy_params.pkl"

        params_path = path / "params.pkl"

        with open(policy_params_path, "rb") as f:
            policy_params_data = pickle.load(f)

        with open(params_path, "rb") as f:
            variant = pickle.load(f)

        env_params = variant["environment_params"]["evaluation"]
        env_kwargs = env_params.pop("kwargs", {})
        env_kwargs["sim"] = self.sim
        env_params["kwargs"] = env_kwargs
        env_params["kwargs"]["commanded_phase_changes"] = False
        env = get_environment_from_params(env_params)
        # Create environment as softlearning expects it for policy initialization
        env = GymAdapter(None, None, env=env)

        # Never transition into perturbation for the reset controllers
        if hasattr(env, "turn_perturb_off"):
            env.turn_perturb_off()

        reset_horizons = []
        policies = []
        wrapped_policies = []

        for phase in range(env.num_phases):
            phase_env = env.unwrapped._envs[phase]
            policy = get_policy_from_variant(variant, GymAdapter(None, None, env=phase_env))
            policy.set_weights(policy_params_data[phase])
            policies.append(policy)

            # Save some time by taking a max reset horizon of 50 steps
            horizon = min(
                50,
                variant.get('sampler_params', {}).get('kwargs', {}).get('max_path_length', 50)
            )
            reset_horizons.append(horizon)
            wrapped_policies.append(self.wrap_policy(policy))

        env.unwrapped._training_phases = []
        env.unwrapped._phase_policies = wrapped_policies
        self.reset_env = env
        self.reset_horizons = reset_horizons

        flipup_path = (
            Path(PROJECT_PATH)
            / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3FlipUpFixed-v0/flipup_slotted_resetfree"
        )
        self.flipup_policy = self._load_policy(flipup_path)

    def wrap_policy(self, policy):
        def wrapped_policy(obs_dict):
            feed_dict = {
                key: obs_dict[key][None, ...]
                for key in policy.observation_keys
            }
            observation = flatten_input_structure(feed_dict)
            with policy.set_deterministic(True):
                action = policy.actions_np(observation)[0]
            return action
        return wrapped_policy

    def run_reset_policy(self):
        # frames = []
        if self.reset_env.get_obs_dict()["object_xyz"][2] > 0.85:
            # If we're already picked up, don't run the reset policies again
            return

        phase = self.reset_env.update_phase()
        dummy_action = np.zeros(self.reset_env.action_space.shape)
        # success = False
        for _ in range(self.MAX_TRIES):
            # If we've managed to pickup, we're good to go
            if self.reset_env.get_obs_dict()["object_xyz"][2] > 0.85:
                # success = True
                break

            phase = self.reset_env.update_phase()
            # print("phase = ", phase)
            self.reset_env.active_env.reset_robot()
            for _ in range(self.reset_horizons[phase]):
                # Action fed in doesn't actually get used
                self.reset_env.step(dummy_action)
                # frames.append(self.reset_env.render(mode="rgb_array", width=480, height=480))

        for _ in range(self._reset_horizons[0]):
            self._reset_envs[0].step(self.flipup_policy(self._reset_envs[0].get_obs_dict()))
            # frames.append(self.reset_env.render(mode="rgb_array", width=480, height=480))

        # print("success = ", success)
        # import skvideo.io
        # if frames:
        #     skvideo.io.vwrite("phase_env_loaded_policies.mp4", np.asarray(frames))


class SawyerDhandInHandObjectRepositionMidairSlottedResetFree(SawyerDhandInHandObjectRepositionMidairFixed):
    def __init__(
            self,
            reset_policy_directories=[],
            reset_robot=True,
            num_reset_attempts: int = 6,
            **kwargs
    ):
        super().__init__(
            reset_policy_directories=[],
            # NOTE: Turn off the robot reset by default because we're assuming the
            # object is starting in hand from a slotted controller
            reset_robot=False,
            **kwargs
        )
        self._num_reset_attempts = num_reset_attempts

        # Load in trained reset controllers for reposition, pickup, flipup
        # Policy is on newton5
        from r3l import PROJECT_PATH
        path = Path(PROJECT_PATH) / "r3l/r3l_agents/softlearning/SawyerDhandInHandValve3AllPhasesResetFree-v1/seed_9369"
        ckpt_path = path / "checkpoint_800"
        policy_params_path = ckpt_path / "policy_params.pkl"
        params_path = path / "params.pkl"

        with open(policy_params_path, "rb") as f:
            policy_params_data = pickle.load(f)

        with open(params_path, "rb") as f:
            variant = pickle.load(f)

        env_params = variant["environment_params"]["evaluation"]
        env_kwargs = env_params.pop("kwargs", {})
        env_kwargs["sim"] = self.sim
        env_params["kwargs"] = env_kwargs
        env_params["kwargs"]["commanded_phase_changes"] = False
        env = get_environment_from_params(env_params)
        # Create environment as softlearning expects it for policy initialization
        env = GymAdapter(None, None, env=env)

        # Never transition into perturbation for the reset controllers
        if hasattr(env, "turn_perturb_off"):
            env.turn_perturb_off()

        reset_horizons = []
        policies = []
        wrapped_policies = []

        for phase in range(env.num_phases):
            phase_env = env.unwrapped._envs[phase]
            policy = get_policy_from_variant(variant, GymAdapter(None, None, env=phase_env))
            policy.set_weights(policy_params_data[phase])
            policies.append(policy)

            # Save some time by taking a max reset horizon of 50 steps
            horizon = min(
                50,
                variant.get('sampler_params', {}).get('kwargs', {}).get('max_path_length', 50)
            )
            reset_horizons.append(horizon)
            wrapped_policies.append(self.wrap_policy(policy))

        env.unwrapped._training_phases = []
        env.unwrapped._phase_policies = wrapped_policies
        self.reset_env = env
        self.reset_horizons = reset_horizons

    def wrap_policy(self, policy):
        def wrapped_policy(obs_dict):
            feed_dict = {
                key: obs_dict[key][None, ...]
                for key in policy.observation_keys
            }
            observation = flatten_input_structure(feed_dict)
            with policy.set_deterministic(True):
                action = policy.actions_np(observation)[0]
            return action
        return wrapped_policy

    def run_reset_policy(self):
        # frames = []
        phase = self.reset_env.update_phase()
        if phase == 4:
            # If we're already picked up, don't run the reset policies again
            return

        phase = self.reset_env.update_phase()
        dummy_action = np.zeros(self.reset_env.action_space.shape)
        # success = False
        for _ in range(self._num_reset_attempts):
            # If we've managed to pickup, we're good to go
            phase = self.reset_env.update_phase()
            if self.reset_env.phase == 4:
                # success = True
                break

            # print("phase = ", phase)
            self.reset_env.active_env.reset_robot()
            for _ in range(self.reset_horizons[phase]):
                # Action fed in doesn't actually get used
                self.reset_env.step(dummy_action)
                # frames.append(self.reset_env.render(mode="rgb_array", width=480, height=480))

        # print("success = ", success)
        # import skvideo.io
        # if frames:
        #     skvideo.io.vwrite("phase_env_loaded_policies.mp4", np.asarray(frames))
