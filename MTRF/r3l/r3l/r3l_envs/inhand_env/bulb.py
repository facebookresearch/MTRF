# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
from enum import Enum
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple

from r3l import PROJECT_PATH, RESET_STATES_PATH
from r3l.utils.quatmath import quat2euler, euler2quat, mat2quat
from r3l.r3l_envs.inhand_env.base import ObjectType
from r3l.r3l_envs.inhand_env.reposition import SawyerDhandInHandObjectRepositionFixed
from r3l.r3l_envs.inhand_env.flipup import SawyerDhandInHandObjectFlipUpFixed
from r3l.r3l_envs.inhand_env.rotate import SawyerDhandInHandObjectRepositionMidairFixed
from r3l.r3l_envs.inhand_env.multi_phase import SawyerDhandInHandManipulateObjectMultiPhase
from r3l.robot.default_configs import (
    DEFAULT_SAWYER_ROBOT_CONFIG, WRIST_TO_HAND_X_OFFSET,
    MOCAP_POS_PALMUP, MOCAP_POS_PALMDOWN, MOCAP_EULER_PALMUP, MOCAP_EULER_PALMDOWN,
    REPOSITION_SAWYER_ROBOT_CONFIG, PICKUP_SAWYER_ROBOT_CONFIG, FLIPUP_SAWYER_ROBOT_CONFIG)
from r3l.utils.circle_math import circle_distance, circle_distance_mod
from r3l.utils.scripted_motions import match_hand_xyz_euler


BULB_CAMERA_CONFIG = {
    "azimuth": -90,
    "distance": 1.5,
    "elevation": 0,
    "lookat": np.array([0.72, 0.15, 1.0]),
}

BULB_REPOSITION_SAWYER_ROBOT_CONFIG = REPOSITION_SAWYER_ROBOT_CONFIG.copy()
BULB_REPOSITION_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.3 - 0.135, 0.705 + 0.4 - 0.135),
        (0.185 - 0.4, 0.185 + 0.4),
        (0.9, 1.1),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
})

BULB_PICKUP_SAWYER_ROBOT_CONFIG = PICKUP_SAWYER_ROBOT_CONFIG.copy()
BULB_PICKUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMDOWN[0] - 0.15 - 0.135, MOCAP_POS_PALMDOWN[0] + 0.15 - 0.135),
        (MOCAP_POS_PALMDOWN[1] - 0.15, MOCAP_POS_PALMDOWN[1] + 0.15),
        (MOCAP_POS_PALMDOWN[2] - 0.175, MOCAP_POS_PALMDOWN[2] + 0.05),
        # Only allow rotation in one direction
        (MOCAP_EULER_PALMDOWN[0], MOCAP_EULER_PALMDOWN[0]),
        (MOCAP_EULER_PALMDOWN[1], MOCAP_EULER_PALMDOWN[1]),
        (MOCAP_EULER_PALMDOWN[2] - np.pi / 6, MOCAP_EULER_PALMDOWN[2] + np.pi / 6),
    ),
})

BULB_FLIPUP_SAWYER_ROBOT_CONFIG = FLIPUP_SAWYER_ROBOT_CONFIG.copy()
BULB_FLIPUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMDOWN[0] - 0.4, MOCAP_POS_PALMDOWN[0] + 0.4),
        (MOCAP_POS_PALMDOWN[1] - 0.4, MOCAP_POS_PALMDOWN[1] + 0.4),
        # NOTE: This matches the pickup range so that there is no clipping on
        # phase transition
        (MOCAP_POS_PALMDOWN[2] - 0.175, MOCAP_POS_PALMDOWN[2] + 0.05),
        # Only allow rotation in one direction
        (MOCAP_EULER_PALMDOWN[0], MOCAP_EULER_PALMDOWN[0] + np.pi),
        (MOCAP_EULER_PALMDOWN[1], MOCAP_EULER_PALMDOWN[1]),
        (MOCAP_EULER_PALMDOWN[2] - np.pi / 6, MOCAP_EULER_PALMDOWN[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0, 0, 0.01, 0.1, 0, 0.05])
})

BULB_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BULB_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (MOCAP_POS_PALMUP[0] - 0.1 - WRIST_TO_HAND_X_OFFSET, MOCAP_POS_PALMUP[0] + 0.1 - WRIST_TO_HAND_X_OFFSET),
        (MOCAP_POS_PALMUP[1] - 0.1, MOCAP_POS_PALMUP[1] + 0.1),
        # Set the max z height at 1.15 (below 1.2 which is the height of the ceiling)
        (MOCAP_POS_PALMUP[2] - 0.175, MOCAP_POS_PALMUP[2] + 0.15),
        (MOCAP_EULER_PALMDOWN[0] - np.pi, MOCAP_EULER_PALMDOWN[0] + np.pi),
        (MOCAP_EULER_PALMUP[1], MOCAP_EULER_PALMUP[1]),
        (MOCAP_EULER_PALMUP[2] - np.pi / 6, MOCAP_EULER_PALMUP[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0, 0, 0])
})

BULB_RESETFREE_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BULB_RESETFREE_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.3 - 0.135, 0.705 + 0.4 - 0.135),
        (0.185 - 0.35, 0.185 + 0.35),
        # Set the max z height at 1.15 (below 1.2 which is the height of the ceiling)
        (MOCAP_POS_PALMUP[2] - 0.175, MOCAP_POS_PALMUP[2] + 0.15),
        (MOCAP_EULER_PALMDOWN[0] - np.pi, MOCAP_EULER_PALMDOWN[0] + np.pi),
        (MOCAP_EULER_PALMUP[1], MOCAP_EULER_PALMUP[1]),
        (MOCAP_EULER_PALMUP[2] - np.pi / 6, MOCAP_EULER_PALMUP[2] + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0.1, 0, 0.02])
})


class SawyerDhandInHandDodecahedronBulbFixed(SawyerDhandInHandObjectFlipUpFixed):
    HEIGHT_THRESHOLD = 0.85
    WRIST_ANGLE_THRESHOLD = 1.57

    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_target_xy_distance_reward": 1.0,
        # prioritize matching the target height
        "object_to_target_z_distance_reward": 2.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "drop_penalty": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            object_type: ObjectType = ObjectType.DodecahedronBulb,
            camera_configs: dict = None,
            **kwargs
    ):
        self.hoop_center = None
        reset_policy_dirs = [
            str(Path(PROJECT_PATH) / "r3l/r3l_agents/softlearning/SawyerDhandInHandDodecahedronPickupFixed-v0/pickup_trained_with_resets"),
            str(Path(PROJECT_PATH) / "r3l/r3l_agents/softlearning/SawyerDhandInHandDodecahedronFlipUpFixed-v0/flipup_trained_with_resets"),
        ]
        reset_state_pkl_path = str(Path(RESET_STATES_PATH) / "dodecahedron/picked_up_flipped_up.pkl")

        env_params = dict(
            task_name="Bulb",
            object_type=object_type,
            sawyer_config=BULB_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
            reset_policy_directories=reset_policy_dirs,
            reset_state_pkl_path=reset_state_pkl_path,
            camera_configs=BULB_CAMERA_CONFIG,
            # NOTE: `reset_robot=False` prevents the environment from doing a
            # hard reset after the object has been picked up by the loaded policy
            reset_robot=False,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)
        self.hoop_center = self.sim.model.site_name2id('hoop_center')

    def run_reset_policy(self):
        # Do not run the reset policy if we're still above the z threshold
        if self.get_obs_dict()["object_xyz"].squeeze()[2] < 0.85:
            super().run_reset_policy()

    def reset_target(self):
        if self.hoop_center is not None:
            target_xyz = self.sim.data.site_xpos[self.hoop_center].copy() + np.array([0, 0, 0.075])
        else:
            target_xyz = next(self._target_xyz_range)
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

        z_dist = obs_dict["object_to_target_xyz_distance"]
        xy_dist = obs_dict["object_to_target_xy_distance"]

        small_bonus = 1 * (xy_dist < 0.1)
        big_bonus = 10 * (xy_dist < 0.1 and z_dist < 0.105)
        reward_dict["small_bonus"] = np.array([small_bonus])
        reward_dict["big_bonus"] = np.array([big_bonus])

        # penalty for dropping the object
        reward_dict["drop_penalty"] = np.array([-5 * (obs_dict["object_xyz"].squeeze()[2] < 0.85)])

        return reward_dict

    def _release_object(self):
        obs_dict = self.get_obs_dict()
        object_z = obs_dict["object_xyz"][2]
        sawyer_x = obs_dict["mocap_euler"][0]
        if object_z > self.HEIGHT_THRESHOLD:
            # TODO: this logic of replacing the sawyer x euler velocity lim is sketchy
            old_velocity_lim = self.sawyer_robot.wrist_act.mocap_velocity_lim.copy()
            velocity_lim = self.sawyer_robot.wrist_act.mocap_velocity_lim.copy()
            velocity_lim[3] = 0.1
            self.sawyer_robot.wrist_act.mocap_velocity_lim = velocity_lim

            if circle_distance(sawyer_x, MOCAP_EULER_PALMUP[0]) < self.WRIST_ANGLE_THRESHOLD:
                hand_xyz = np.array([0.72 - 3 * 0.135, 0.15, 0.95])
                hand_euler = MOCAP_EULER_PALMUP
                match_hand_xyz_euler(hand_xyz, hand_euler, self)

                hand_xyz = np.array([0.72 - 3 * 0.135, 0.15, 0.95])
                hand_euler = MOCAP_EULER_PALMDOWN
                match_hand_xyz_euler(hand_xyz, hand_euler, self)
            else:
                self.do_simulation(np.zeros(16), 300)

            self.sawyer_robot.wrist_act.mocap_velocity_lim = old_velocity_lim


class SawyerDhandInHandDodecahedronBulbResetFree(SawyerDhandInHandDodecahedronBulbFixed):
    def __init__(
            self,
            reset_every_n_episodes: int = np.inf,
            sawyer_config: dict = BULB_RESETFREE_SAWYER_ROBOT_CONFIG,
            reset_offset: np.ndarray = None,
            release_object_on_reset: bool = True,
            camera_configs: dict = None,
            **kwargs
    ):
        self._release_object_on_reset = release_object_on_reset
        super().__init__(
            reset_every_n_episodes=reset_every_n_episodes,
            sawyer_config=BULB_RESETFREE_SAWYER_ROBOT_CONFIG,
            camera_configs=BULB_CAMERA_CONFIG,
            **kwargs
        )

    def run_reset_policy(self):
        if self._should_reset():
            super().run_reset_policy()

    def reset(self):
        if self._release_object_on_reset:
            self._release_object()
        return super().reset()


class SawyerDhandInHandDodecahedronBulbResetController(SawyerDhandInHandManipulateObjectMultiPhase):
    DISTANCE_THRESHOLD = 0.1
    Z_DISTANCE_THRESHOLD = 0.85
    OBJ_TO_TARGET_DISTANCE_THRESHOLD = 0.2
    SAWYER_ANGLE_THRESHOLD = np.pi / 6

    class Phase(Enum):
        BULB_INSERTION = 0
        PICKUP_FLIPUP = 1

    def __init__(
            self,
            x_range: float = 0.15,
            y_range: float = 0.15,
            object_type: ObjectType = ObjectType.DodecahedronBulb,
            use_dict_obs: bool = True,
            **kwargs
    ):
        if object_type != ObjectType.DodecahedronBulb:
            raise ValueError("This environment requires DodecahedronBulb object type")

        arena_center = np.array([0.72, 0.15, 0.76])
        phase_envs_params = [
            # Bulb insertion
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "BulbFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [arena_center],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, 0])],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs,
                    'reset_robot': False,
                },
            },
            # Reset controller = reposition + pickup + flipup
            {
                "domain": f"SawyerDhandInHandDodecahedron",
                "task": "PickupFlipUpFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_RESETFREE_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "UniformRange",
                        "values": [
                            np.array([arena_center[0] - x_range, arena_center[1] - y_range, arena_center[2]]),
                            np.array([arena_center[0] + x_range, arena_center[1] + y_range, arena_center[2]]),
                        ],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([arena_center[0], arena_center[1], 0.975])],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs,
                },
            },
        ]

        super().__init__(
            phase_envs_params=phase_envs_params,
            object_type=object_type,
            use_dict_obs=use_dict_obs,
            camera_configs=BULB_CAMERA_CONFIG,
            **kwargs
        )

    def task_graph(self):
        Phase = self.Phase

        if self.phase == Phase.BULB_INSERTION.value:
            self._envs[Phase.BULB_INSERTION.value]._release_object()

        # Construct a new adjacency matrix
        self.task_adj_matrix = np.zeros((self.num_phases, self.num_phases))

        rc_env = self._envs[Phase.PICKUP_FLIPUP.value]
        rc_obs = rc_env.get_obs_dict()
        rc_rew = rc_env.get_reward_dict(None, rc_obs)

        obs_dict = self.get_obs_dict()
        object_xyz = obs_dict["object_xyz"]
        object_to_target_xy_dist = obs_dict["object_to_target_xy_distance"]
        sawyer_to_target_circle_dist = -(
            rc_rew["sawyer_to_target_x_circle_distance_reward"]
            + rc_rew["sawyer_to_target_z_circle_distance_reward"]
        )

        # above height, within xy, within sawyer x
        if (object_xyz[2] <= self.Z_DISTANCE_THRESHOLD
                or object_to_target_xy_dist >= self.DISTANCE_THRESHOLD
                or sawyer_to_target_circle_dist >= self.SAWYER_ANGLE_THRESHOLD):
            self.task_adj_matrix[Phase.PICKUP_FLIPUP.value][Phase.PICKUP_FLIPUP.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.PICKUP_FLIPUP.value] = 1.0
        else:
            self.task_adj_matrix[Phase.PICKUP_FLIPUP.value][Phase.BULB_INSERTION.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.BULB_INSERTION.value] = 1.0

        if self._verbose:
            print("TASK ADJACENCY MATRIX:\n", self.task_adj_matrix)

        # The `self.phase` row of the adjacency matrix gives you next-phase
        # transition probabilities.
        task_adj_list = self.task_adj_matrix[self.phase]
        assert np.sum(task_adj_list) == 1
        if self._random_task_graph:
            next_phase = self.np_random.choice(self.num_phases) # Removed the probability from the choice list
        else:
            next_phase = self.np_random.choice(self.num_phases, p=task_adj_list)

        return next_phase


class SawyerDhandInHandDodecahedronBulbPhased(SawyerDhandInHandManipulateObjectMultiPhase):
    """
    5 phases = Reposition + Pickup + Flip Up + Bulb Slotting + Perturb
    """
    DISTANCE_THRESHOLD = 0.1
    Z_DISTANCE_THRESHOLD = 0.85
    OBJ_TO_TARGET_DISTANCE_THRESHOLD = 0.2
    SAWYER_ANGLE_THRESHOLD = np.pi / 6

    class Phase(Enum):
        REPOSITION = 0
        PICKUP = 1
        FLIPUP = 2
        BULB_INSERTION = 3
        PERTURB = 4

    def __init__(
            self,
            x_range: float = 0.15,
            y_range: float = 0.15,
            object_type: ObjectType = ObjectType.DodecahedronBulb,
            n_bins: int = 100,
            phase_to_color_idx: dict = None,
            use_dict_obs: bool = True,
            **kwargs
    ):
        if object_type != ObjectType.DodecahedronBulb:
            raise ValueError("This environment requires DodecahedronBulb object type")

        Phase = self.Phase
        if phase_to_color_idx is None:
            self.phase_to_color_idx = {
                Phase.REPOSITION.value: np.array([231/255, 76/255, 60/255, 1]),
                Phase.PICKUP.value: np.array([52/255, 152/255, 219/255, 1]),
                Phase.FLIPUP.value: np.array([230/255, 126/255, 34/255, 1]),
                Phase.BULB_INSERTION.value: np.array([46/255, 204/255, 113/255, 1]),
                Phase.PERTURB.value: np.array([44/255, 62/255, 80/255, 1]),
            }
        else:
            self.phase_to_color_idx = phase_to_color_idx

        arena_center = np.array([0.72, 0.15, 0.76])
        phase_envs_params = [
            # Reposition
            {
                "domain": f"SawyerDhandInHandDodecahedron",
                "task": "RepositionFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_REPOSITION_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "UniformRange",
                        "values": [
                            np.array([arena_center[0] - x_range, arena_center[1] - y_range, arena_center[2]]),
                            np.array([arena_center[0] + x_range, arena_center[1] + y_range, arena_center[2]]),
                        ],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [arena_center],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs,
                },
            },
            # Pickup
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "PickupFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_PICKUP_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [arena_center],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([arena_center[0], arena_center[1], 0.92])],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs
                },
            },
            # Flip Up
            {
                "domain": f"SawyerDhandInHandDodecahedron",
                "task": "FlipUpFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_FLIPUP_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0.72, 0.15, 0.78])],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([arena_center[0], arena_center[1], 0.975])],
                    },
                    'reset_robot': False,
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs,
                },
            },
            # Bulb insertion
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "BulbFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [arena_center],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, 0])],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs,
                    'reset_robot': False,
                },
            },
            # XY Perturbation
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "RepositionFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BULB_REPOSITION_SAWYER_ROBOT_CONFIG,
                    "collect_bin_counts": True,
                    "n_bins": n_bins,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [arena_center],
                    },
                    "init_euler_range_params": {
                        "type": "UniformRange",
                        "values": [np.array([0, 0, -np.pi]), np.array([0, 0, np.pi])],
                    },
                    # Hide the target for the perturbation phase
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.zeros(3)],
                    },
                    'reward_keys_and_weights': {
                        "xy_discrete_sqrt_count_reward": 1.0,
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs
                },
            },
        ]

        super().__init__(
            phase_envs_params=phase_envs_params,
            object_type=object_type,
            use_dict_obs=use_dict_obs,
            camera_configs=BULB_CAMERA_CONFIG,
            **kwargs
        )

    def task_graph(self):
        Phase = self.Phase

        if self.phase == Phase.BULB_INSERTION.value:
            self._envs[Phase.BULB_INSERTION.value]._release_object()

        # Construct a new adjacency matrix
        self.task_adj_matrix = np.zeros((self.num_phases, self.num_phases))

        # Collect environment obs and reward dicts to determine transitions
        repos_env = self._envs[Phase.REPOSITION.value]
        repos_obs = repos_env.get_obs_dict()

        repos_xy_dist = repos_obs["object_to_target_xy_distance"]
        object_xyz = repos_obs["object_xyz"]
        # bulb_env = self._envs[Phase.BULB_INSERTION.value]
        # obj_to_bulb_target_dist = bulb_env.get_obs_dict()["object_to_target_xyz_distance"]

        flipup_env = self._envs[Phase.FLIPUP.value]
        flipup_obs = flipup_env.get_obs_dict()
        flipup_reward = flipup_env.get_reward_dict(None, flipup_obs)
        sawyer_to_target_circle_dist = -(
            flipup_reward["sawyer_to_target_x_circle_distance_reward"]
            + flipup_reward["sawyer_to_target_z_circle_distance_reward"]
        )
        flipup_xy_dist = flipup_obs["object_to_target_xy_distance"]

        if object_xyz[2] <= self.Z_DISTANCE_THRESHOLD and repos_xy_dist >= self.DISTANCE_THRESHOLD:
            """
            If the object is on the table (not picked up) and the object is more
            than `DISTANCE_THRESHOLD` away from the center
            -> PERTURB        (with probability p_repos)
            -> REPOSITION     (with probability 1 - p_repos)
            """
            # All other phases should transition to reposition with probability 1
            self.task_adj_matrix[Phase.PERTURB.value][Phase.REPOSITION.value] = 1.0
            self.task_adj_matrix[Phase.PICKUP.value][Phase.REPOSITION.value] = 1.0
            self.task_adj_matrix[Phase.FLIPUP.value][Phase.REPOSITION.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.REPOSITION.value] = 1.0

            if not self._perturb_off:
                # Reposition -> Perturb if not successful
                self.task_adj_matrix[Phase.REPOSITION.value][Phase.PERTURB.value] = 1.0
            else:
                self.task_adj_matrix[Phase.REPOSITION.value][Phase.REPOSITION.value] = 1.0
        elif object_xyz[2] <= self.Z_DISTANCE_THRESHOLD and repos_xy_dist < self.DISTANCE_THRESHOLD:
            """
            Otherwise, transition into the pickup task.
            Condition should be that the object is within `DISTANCE_THRESHOLD` in
            xy distance, and within `ANGLE_THRESHOLD` in circle distance.
            """
            # All phases transition to reorient with probability 1, including self-loop
            self.task_adj_matrix[Phase.PERTURB.value][Phase.PICKUP.value] = 1.0
            self.task_adj_matrix[Phase.REPOSITION.value][Phase.PICKUP.value] = 1.0
            self.task_adj_matrix[Phase.PICKUP.value][Phase.PICKUP.value] = 1.0
            self.task_adj_matrix[Phase.FLIPUP.value][Phase.PICKUP.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.PICKUP.value] = 1.0
        elif ((object_xyz[2] > self.Z_DISTANCE_THRESHOLD and sawyer_to_target_circle_dist > self.SAWYER_ANGLE_THRESHOLD)
                or (object_xyz[2] > self.Z_DISTANCE_THRESHOLD
                    and sawyer_to_target_circle_dist <= self.SAWYER_ANGLE_THRESHOLD
                    and flipup_xy_dist >= self.OBJ_TO_TARGET_DISTANCE_THRESHOLD)):
            self.task_adj_matrix[Phase.PERTURB.value][Phase.FLIPUP.value] = 1.0
            self.task_adj_matrix[Phase.REPOSITION.value][Phase.FLIPUP.value] = 1.0
            self.task_adj_matrix[Phase.PICKUP.value][Phase.FLIPUP.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.FLIPUP.value] = 1.0
            self.task_adj_matrix[Phase.FLIPUP.value][Phase.FLIPUP.value] = 1.0
        elif (object_xyz[2] > self.Z_DISTANCE_THRESHOLD
                and sawyer_to_target_circle_dist <= self.SAWYER_ANGLE_THRESHOLD
                and flipup_xy_dist < self.OBJ_TO_TARGET_DISTANCE_THRESHOLD):
            self.task_adj_matrix[Phase.PERTURB.value][Phase.BULB_INSERTION.value] = 1.0
            self.task_adj_matrix[Phase.REPOSITION.value][Phase.BULB_INSERTION.value] = 1.0
            self.task_adj_matrix[Phase.PICKUP.value][Phase.BULB_INSERTION.value] = 1.0
            self.task_adj_matrix[Phase.FLIPUP.value][Phase.BULB_INSERTION.value] = 1.0
            self.task_adj_matrix[Phase.BULB_INSERTION.value][Phase.BULB_INSERTION.value] = 1.0
        else:
            print("Should not have reached this condition with: "
                  f"repos_xy_dist={repos_xy_dist}, object_xyz={object_xyz}")
            raise NotImplementedError

        if self._verbose:
            print("TASK ADJACENCY MATRIX:\n", self.task_adj_matrix)

        # The `self.phase` row of the adjacency matrix gives you next-phase
        # transition probabilities.
        task_adj_list = self.task_adj_matrix[self.phase]
        assert np.sum(task_adj_list) == 1
        if self._random_task_graph:
            next_phase = self.np_random.choice(self.num_phases) # Removed the probability from the choice list
        else:
            next_phase = self.np_random.choice(self.num_phases, p=task_adj_list)

        return next_phase

    def set_color_phase(self, phase):
        for phase_idx in range(self.num_phases):
            if phase_idx == phase:
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(f'pi{phase}')] = self.phase_to_color_idx[phase]
            else:
                darker_color = self.phase_to_color_idx[phase_idx] * 0.25
                self.sim.model.geom_rgba[self.sim.model.geom_name2id(f'pi{phase_idx}')] = darker_color

    def configure_phase(self, phase_idx):
        super().configure_phase(phase_idx)
        if self._verbose:
            print(f"NEXT PHASE = {phase_idx}")
        self.set_color_phase(phase_idx)
