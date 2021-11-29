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
from r3l.r3l_envs.inhand_env.pickup import SawyerDhandInHandObjectPickupFixed
from r3l.r3l_envs.inhand_env.pose import SawyerDhandInHandObjectPoseFixed
from r3l.r3l_envs.inhand_env.multi_phase import SawyerDhandInHandManipulateObjectMultiPhase
from r3l.robot.default_configs import DEFAULT_SAWYER_ROBOT_CONFIG
from r3l.utils.circle_math import circle_distance, circle_distance_mod


BASKET_REPOSITION_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BASKET_REPOSITION_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.3 - 0.135, 0.705 + 0.4 - 0.135),
        (0.185 - 0.4, 0.185 + 0.4),
        (0.9, 0.9 + 0.05),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
})

BASKET_PICKUP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BASKET_PICKUP_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        # 0.135 is the wrist to hand offset
        (0.705 - 0.25 - 0.135, 0.705 - 0.135),
        (0.185 - 0.15, 0.185 + 0.15),
        (1.0 - 0.175, 1.0 + 0.3),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.001, 0.001, 0.01, 0, 0, 0]),
})

BASKET_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BASKET_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        # 0.135 is the wrist to hand offset
        (0.705 - 0.25 - 0.135, 0.705 + 0.25 - 0.135),
        (0.185 - 0.15, 0.185 + 0.15), # middle +/- 0.1
        (1.0 + 0.18, 1.0 + 0.4),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
    "mocap_velocity_lim": np.array([0.01, 0.001, 0.001, 0, 0, 0]),
})

# BASKET_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
# BASKET_SAWYER_ROBOT_CONFIG.update({
#     "mocap_range": (
#         # 0.135 is the wrist to hand offset
#         (0.705 - 0.25 - 0.135, 0.705 + 0.25 - 0.135),
#         (0.185 - 0.15, 0.185 + 0.15), # middle +/- 0.1
#         (1.0 + 0.18, 1.0 + 0.4),
#         (-np.pi, -np.pi),
#         (0, 0),
#         (0 - np.pi / 6, 0 + np.pi / 6),
#     ),
#     "mocap_velocity_lim": np.array([0.01, 0.01, 0.01, 0, 0, 0]),
# })

BASKET_RESETFREE_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BASKET_RESETFREE_SAWYER_ROBOT_CONFIG.update({
    "mocap_range": (
        (0.705 - 0.3 - 0.135, 0.705 + 0.4 - 0.135),
        (0.185 - 0.4, 0.185 + 0.4),
        (0.9, 1.4),
        (-np.pi, -np.pi),
        (0, 0),
        (0 - np.pi / 6, 0 + np.pi / 6),
    ),
})

BASKET_DROP_SAWYER_ROBOT_CONFIG = DEFAULT_SAWYER_ROBOT_CONFIG.copy()
BASKET_DROP_SAWYER_ROBOT_CONFIG.update({
    # Allow all kinds of configurations, but don't let the Sawyer move
    "mocap_range": (
        # 0.135 is the wrist to hand offset
        (0.705 - 0.25 - 0.135, 0.705 + 0.25 - 0.135), # middle +/- 0.25
        (0.185 - 0.25, 0.185 + 0.25), # middle +/- 0.25
        (0.92, 1.0 + 0.3),
        (-np.pi - np.pi, -np.pi + np.pi),
        (0 - np.pi, 0 + np.pi),
        (0 - np.pi, 0 + np.pi),
    ),
    "mocap_velocity_lim": np.array([0.005, 0.005, 0.01, 0, 0, 0]),
})


class SawyerDhandInHandDodecahedronBasketFixed(SawyerDhandInHandObjectPickupFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "object_to_target_xy_distance_reward": 2.0,
        "object_to_target_z_distance_reward": 2.0,
        "small_bonus": 1.0,
        "big_bonus": 1.0,
        "drop_penalty": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            object_type: ObjectType = ObjectType.DodecahedronBasket,
            **kwargs
    ):
        self.hoop_center = None
        reset_policy_dirs = [
            str(Path(PROJECT_PATH) / "r3l/r3l_agents/softlearning/SawyerDhandInHandDodecahedronPickupFixed-v0/pickup_trained_with_resets"),
        ]
        reset_state_pkl_path = str(Path(RESET_STATES_PATH) / "dodecahedron_basket/picked_up_higher.pkl")

        env_params = dict(
            task_name="Basket",
            object_type=object_type,
            sawyer_config=BASKET_SAWYER_ROBOT_CONFIG,
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
            reset_state_pkl_path=reset_state_pkl_path,
            # NOTE: `reset_robot=False` prevents the environment from doing a
            # hard reset after the object has been picked up by the loaded policy
            reset_robot=False,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)
        if object_type == ObjectType.DodecahedronBasket:
            self.hoop_center = self.sim.model.site_name2id('hoop_center')

    def run_reset_policy(self):
        # Do not run the reset policy if we're still above the z threshold
        if self.get_obs_dict()["object_xyz"].squeeze()[2] < 0.85:
            super().run_reset_policy()

    def reset_target(self):
        if self.hoop_center is not None:
            target_xyz = self.sim.data.site_xpos[self.hoop_center].copy() + np.array([0, 0, 0.175])
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

        # penalty for dropping the object
        reward_dict["drop_penalty"] = np.array([-1 * (obs_dict["object_xyz"].squeeze()[2] < 0.85)])
        reward_dict["small_bonus"] = 10 * (obs_dict["object_to_target_xyz_distance"].squeeze() < 0.1)
        reward_dict["big_bonus"] = 50 * (obs_dict["object_to_target_xyz_distance"].squeeze() < 0.05)

        return reward_dict


class SawyerDhandInHandDodecahedronBasketResetFree(SawyerDhandInHandDodecahedronBasketFixed):
    HEIGHT_THRESHOLD = 0.85

    def __init__(
            self,
            reset_every_n_episodes: int = np.inf,
            sawyer_config: dict = BASKET_RESETFREE_SAWYER_ROBOT_CONFIG,
            reset_offset: np.ndarray = None,
            **kwargs
    ):
        super().__init__(
            reset_every_n_episodes=np.inf,
            sawyer_config=BASKET_RESETFREE_SAWYER_ROBOT_CONFIG,
            reset_offset=np.array([-0.15, 0, 0.125]),
            **kwargs
        )

    def _release_object(self):
        if self._verbose:
            print("Dropping object")
        self.do_simulation(np.zeros(16), 300)

    def run_reset_policy(self):
        if self._should_reset():
            super().run_reset_policy()

    def reset(self):
        if self.get_obs_dict()["object_xyz"][2] > self.HEIGHT_THRESHOLD:
            self._release_object()
        return super().reset()


class SawyerDhandInHandDodecahedronBasketDropFixed(SawyerDhandInHandObjectPoseFixed):
    DEFAULT_REWARD_KEYS_AND_WEIGHTS = {
        "pose_distance_reward": 1.0,
        "target_to_hand_xyz_distance_reward": 1.0,
    }

    def __init__(
            self,
            reward_keys_and_weights: dict = DEFAULT_REWARD_KEYS_AND_WEIGHTS,
            object_type: ObjectType = ObjectType.DodecahedronBasket,
            **kwargs
    ):
        self.hoop_center = None
        env_params = dict(
            object_type=object_type,
            sawyer_config=BASKET_DROP_SAWYER_ROBOT_CONFIG,
            reward_keys_and_weights=reward_keys_and_weights,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)
        self.hoop_center = self.sim.model.site_name2id('hoop_center')

    def reset_target(self):
        if self.hoop_center is not None:
            target_xyz = self.sim.data.site_xpos[self.hoop_center].copy() + np.array([0, 0, 0.15])
        else:
            target_xyz = next(self._target_xyz_range)
        target_euler = next(self._target_euler_range)
        self.sim.model.body_pos[self.target_bid] = target_xyz
        self.sim.model.body_quat[self.target_bid] = euler2quat(target_euler)
        self._last_target_xyz = target_xyz
        self._last_target_euler = target_euler

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        score = reward_dict["pose_distance_reward"] + reward_dict["target_to_hand_xyz_distance_reward"]
        solved = bool(obs_dict["pose_dist"] < 0.05 and obs_dict["target_to_hand_xyz_distance"] < 0.1)
        return collections.OrderedDict((
            ("score", np.array([score])),
            ("solved", np.array([solved])),
        ))


class SawyerDhandInHandDodecahedronBasketResetController(SawyerDhandInHandManipulateObjectMultiPhase):
    DISTANCE_THRESHOLD = 0.3
    Z_THRESHOLD = 0.85

    class Phase(Enum):
        BASKET = 0
        PICKUP = 1

    def __init__(
            self,
            x_range: float = 0.15,
            y_range: float = 0.15,
            object_type: ObjectType = ObjectType.DodecahedronBasket,
            use_dict_obs: bool = True,
            **kwargs
    ):
        if object_type != ObjectType.DodecahedronBasket:
            raise ValueError("This environment requires DodecahedronBulb object type")

        arena_center = np.array([0.72, 0.15, 0.76])
        repos_spot = np.array([0.5, 0.15, 0.76])
        phase_envs_params = [
            # Basket
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "BasketFixed-v0",
                "env_kwargs": {
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
            # Reset controller = reposition + pickup
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "PickupFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BASKET_RESETFREE_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [repos_spot],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([repos_spot[0], repos_spot[1], 1.2])],
                    },
                    "target_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
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
            **kwargs
        )

    def task_graph(self):
        Phase = self.Phase

        # Drop the object if we were just in the midair reposition phase
        if self.phase == Phase.BASKET.value:
            self._release_object()

        # Construct a new adjacency matrix
        self.task_adj_matrix = np.zeros((self.num_phases, self.num_phases))

        rc_env = self._envs[Phase.PICKUP.value]
        rc_obs = rc_env.get_obs_dict()
        rc_rew = rc_env.get_reward_dict(None, rc_obs)

        object_xyz = rc_obs["object_xyz"]
        object_to_target_xy_dist = rc_obs["object_to_target_xy_distance"]

        # above height, within xy
        if (object_xyz[2] <= self.Z_THRESHOLD
                or object_to_target_xy_dist >= self.DISTANCE_THRESHOLD):
            self.task_adj_matrix[Phase.PICKUP.value][Phase.PICKUP.value] = 1.0
            self.task_adj_matrix[Phase.BASKET.value][Phase.PICKUP.value] = 1.0
        else:
            self.task_adj_matrix[Phase.PICKUP.value][Phase.BASKET.value] = 1.0
            self.task_adj_matrix[Phase.BASKET.value][Phase.BASKET.value] = 1.0

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


class SawyerDhandInHandDodecahedronBasketPhased(SawyerDhandInHandManipulateObjectMultiPhase):
    """
    5 phases = Reposition + Pickup + Perturb + Move to Basket + Drop
    """
    DISTANCE_THRESHOLD = 0.1
    Z_DISTANCE_THRESHOLD = 1.1
    OBJ_TO_BASKET_DISTANCE_THRESHOLD = 0.1

    class Phase(Enum):
        REPOSITION = 0
        PICKUP = 1
        PERTURB = 2
        REPOSITION_MIDAIR = 3

    def __init__(
            self,
            x_range: float = 0.15,
            y_range: float = 0.15,
            object_type: ObjectType = ObjectType.DodecahedronBasket,
            n_bins: int = 100,
            phase_to_color_idx: dict = None,
            use_dict_obs: bool = True,
            **kwargs
    ):
        if object_type != ObjectType.DodecahedronBasket:
            raise ValueError("This environment requires DodecahedronBasket object type")

        Phase = self.Phase
        if phase_to_color_idx is None:
            self.phase_to_color_idx = {
                Phase.REPOSITION.value: np.array([1, 0, 0, 1]),
                Phase.PICKUP.value: np.array([0, 0, 1, 1]),
                Phase.PERTURB.value: np.array([0, 0, 0, 1]),
                Phase.REPOSITION_MIDAIR.value: np.array([0, 1, 1, 1]),
            }
        else:
            self.phase_to_color_idx = phase_to_color_idx

        arena_center = np.array([0.72, 0.15, 0.76])
        repos_spot = np.array([0.5, 0.15, 0.76])
        phase_envs_params = [
            # Reposition
            {
                "domain": f"SawyerDhandInHandDodecahedron",
                "task": "RepositionFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BASKET_REPOSITION_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "UniformRange",
                        "values": [
                            np.array([arena_center[0] - x_range, arena_center[1] - y_range, arena_center[2]]),
                            np.array([arena_center[0] + x_range, arena_center[1] + y_range, arena_center[2]]),
                        ],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [repos_spot],
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
                    "sawyer_config": BASKET_PICKUP_SAWYER_ROBOT_CONFIG,
                    "init_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [repos_spot],
                    },
                    "init_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
                    },
                    "target_xyz_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([repos_spot[0], repos_spot[1], 1.2])],
                    },
                    "target_euler_range_params": {
                        "type": "DiscreteRange",
                        "values": [np.array([0, 0, np.pi])],
                    },
                    'object_type': object_type,
                    'use_dict_obs': use_dict_obs
                },
            },
            # XY Perturbation
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "RepositionFixed-v0",
                "env_kwargs": {
                    "sawyer_config": BASKET_REPOSITION_SAWYER_ROBOT_CONFIG,
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
            # Reposition Midair
            {
                "domain": "SawyerDhandInHandDodecahedron",
                "task": "BasketFixed-v0",
                "env_kwargs": {
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
        ]

        super().__init__(
            phase_envs_params=phase_envs_params,
            object_type=object_type,
            use_dict_obs=use_dict_obs,
            **kwargs
        )

    def task_graph(self):
        Phase = self.Phase

        # Drop the object if we were just in the midair reposition phase
        if self.phase == Phase.REPOSITION_MIDAIR.value:
            self._release_object()

        # Construct a new adjacency matrix
        self.task_adj_matrix = np.zeros((self.num_phases, self.num_phases))

        # Collect environment obs and reward dicts to determine transitions
        repos_env = self._envs[Phase.REPOSITION.value]
        repos_obs = repos_env.get_obs_dict()

        repos_xy_dist = repos_obs["object_to_target_xy_distance"]
        object_xyz = repos_obs["object_xyz"]
        basket_env = self._envs[Phase.REPOSITION_MIDAIR.value]
        # obj_to_basket_dist = basket_env.get_obs_dict()["object_to_target_xyz_distance"]

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
            self.task_adj_matrix[Phase.REPOSITION_MIDAIR.value][Phase.REPOSITION.value] = 1.0

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
            self.task_adj_matrix[Phase.REPOSITION_MIDAIR.value][Phase.PICKUP.value] = 1.0
        elif object_xyz[2] > self.Z_DISTANCE_THRESHOLD:
            self.task_adj_matrix[Phase.PERTURB.value][Phase.REPOSITION_MIDAIR.value] = 1.0
            self.task_adj_matrix[Phase.REPOSITION.value][Phase.REPOSITION_MIDAIR.value] = 1.0
            self.task_adj_matrix[Phase.PICKUP.value][Phase.REPOSITION_MIDAIR.value] = 1.0
            self.task_adj_matrix[Phase.REPOSITION_MIDAIR.value][Phase.REPOSITION_MIDAIR.value] = 1.0
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
            # Removed the probability from the choice list
            next_phase = self.np_random.choice(self.num_phases)
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
