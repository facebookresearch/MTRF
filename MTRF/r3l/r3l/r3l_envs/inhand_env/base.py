# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
from enum import Enum
from gym import utils
import numpy as np
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple

from r3l.r3l_envs.base_env.base_env_deltapos import SawyerDhandDeltaBaseEnv
from r3l.utils.quatmath import quat2euler, euler2quat
from r3l.utils.circle_math import circle_distance, circle_distance_mod
from r3l.utils.range import get_range_from_params
from r3l.robot.object import ObjectState
from r3l.robot.default_configs import (
    DEFAULT_DHAND_ROBOT_CONFIG,
    DEFAULT_SAWYER_ROBOT_CONFIG,
    DEFAULT_OBJECT_CONFIG,
)

class ObjectType(Enum):
    Valve3 = 1
    Rod = 2
    Dodecahedron = 3
    DodecahedronBasket = 4
    DodecahedronBulb = 5
    Dumbbell = 6
    Mug = 7
    Pipe = 8

DHAND_INHAND_ASSET_PATH = Path(__file__).absolute().parent / "assets"
OBJECT_TYPE_TO_MODEL_PATH = {
    ObjectType.Valve3: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_valve.xml"),
    ObjectType.Rod: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_rod.xml"),
    ObjectType.Dodecahedron: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_dodecahedron.xml"),
    ObjectType.DodecahedronBasket: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_dodecahedron_basket.xml"),
    ObjectType.DodecahedronBulb: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_dodecahedron_bulb.xml"),
    ObjectType.Dumbbell: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_dumbbell.xml"),
    ObjectType.Mug: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_mug.xml"),
    ObjectType.Pipe: str(DHAND_INHAND_ASSET_PATH / "sawyer_dhand_inhand_pipe.xml"),
}


class SawyerDhandInHandObjectBaseEnv(SawyerDhandDeltaBaseEnv):
    VALVE_DEFAULT_OBSERVATION_KEYS = (
        "dhand_qpos", # "sawyer_arm_qpos",
        "dhand_qvel", # "sawyer_arm_qvel",
        "mocap_pos", "mocap_euler",

        # Object info
        "object_xyz", # Not including object euler and qvel
        # object "z orientation" accounts for flips
        "object_top_angle_cos",
        "object_top_angle_sin",
        "target_xyz",
        # target "z orientation" accounting for flips
        "target_top_angle_cos",
        "target_top_angle_sin",

        "object_to_hand_xyz",
        # These all account for site symmetry
        "object_site1_to_target_site1_xyz_err",
        "object_site2_to_target_site2_xyz_err",
        "object_site3_to_target_site3_xyz_err"
    )

    ROD_DEFAULT_OBSERVATION_KEYS = (
        "dhand_qpos", "sawyer_arm_qpos",
        "mocap_pos", "mocap_euler",

        # Object info
        "object_xyz", # Not including object euler and qvel
        "object_z_orientation_cos", "object_z_orientation_sin",

        "target_xyz",
        "target_z_orientation_cos", "target_z_orientation_sin",

        "object_to_hand_xyz",
    )

    PIPE_DEFAULT_OBSERVATION_KEYS = (
        "dhand_qpos", "sawyer_arm_qpos",
        "mocap_pos", "mocap_euler",

        # Object info
        "object_xyz", # Not including object euler and qvel
        # "object_z_orientation_cos", "object_z_orientation_sin",

        "target_xyz",
        # "target_z_orientation_cos", "target_z_orientation_sin",

        "object_to_hand_xyz",
        "object_normal_to_target_normal_err",
        "object_parallel_to_target_parallel_err",
    )

    DODECAHEDRON_DEFAULT_OBSERVATION_KEYS = (
        "dhand_qpos", "sawyer_arm_qpos",
        "mocap_pos", "mocap_euler",

        # Object info
        "object_xyz", # Not including object euler and qvel
        "target_xyz",

        "object_to_hand_xyz",
        "object_to_target_xyz",
    )

    def __init__(
            self,
            model_path=None,
            sim=None,
            object_type: ObjectType = ObjectType.Rod,
            observation_keys: list = ROD_DEFAULT_OBSERVATION_KEYS,
            collect_bin_counts: bool = False,
            n_bins: int = 32,
            reset_policy_directories: list = [],
            reset_state_pkl_path: str = None,
            reset_offset: np.ndarray = None,
            save_reset_videos: bool = False,
            **kwargs
    ):
        self.object_type = object_type

        # Discrete count rewards
        self.x_range = (0.72 - 0.4, 0.72 + 0.4)
        self.y_range = (0.15 - 0.4, 0.15 + 0.4)
        self._collect_bin_counts = collect_bin_counts
        self.n_bins = n_bins
        self.x_bins = np.linspace(self.x_range[0], self.x_range[1], self.n_bins)
        self.y_bins = np.linspace(self.y_range[0], self.y_range[1], self.n_bins)
        self.bin_counts = np.ones((self.n_bins + 1, self.n_bins + 1))

        # Loaded policy resets
        self._reset_dist = np.zeros(1)
        self._reset_policies = []
        self._wrapped_reset_policies = []
        self._reset_horizons = []
        self._reset_envs = []
        self._reset_imgs = []
        self._save_reset_videos = save_reset_videos

        # Track goal position
        self._last_target_xyz = np.zeros(3)
        self._last_target_euler = np.zeros(3)
        self._pose_target = 0.7 * np.array([
            1, 1, 1, 1,
            -1, -1, -1, -1,
            1, 1, 1, 1,
            -1, -1, -1, -1,
        ])

        # Initialize observation keys depending on object type
        if self.object_type in (
                ObjectType.Dodecahedron,
                ObjectType.DodecahedronBasket,
                ObjectType.DodecahedronBulb):
            observation_keys = self.DODECAHEDRON_DEFAULT_OBSERVATION_KEYS
        elif self.object_type in (ObjectType.Valve3,):
            observation_keys = self.VALVE_DEFAULT_OBSERVATION_KEYS
        elif self.object_type in (ObjectType.Rod,):
            observation_keys = self.ROD_DEFAULT_OBSERVATION_KEYS
        elif self.object_type in (ObjectType.Pipe,):
            observation_keys = self.PIPE_DEFAULT_OBSERVATION_KEYS

        try:
            if self.script_motion:
                observation_keys = observation_keys + ("step_counter",)
        except AttributeError:
            pass

        # Init the environment + sim
        env_params = dict(
            observation_keys=observation_keys,
            dhand_config=DEFAULT_DHAND_ROBOT_CONFIG,
            sawyer_config=DEFAULT_SAWYER_ROBOT_CONFIG,
            object_config=DEFAULT_OBJECT_CONFIG,
        )
        env_params.update(kwargs)
        if sim:
            super().__init__(sim=sim, **env_params)
        else:
            assert object_type in OBJECT_TYPE_TO_MODEL_PATH
            object_model_path = OBJECT_TYPE_TO_MODEL_PATH.get(object_type)
            # If a model_path is already specified, make sure that it matches the object type.
            if model_path is not None:
                assert model_path == object_model_path
            else:
                model_path = object_model_path
            super().__init__(
                model_path=model_path,
                observation_keys=observation_keys,
                **kwargs
            )

            # Boost damping in the arms to avoid high accelerations
            # TODO(justinvyu): Is this also the case on the hardware?
            self.sim.model.dof_damping[:7] *= 10
            self.sim.model.dof_armature[:7] *= 10

        # Get model ids
        self.target_bid = self.sim.model.body_name2id('target')
        self.grasp_left = self.sim.model.site_name2id('grasp_left')
        self.grasp_right = self.sim.model.site_name2id('grasp_right')
        self.grasp_id = self.sim.model.site_name2id('grasp')
        if self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            self.rod_center = self.sim.model.site_name2id('rod_center')
            self.rod_left = self.sim.model.site_name2id('rod_left')
            self.rod_right = self.sim.model.site_name2id('rod_right')
            self.rod_normal = self.sim.model.site_name2id('rod_normal')
            self.rod_parallel = self.sim.model.site_name2id('rod_parallel')

            self.target_center = self.sim.model.site_name2id('target_center')
            self.target_left = self.sim.model.site_name2id('target_left')
            self.target_right = self.sim.model.site_name2id('target_right')
            self.target_normal = self.sim.model.site_name2id('target_normal')
            self.target_parallel = self.sim.model.site_name2id('target_parallel')
        elif self.object_type == ObjectType.Valve3:
            self.valve_1 = self.sim.model.site_name2id('valve3_1')
            self.valve_2 = self.sim.model.site_name2id('valve3_2')
            self.valve_3 = self.sim.model.site_name2id('valve3_3')
            self.target_1 = self.sim.model.site_name2id('valve3_target_1')
            self.target_2 = self.sim.model.site_name2id('valve3_target_2')
            self.target_3 = self.sim.model.site_name2id('valve3_target_3')

        # Track goal position
        self._last_target_xyz = self.sim.data.get_body_xpos("target").copy()
        self._last_target_euler = quat2euler(self.sim.data.get_body_xquat("target").copy())

        # Object offset = the offset relative to the object that the hand is reset to
        self.obj_offset = reset_offset
        if self.obj_offset is None:
            self.obj_offset = np.array([-0.15, 0, 0.175])

        if reset_state_pkl_path:
            self._use_reset_state = True
            with open(reset_state_pkl_path, 'rb') as f:
                reset_states = pickle.load(f)
                if isinstance(reset_states, dict):
                    assert "sawyer" in reset_states
                    assert "dhand" in reset_states
                    assert "object" in reset_states
                    self._reset_sawyer_state = reset_states["sawyer"]
                    self._reset_dhand_state = reset_states["dhand"]
                    self._reset_object_state = reset_states["object"]
                else:
                    self._reset_sawyer_state = reset_states[0]
                    self._reset_dhand_state = reset_states[1]
                    self._reset_object_state = reset_states[2]
        else:
            self._use_reset_state = False
            reset_policy_directories = reset_policy_directories or []
            for directory in reset_policy_directories:
                self._load_policy(directory)

    def _load_policy(self, saved_policy_dir, phase_num=None):
        try:
            import softlearning
        except ModuleNotFoundError:
            print('Error: softlearning package not found. Unable to load reset policy.')
            return None
        from softlearning.policies.utils import get_policy_from_variant
        from softlearning.environments.utils import get_environment_from_params

        policy_dir = Path(saved_policy_dir)
        variant_path = policy_dir / "params.pkl"
        policy_weights_path = policy_dir / "policy_params.pkl"
        assert variant_path.exists() and policy_weights_path.exists(), (
            "Error loading policy and variant: we expect a file at:\n"
            + str(variant_path) + "\n" + str(policy_weights_path)
        )
        with open(variant_path, 'rb') as f:
            variant = pickle.load(f)
        with open(policy_weights_path, 'rb') as f:
            policy_weights = pickle.load(f)

        if phase_num is not None:
            policy_weights = policy_weights[phase_num]

        from softlearning.environments.adapters.gym_adapter import GymAdapter
        from softlearning.models.utils import flatten_input_structure
        reset_env_params = variant["environment_params"]["evaluation"]
        reset_env_kwargs = reset_env_params.pop("kwargs", {})
        reset_env_kwargs["sim"] = self.sim
        reset_env_params["kwargs"] = reset_env_kwargs
        reset_env = get_environment_from_params(reset_env_params)
        # Create environment as softlearning expects it for policy initialization
        reset_env = GymAdapter(None, None, env=reset_env)
        self._reset_envs.append(reset_env)

        reset_policy = get_policy_from_variant(variant, reset_env)
        reset_policy.set_weights(policy_weights)
        self._reset_policies.append(reset_policy)

        # Save some time by taking a max reset horizon of 50 steps
        horizon = min(
            50,
            variant.get('sampler_params', {}).get('kwargs', {}).get('max_path_length', 50)
        )
        self._reset_horizons.append(horizon)

        def wrapped_policy(obs_dict):
            feed_dict = {
                key: obs_dict[key][None, ...]
                for key in reset_policy.observation_keys
            }
            observation = flatten_input_structure(feed_dict)
            with reset_policy.set_deterministic(True):
                action = reset_policy.actions_np(observation)[0]
            return action
        self._wrapped_reset_policies.append(wrapped_policy)
        return wrapped_policy

    def run_reset_policy(self):
        if self._use_reset_state:
            # Reset angles so that mocap doesn't go crazy after many iters
            self.sawyer_robot.reset(command_angles=True)
            self.sawyer_robot.set_state(self._reset_sawyer_state, command_angles=False)
            self.do_simulation(self.act_mid - 0.75 * self.act_rng, 300) # wait for arm to be stable
            self.dhand_robot.set_state(self._reset_dhand_state)
            self.object.set_state(self._reset_object_state)
        else:
            self._reset_imgs = []
            for reset_policy, reset_env, horizon in zip(self._wrapped_reset_policies, self._reset_envs, self._reset_horizons):
                reset_env.reset_robot()
                for _ in range(horizon):
                    action = reset_policy(reset_env.get_obs_dict())
                    reset_env.step(action)
                    if self._save_reset_videos:
                        self._reset_imgs.append(self.render(width=480, height=480, mode="rgb_array"))
                self._reset_dist = self.get_obs_dict()["object_to_target_xy_distance"]

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_obs_dict()
        # Object data
        object_qpos, object_qvel = obs_dict["object_qpos"], obs_dict["object_qvel"]
        object_xyz, object_quat = object_qpos[:3], object_qpos[3:]
        object_euler = quat2euler(object_quat)
        object_z_orientation = object_euler[2]

        # Target data
        # NOTE: Targets need to be set in the OptiTrack coordinate space for hardware
        target_xyz = self._last_target_xyz.copy()
        target_euler = self._last_target_euler.copy()
        target_z_orientation = target_euler[2]

        # TODO(justinvyu): Add an offset to the end of the Sawyer xyz
        grasp_xyz = self.sim.data.get_site_xpos("grasp").copy()
        if self.is_hardware:
            grasp_xyz = obs_dict["mocap_pos"][:3]

        relative_xyz = object_xyz - target_xyz
        circle_dist = circle_distance(object_euler, target_euler)
        hand_to_obj_dist = np.linalg.norm(grasp_xyz - object_xyz)
        hand_to_target_dist = np.linalg.norm(grasp_xyz - target_xyz)

        x, y, z = object_xyz
        x_d, y_d = np.digitize(x, self.x_bins), np.digitize(y, self.y_bins)

        hand_pose_dist = np.linalg.norm(obs_dict["dhand_qpos"] - self._pose_target)

        circle_dist_mod_180 = np.array([
            circle_distance_mod(object_z_orientation, target_z_orientation, mod=np.pi)
        ])

        obs_dict.update(collections.OrderedDict((
            # Object info
            ("object_xyz", object_xyz),
            ("object_xy_discrete", np.array([x_d, y_d])),
            ("object_quat", object_quat),
            ("object_euler", object_euler),
            ("object_z_orientation", object_z_orientation),
            ("object_z_orientation_cos", np.array([np.cos(object_z_orientation)])),
            ("object_z_orientation_sin", np.array([np.sin(object_z_orientation)])),
            ("object_qvel", object_qvel),

            # Target info
            ("target_xyz", target_xyz),
            ("target_euler", target_euler),
            ("target_quat", euler2quat(target_euler)),
            ("target_z_orientation", target_z_orientation),
            ("target_z_orientation_cos", np.array([np.cos(target_z_orientation)])),
            ("target_z_orientation_sin", np.array([np.sin(target_z_orientation)])),

            # Distances
            ("object_to_target_xyz_distance", np.linalg.norm(relative_xyz)),
            ("object_to_target_xy_distance", np.linalg.norm(relative_xyz[:2])),
            ("object_to_target_x_distance", np.abs(relative_xyz[0])),
            ("object_to_target_y_distance", np.abs(relative_xyz[1])),
            ("object_to_target_z_distance", np.abs(relative_xyz[2])),
            ("object_to_target_circle_distances", circle_dist),
            ("object_to_target_circle_distance", np.linalg.norm(circle_dist)),
            ("object_to_target_mod_120_circle_distance", np.array([0])),
            ("object_to_target_mod_180_circle_distance", circle_dist_mod_180),
            ("object_to_hand_xyz_distance", hand_to_obj_dist),
            ("target_to_hand_xyz_distance", hand_to_target_dist),
            ("pose_dist", hand_pose_dist),

            # Relative vectors
            ("object_to_target_xyz", relative_xyz),
            ("object_to_hand_xyz", grasp_xyz - object_xyz),
            ("grasp_xyz", grasp_xyz),
        )))
        if self._wrapped_reset_policies:
            obs_dict["reset_policy_xy_distance"] = self._reset_dist.copy()

        # Add these as placeholders (to be populated for each object specifically)

        # Valve specific observations
        obs_dict["object_site1_to_target_site1_xyz_err"] = np.zeros(3)
        obs_dict["object_site2_to_target_site2_xyz_err"] = np.zeros(3)
        obs_dict["object_site3_to_target_site3_xyz_err"] = np.zeros(3)
        obs_dict["object_top_angle_cos"] = np.array([0])
        obs_dict["object_top_angle_sin"] = np.array([0])
        obs_dict["object_top_angle"] = np.array([0])
        obs_dict["target_top_angle_cos"] = np.array([0])
        obs_dict["target_top_angle_sin"] = np.array([0])
        obs_dict["target_top_angle"] = np.array([0])

        # Rod/pipe specific observations
        obs_dict["object_normal_to_target_normal_err"] = np.zeros(3)
        obs_dict["object_parallel_to_target_parallel_err"] = np.zeros(3)
        obs_dict["object_normal_to_target_normal_distance"] = np.zeros(1)
        obs_dict["object_parallel_to_target_parallel_distance"] = np.zeros(1)

        obs_dict['step_counter'] = np.array([0])

        if not self.initializing and self.object_type == ObjectType.Valve3:
            target1_xpos = self.sim.data.site_xpos[self.target_1].copy()
            valve1_xpos = self.sim.data.site_xpos[self.valve_1].copy()

            valve_dir = np.array(valve1_xpos - object_xyz)
            # Project onto xy plane
            valve_dir[2] = 0
            # Unit vector in direction of the red prong
            valve_dir /= np.linalg.norm(valve_dir)

            target_dir = np.array(target1_xpos - target_xyz)
            # Project onto xy plane
            target_dir[2] = 0
            # Unit vector in direction of the red prong
            target_dir /= np.linalg.norm(target_dir)

            object_angle = np.arctan2(valve_dir[1], valve_dir[0])
            obs_dict["object_top_angle"] = np.array([object_angle])
            obs_dict["object_top_angle_cos"] = np.array([valve_dir[0]])
            obs_dict["object_top_angle_sin"] = np.array([valve_dir[1]])
            target_angle = np.arctan2(target_dir[1], target_dir[0])
            obs_dict["target_top_angle"] = np.array([target_angle])
            obs_dict["target_top_angle_cos"] = np.array([target_dir[0]])
            obs_dict["target_top_angle_sin"] = np.array([target_dir[1]])

            obs_dict["object_to_target_mod_120_circle_distance"] = np.array([
                circle_distance_mod(object_angle, target_angle, mod=(2 * np.pi / 3))
            ])

            # Calculating site distances (of closest matching sites)
            perm_1_dist = np.mean([
                self.get_site_distance(self.target_1, self.valve_1),
                self.get_site_distance(self.target_2, self.valve_2),
                self.get_site_distance(self.target_3, self.valve_3)
            ])
            perm_2_dist = np.mean([
                self.get_site_distance(self.target_1, self.valve_2),
                self.get_site_distance(self.target_2, self.valve_3),
                self.get_site_distance(self.target_3, self.valve_1)
            ])
            perm_3_dist = np.mean([
                self.get_site_distance(self.target_1, self.valve_3),
                self.get_site_distance(self.target_2, self.valve_1),
                self.get_site_distance(self.target_3, self.valve_2)
            ])

            if perm_1_dist <= perm_2_dist and perm_1_dist <= perm_3_dist:
                # Use the first permutation because it is the closest
                valve_1 = self.valve_1
                valve_2 = self.valve_2
                valve_3 = self.valve_3
                object_to_target_dist = perm_1_dist
            elif perm_2_dist <= perm_1_dist and perm_2_dist <= perm_3_dist:
                # Use the 2nd
                valve_1 = self.valve_2
                valve_2 = self.valve_3
                valve_3 = self.valve_1
                object_to_target_dist = perm_2_dist
            elif perm_3_dist <= perm_1_dist and perm_3_dist <= perm_2_dist:
                # Use the 3rd
                valve_1 = self.valve_3
                valve_2 = self.valve_1
                valve_3 = self.valve_2
                object_to_target_dist = perm_3_dist

            obs_dict["object_sites_to_target_sites_distance"] = object_to_target_dist
            obs_dict["object_site1_to_target_site1_xyz_err"] = self.sim.data.site_xpos[self.target_1] - self.sim.data.site_xpos[valve_1]
            obs_dict["object_site2_to_target_site2_xyz_err"] = self.sim.data.site_xpos[self.target_2] - self.sim.data.site_xpos[valve_2]
            obs_dict["object_site3_to_target_site3_xyz_err"] = self.sim.data.site_xpos[self.target_3] - self.sim.data.site_xpos[valve_3]

        if not self.initializing and self.object_type in (ObjectType.Rod, ObjectType.Pipe):
            object_center = self.sim.data.site_xpos[self.rod_center]
            target_center = self.sim.data.site_xpos[self.target_center]
            object_normal = self.sim.data.site_xpos[self.rod_normal]
            object_parallel = self.sim.data.site_xpos[self.rod_parallel]
            target_normal = self.sim.data.site_xpos[self.target_normal]
            target_parallel = self.sim.data.site_xpos[self.target_parallel]

            object_normal_vec = obs_dict["object_normal_vector"] = object_normal - object_center
            target_normal_vec = obs_dict["target_normal_vector"] = target_normal - target_center
            object_parallel_vec = obs_dict["object_parallel_vector"] = object_parallel - object_center
            target_parallel_vec = obs_dict["target_parallel_vector"] = target_parallel - target_center

            normal_err = obs_dict["object_normal_to_target_normal_err"] = object_normal_vec - target_normal_vec
            parallel_err = obs_dict["object_parallel_to_target_parallel_err"] = object_parallel_vec - target_parallel_vec
            obs_dict["object_normal_to_target_normal_distance"] = np.linalg.norm(normal_err)
            obs_dict["object_parallel_to_target_parallel_distance"] = np.linalg.norm(parallel_err)

        return obs_dict

    def get_site_distance(self, site1, site2):
        return np.linalg.norm(self.sim.data.site_xpos[site1] - self.sim.data.site_xpos[site2])

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if self._collect_bin_counts:
            x_d, y_d = self.last_obs_dict["object_xy_discrete"]
            self.bin_counts[x_d, y_d] += 1
        return obs, rew, done, info

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        xy_d = obs_dict["object_xy_discrete"].squeeze()
        x_d, y_d = xy_d[0], xy_d[1]
        circle_dist = obs_dict["object_to_target_circle_distance"]
        circle_dist_mod_180 = obs_dict["object_to_target_mod_180_circle_distance"]
        xy_dist = obs_dict["object_to_target_xy_distance"]
        x_dist = obs_dict["object_to_target_x_distance"]
        y_dist = obs_dict["object_to_target_y_distance"]
        z_dist = obs_dict["object_to_target_z_distance"]
        xyz_dist = obs_dict["object_to_target_xyz_distance"]
        hand_to_obj_distance = obs_dict["object_to_hand_xyz_distance"]
        hand_to_target_distance = obs_dict["target_to_hand_xyz_distance"]

        reward_dict = collections.OrderedDict((
            # Negative average distance btwn ends of object and hand
            ('object_to_hand_xyz_distance_reward', -hand_to_obj_distance),
            ('target_to_hand_xyz_distance_reward', -hand_to_target_distance),
            # Negative distance between center of object to desired location
            ("object_to_target_xy_distance_reward", -xy_dist),
            ("object_to_target_xyz_distance_reward", -xyz_dist),
            ("object_to_target_x_distance_reward", -x_dist),
            ("object_to_target_y_distance_reward", -y_dist),
            ("object_to_target_z_distance_reward", -z_dist),

            ("object_to_target_circle_distance_reward", -circle_dist),
            ("object_to_target_mod_120_circle_distance_reward", -obs_dict["object_to_target_mod_120_circle_distance"]),
            ("object_to_target_mod_180_circle_distance_reward", -circle_dist_mod_180),
            ("object_to_target_circle_distance_normalized_reward", -circle_dist / np.pi),
            # Measuring spread of the Dhand fingers compared to desired grip
            ('span_dist', np.array([0.])),
            ('hand_pose_dist_reward', -obs_dict["pose_dist"]),

            ('xy_discrete_sqrt_count_reward', 1 / np.sqrt(self.bin_counts[x_d, y_d])),
            # ('xy_discrete_invlog_count_reward', 1 / np.log(1 + self.bin_counts[x_d, y_d])),

            ("object_normal_to_target_normal_distance_reward", -obs_dict["object_normal_to_target_normal_distance"]),
            ("object_parallel_to_target_parallel_distance_reward", -obs_dict["object_parallel_to_target_parallel_distance"]),
        ))

        if not self.initializing:
            reward_dict["span_dist"] = -.03 * np.linalg.norm(
                self.sim.data.qpos[7:23] - self.act_mid - 0.5 * self.act_rng)

        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return collections.OrderedDict()

    def _discretize_xys(self, xys):
        assert isinstance(xys, np.ndarray) and xys.ndim == 2 and xys.shape[1] == 2
        x_d = np.expand_dims(np.digitize(xys[:, 0], self.x_bins), 1)
        y_d = np.expand_dims(np.digitize(xys[:, 1], self.y_bins), 1)
        return np.concatenate((x_d, y_d), axis=1)

    def get_count_bonuses(self, xys):
        """
        Parameters
        ----------
        xys : np.ndarray (batch_size x 2)
            Must have batch_size as its first dimension, even if it's only 1
        """
        xy_d = self._discretize_xys(xys)
        count_bonuses = 1 / np.sqrt(self.bin_counts[xy_d[:, 0], xy_d[:, 1]])
        return count_bonuses.reshape((-1, 1))

    def reset_target(self):
        pass

    def set_goal(self, goal_index):
        """Required to implement this method for softlearning MultiSAC."""
        pass
