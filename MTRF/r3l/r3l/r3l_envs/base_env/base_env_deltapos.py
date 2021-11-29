# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import abc
import collections
import gym
from gym import spaces, utils
import mujoco_py
import numpy as np
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple
from r3l.r3l_envs.base_env.mujoco_env import MujocoEnv
from r3l.robot.dhand_robot import DhandRobot, DhandRobotState
from r3l.robot.sawyer_robot import SawyerRobot, SawyerRobotState
from r3l.robot.object import Object
from r3l.utils.quatmath import euler2quat, quat2euler
from r3l.utils.misc import get_git_rev
from r3l.robot.default_configs import (
    DEFAULT_DHAND_ROBOT_CONFIG,
    DEFAULT_SAWYER_ROBOT_CONFIG,
    DEFAULT_OBJECT_CONFIG,
)

# Define helper types for observation, action and state.
Observation = NewType('Observation', np.ndarray)
Action = NewType('Action', np.ndarray)


def make_box_space(low: Union[float, Sequence[float]],
                   high: Union[float, Sequence[float]],
                   shape: Optional[int] = None) -> gym.spaces.Box:
    """Returns a Box gym space."""
    try:
        return spaces.Box(low, high, shape, dtype=np.float32)
    except TypeError:
        return spaces.Box(low, high, shape)


class SawyerDhandDeltaBaseEnv(MujocoEnv, utils.EzPickle, metaclass=abc.ABCMeta):
    def __init__(
        self,
        model_path=None,
        sim=None,
        task_name="",
        frame_skip=40,
        is_hardware=False,
        reset_every_n_episodes: int = 1,
        reset_robot: bool = True,
        readjust_to_object_in_reset: bool = True,
        readjust_hand_xyz: bool = True,
        readjust_hand_euler: bool = False,
        use_dict_obs: bool = True,
        observation_keys: Optional[Sequence[str]] = None,
        reward_keys_and_weights: dict = None,
        dhand_config: dict = DEFAULT_DHAND_ROBOT_CONFIG,
        sawyer_config: dict = DEFAULT_SAWYER_ROBOT_CONFIG,
        object_config: dict = DEFAULT_OBJECT_CONFIG,
        camera_configs: dict = None,
        render_cameras: Sequence[int] = [-1],
        verbose: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        reset_every_n_episodes : int or np.inf
            [-np.inf, 0] + {np.inf}     Reset free
            {1}                         Full resets every time
            [2, np.inf)                 Reset intermittently every nth reset
        reset_robot : bool
        """
        if not hasattr(self, "_ezpickle_kwargs"):
            utils.EzPickle.__init__(self)

        # self.git_sha = get_git_rev(__file__)
        self.initializing = True

        self.task_name = task_name
        self._verbose = verbose
        self._reset_every_n_episodes = reset_every_n_episodes
        self._reset_count = 0
        self._reset_robot = reset_robot
        self._readjust_to_object_in_reset = readjust_to_object_in_reset
        self._readjust_hand_xyz = readjust_hand_xyz
        self._readjust_hand_euler = readjust_hand_euler

        self._observation_keys = observation_keys
        self._reward_keys_and_weights = reward_keys_and_weights
        self._use_dict_obs = use_dict_obs

        self.obj_offset = np.array([0, 0, 0.175])

        self._onscreen_renderer = None
        self._offscreen_renderers = {}

        # Create and configure env
        if sim:
            super().__init__(sim=sim, frame_skip=frame_skip)
        else:
            super().__init__(model_path, frame_skip=frame_skip)
        self.configure()

        self._is_hardware = is_hardware
        self.dhand_robot = DhandRobot(env=self, config_params=dhand_config)
        self.sawyer_robot = SawyerRobot(
            env=self, is_hardware=is_hardware, config_params=sawyer_config)
        self.object = Object(env=self,config_params=object_config)

        assert (
            self.dhand_robot.is_hardware
            == self.sawyer_robot.is_hardware
            == self.object.is_hardware
        ), "ERROR: Configuration parameters for robots not aligned"

        self._image_service = None
        # if camera_configs is not None:
        #     from r3l.utils.camera import get_image_service
        #     self._image_service = get_image_service(**camera_configs)

        self._camera_configs = camera_configs
        if camera_configs is None:
            self._camera_configs = {
                "azimuth": -90,
                "distance": 1.5,
                "elevation": -30,
                "lookat": np.array([0.72, 0.15, 0.76]),
            }

        self._render_cameras = render_cameras

        # Initialize observation space
        self.observation_space = self._initialize_observation_space()

        # Reset action space to -1 -> +1 for all action dimensions.
        self.action_space = self._initialize_action_space()

        # Ensure gym does not try to patch `_step` and `_reset`.
        self._gym_disable_underscore_compat = True

        self.initializing = False
        self.last_obs_dict = None

    @property
    def is_hardware(self):
        return self._is_hardware

    def configure(self):
        # update init pos
        # TODO: Move this elsewhere
        self.init_qpos[:7] = np.array([0.7, 0.133, -0.503, 1.067, -2.308, 0.976, 0.0973])

        # Dhand Motor Control Ranges
        self.act_mid = 0.5 * (
            self.model.actuator_ctrlrange[:, 0]
            + self.model.actuator_ctrlrange[:, 1]
        )
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1]
            - self.model.actuator_ctrlrange[:,0]
        )

        self.sim.reset()
        self.set_state(self.init_qpos, self.init_qvel)

        self.sim.forward()

    def _initialize_observation_space(self) -> gym.Space:
        """Returns the observation space to use for this environment.

        The default implementation calls `_get_obs()` and returns a dictionary
        space if the observation is a mapping, or a box space otherwise.
        """
        observation = self._get_obs()
        if isinstance(observation, dict):
            assert self._use_dict_obs
            return spaces.Dict({
                key: make_box_space(-np.inf, np.inf, shape=np.shape(value))
                for key, value in observation.items()
            })
        return make_box_space(-np.inf, np.inf, shape=observation.shape)

    def _initialize_action_space(self) -> gym.Space:
        return spaces.Box(low=-np.ones_like((self.action_space.low)),
                          high=np.ones_like((self.action_space.high)))

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """Runs one timestep of the environment with the given action.

        Subclasses must override 4 subcomponents of step:
        - `_step`: Applies an action to the robot
        - `get_obs_dict`: Returns the current observation of the robot.
        - `get_reward_dict`: Calculates the reward for the step.
        - `get_done`: Returns whether the episode should terminate.

        Parameters
        ----------
        action : Action
            An action to control the environment.

        Returns
        -------
        observation : Observation
            The observation of the environment after the timestep.
        reward : float
            The amount of reward obtained during the timestep.
        done : bool
            Whether the episode has ended. `env.reset()` should be called
            if this is True.
        info : Dict
            Auxiliary information about the timestep.
        """
        # Perform the step.
        self._step(action)

        # Get the observation after the step.
        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        flattened_obs = self._get_obs(obs_dict)

        # Get the rewards for the observation.
        batched_action = np.expand_dims(np.atleast_1d(action), axis=0)
        batched_obs_dict = {
            k: np.expand_dims(np.atleast_1d(v), axis=0)
            for k, v in obs_dict.items()
        }
        batched_reward_dict = self.get_reward_dict(batched_action,
                                                   batched_obs_dict)

        # Calculate the total reward.
        reward_dict = {k: v.item() for k, v in batched_reward_dict.items()}
        self.last_reward_dict = reward_dict
        reward = self._get_total_reward(reward_dict)

        # Calculate the score.
        batched_score_dict = self.get_score_dict(batched_obs_dict,
                                                 batched_reward_dict)
        score_dict = {k: v.item() for k, v in batched_score_dict.items()}
        self.last_score_dict = score_dict

        # Get whether the episode should end.
        dones = self.get_done(batched_obs_dict, batched_reward_dict)
        done = dones.item()
        self.is_done = done

        # Combine the dictionaries as the auxiliary information.
        info = collections.OrderedDict()
        info.update(('obs/' + key, val) for key, val in obs_dict.items())
        info.update(('reward/' + key, val) for key, val in reward_dict.items())
        info['reward/total'] = reward
        info.update(('score/' + key, val) for key, val in score_dict.items())

        # info = {}
        return flattened_obs, reward, done, info

    def _step(self, action: Action):
        """Task-specific step for the environment."""
        dhand_act, sawyer_act = action[:16], action[16:]
        if not self.initializing:
            # NOTE: Order matters here, because dhand_robot wait controls freq
            self.sawyer_robot.step(sawyer_act)
            self.dhand_robot.step(dhand_act)

        # TODO: Move this into object.py
        if self.object.is_hardware:
            # NOTE: Match hardware to sim by adding an offset to the object z
            object_qpos = self.last_obs_dict["object_qpos"] + np.array([0, 0, 0.8, 0, 0, 0, 0])
            self.sim.data.qpos[self.object.config.qpos_indices] = object_qpos
            self.sim.forward()

    def _get_total_reward(
        self,
        reward_dict: Dict[str, np.ndarray],
    ) -> float:
        """Returns the total reward for the given reward dictionary.

        The default implementation extracts the keys from `reward_keys` and sums
        the values.

        Parameters
        ----------
        reward_dict : Dict[str, np.ndarray]
            A dictionary of rewards. The values may have a batch dimension.

        Returns
        -------
        reward : float
            The total reward for the dictionary.
        """
        if self._reward_keys_and_weights:
            reward_values = (
                reward_dict[key] * weight
                for key, weight in self._reward_keys_and_weights.items()
            )
        else:
            reward_values = reward_dict.values()
        reward = np.sum(np.fromiter(reward_values, dtype=float))
        return reward

    def reset_robot(self):
        """This method performs a robot reset of the Sawyer + Dhand in the
        following sequence:
        1. Set the Sawyer joint angles and Dhand qpos to the env initial qpos
        2. Set the velocities for Sawyer + Dhand joints to the env initial qvel (0)
        3. If running on hardware: do a reset of the Dhand motors
        4. If we want to recenter the hand over the object upon reset, then
           calculate the object position and set the Sawyer state accordingly.
        5. Let the simulation run for some ticks, and step the sim forward.
        """
        if not self._reset_robot:
            return

        self.sim.data.qpos[:-7] = self.init_qpos[:-7]
        self.sim.data.qvel[:] = self.init_qvel[:]

        if self.dhand_robot.is_hardware:
            self.dhand_robot.reset()

        if self._readjust_to_object_in_reset:
            obj_xyz = self.object.get_state().qpos[:3]
            hand_xyz, hand_euler = (
                self.sawyer_robot.reset_state.pos,
                quat2euler(self.sawyer_robot.reset_state.quat)
            )
            if self._readjust_hand_xyz:
                hand_xyz = obj_xyz + self.obj_offset
            if self._readjust_hand_euler:
                hand_euler = self.get_hand_euler()
            self.sawyer_robot.set_state(SawyerRobotState(
                pos=hand_xyz,
                quat=euler2quat(hand_euler),
            ))
        else:
            # We want to return the Sawyer mocap to its mean position
            self.sawyer_robot.reset(command_angles=True)

        # wait for sim to stablize
        self.do_simulation(self.act_mid - 0.75 * self.act_rng, 300)
        self.sim.forward()

    def get_hand_euler(self):
        # TODO: this currently only works for the rod
        obj_xyProj = self.sim.data.body_xmat[-1].reshape(3,3)[:, 2]
        obj_xyProj[2] = 0 # project on XY plane
        obj_xyProj /= np.linalg.norm(obj_xyProj) # normalize
        sign = -1 if obj_xyProj[0] < 0 else 1 # object is symmetric
        obj_zTheta = np.arcsin(sign * obj_xyProj[1])
        hand_zTheta = obj_zTheta + np.pi / 2 # hand should be perpendicular to obj
        if(hand_zTheta < np.pi/2): # wrap around
            hand_zTheta = -hand_zTheta
        else:
            hand_zTheta = -hand_zTheta + np.pi
        hand_euler = np.array([np.pi, 0, hand_zTheta])
        return hand_euler

    def reset(self):
        """Resets the environment. This completely overrides the reset of
        MujocoEnv to utilize an abstract method `_reset` to be implemented by
        subclasses instead of using the `reset_model` method.

        Returns:
            The initial observation of the environment after resetting.
        """
        # === new ===
        # TODO: Removed this due to releasing the object (move releasing for
        # the object in sim into the dhand_robot reset) Currently just doing
        # a set_state which might be dangerous if the object is in the hand
        # self.sim.data.qpos[:-7] = self.init_qpos[:-7]
        # self.sim.data.qvel[:] = self.init_qvel[:]

        # Soft reset of the Sawyer in order to allow object to reset
        if self.sawyer_robot.is_hardware:
            self.sawyer_robot.reset()

        # Soft reset of the hand
        if self.dhand_robot.is_hardware:
            self.dhand_robot.reset()

        # Perform environment specific reset (aka object reset)
        self._reset()

        # Reset the robot
        if self._reset_robot:
            self.reset_robot()

        self._reset_count += 1
        return self._get_obs()

    def _should_reset(self):
        return (
            self._reset_count == 0
            or (self._reset_every_n_episodes > 0
                and self._reset_every_n_episodes != np.inf
                and self._reset_count % self._reset_every_n_episodes == 0)
        )

    def _get_obs(self, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> Observation:
        """Returns the current observation of the environment.

        This matches the environment's observation space.
        """
        if obs_dict is None:
            obs_dict = self.get_obs_dict()
        if self._use_dict_obs:
            if self._observation_keys:
                obs = collections.OrderedDict(
                    (key, obs_dict[key]) for key in self._observation_keys)
            else:
                obs = obs_dict
        else:
            if self._observation_keys:
                obs_values = (obs_dict[key] for key in self._observation_keys)
            else:
                assert isinstance(obs_dict, collections.OrderedDict), \
                    'Must use OrderedDict if not using `observation_keys`'
                obs_values = obs_dict.values()
            obs = np.concatenate([np.ravel(v) for v in obs_values])
        return obs

    def evaluate_success(self, paths, logger=None):
        success = 0.0
        for p in paths:
            if np.mean(p['env_infos']['solved'][-4:]) > 0.0:
                success += 1.0
        success_rate = 100.0 * success / len(paths)

        if logger is None:
            # nowhere to log so return the value
            return success_rate
        else:
            # log the success
            logger.log_kv('success_rate', success_rate)
            return None

    # --------------------------------
    # Get and set states
    # --------------------------------
    def get_env_state(self):
        return dict(
            qp=self.data.qpos.copy(),
            qv=self.data.qvel.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
            site_pos=self.data.site_xpos.copy(),
            body_pos=self.model.body_pos.copy(),
            body_quat=self.model.body_quat.copy(),
        )

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.sim.data.mocap_pos[:] = state['mocap_pos']
        self.sim.data.mocap_quat[:] = state['mocap_quat']
        self.sim.data.site_xpos[:] = state['site_pos']
        self.sim.model.body_pos[:] = state['body_pos']
        self.sim.model.body_quat[:] = state['body_quat']
        self.sim.forward()

    # --------------------------------
    # Rendering
    # --------------------------------
    def render(
            self,
            mode: str = "rgb_array",
            width: int = 480,
            height: int = 480,
            render_cameras: list = None,
            use_hardware: bool = False,
            stack_channel: bool = True,
    ):

        if render_cameras is None:
            render_cameras = self._render_cameras

        # if self.is_hardware and self._image_service is not None:
        #     return self._image_service.get_image(width=width, height=height)

        if mode == "rgb_array" or mode == "offscreen":
            if not self._offscreen_renderers:
                for camera_id in render_cameras:
                    self._mj_offscreen_viewer_setup(camera_id=camera_id)

            images = []
            for camera_id in render_cameras:
                viewer = self._offscreen_renderers[-1]
                viewer.render(width, height, camera_id)
                data = viewer.read_pixels(width, height, depth=False)[::-1, :, :]
                if camera_id == 0:
                    data = data[::-1, ::-1, :]
                images.append(data)
            if stack_channel:
                return np.dstack(images)
            else:
                return np.concatenate(images, axis=1)

        elif mode == "human" or mode == "onscreen":
            if not self._onscreen_renderer:
                self._mj_onscreen_viewer_setup()
            self._onscreen_renderer.render()
        else:
            raise NotImplementedError("Invalid mode = {}".format(mode))

    def _mj_offscreen_viewer_setup(self, camera_id: int = -1):
        """Initializes offscreen viewer for headless rendering.
        """
        viewer = mujoco_py.MjRenderContextOffscreen(
            self.sim, device_id=camera_id)
        if camera_id == -1:  # free camera
            if "azimuth" in self._camera_configs:
                viewer.cam.azimuth = self._camera_configs["azimuth"]
            if "distance" in self._camera_configs:
                viewer.cam.distance = self._camera_configs["distance"]
            if "elevation" in self._camera_configs:
                viewer.cam.elevation = self._camera_configs["elevation"]
            if "lookat" in self._camera_configs:
                viewer.cam.lookat[:] = self._camera_configs["lookat"]
        elif camera_id == 0:
            # viewer.cam.azimuth = 0
            # # viewer.cam.azimuth = -135  # This is facing the Sawyer slightly
            # viewer.cam.distance = 2.0
            # viewer.cam.elevation = -30
            # viewer.cam.lookat[:] = self.init_qpos[-7:-4]
            pass
        elif camera_id == 1:
            # viewer.cam.azimuth = 90
            # # viewer.cam.azimuth = -135  # This is facing the Sawyer slightly
            # viewer.cam.distance = 2.0
            # viewer.cam.elevation = -30
            # viewer.cam.lookat[:] = self.init_qpos[-7:-4]
            pass

        self._offscreen_renderers[camera_id] = viewer

    def _mj_onscreen_viewer_setup(self):
        """Initializes onscreen viewer to show in window.
        """
        viewer = mujoco_py.MjViewer(self.sim)
        viewer.cam.azimuth = -90
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -30
        viewer.cam.lookat[:] = self.init_qpos[-7:-4]
        self._onscreen_renderer = viewer

    def close(self):
        pass

    # --------------------------------
    # Abstract methods to override
    # --------------------------------
    # @abc.abstractmethod
    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns
        -------
        obs_dict : Dict[str, np.ndarray]
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        obs_dict = collections.OrderedDict((
            ('t', np.array([self.sim.data.time])),
        ))
        obs_dict.update(self.dhand_robot.get_obs_dict())
        obs_dict.update(self.sawyer_robot.get_obs_dict())
        obs_dict.update(self.object.get_obs_dict())
        return obs_dict

    @abc.abstractmethod
    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation.

        Parameters
        ----------
        action : np.ndarray
            A batch of actions.
        obs_dict : Dict[str, np.ndarray]
            A dictionary of batched observations. The batch dimension
            matches the batch dimension of the actions.

        Returns
        -------
        reward_dict : Dict[str, np.ndarray]
            A dictionary of reward components. The values should be batched to
            match the given actions and observations.
        """

    @abc.abstractmethod
    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment.

        Parameters
        ----------
        obs_dict : Dict[str, np.ndarray]
            A dictionary of batched observations.
        reward_dict : Dict[str, np.ndarray]
            A dictionary of batched rewards to correspond with the observations.

        Returns
        -------
        score_dict : Dict[str, np.ndarray]
            A dictionary of scores.
        """

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate.

        Parameters
        ----------
        obs_dict : Dict[str, np.ndarray]
            A dictionary of batched observations.
        reward_dict : Dict[str, np.ndarray]
            A dictionary of batched rewards to correspond with the observations.

        Returns
        -------
        done : bool
            A boolean to denote if the episode should terminate. This should
            have the same batch dimension as the observations and rewards.
        """
        del obs_dict
        return np.zeros_like(next(iter(reward_dict.values())), dtype=bool)

    @abc.abstractmethod
    def _reset(self):
        raise NotImplementedError
