# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import copy
from enum import Enum
import gym
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, NewType, Sequence, Union, Tuple
import abc

from r3l.r3l_envs.inhand_env.base import SawyerDhandInHandObjectBaseEnv, ObjectType
from r3l.utils.quatmath import quat2euler, euler2quat
from r3l.utils.circle_math import circle_distance, circle_distance_mod


class SawyerDhandInHandManipulateObjectMultiPhase(SawyerDhandInHandObjectBaseEnv, metaclass=abc.ABCMeta):
    def __init__(
            self,
            phase_envs_params,
            phase_policy_directories=[],
            training_phases=None,
            shared_viewer_mode="human",
            max_episodes_in_phase=np.inf,
            max_episodes_stuck=np.inf,
            stuck_radius=0.05,
            commanded_phase_changes=False,
            drop_object_on_pickup=False,
            random_task_graph=False,
            perturb_off=False,
            observation_keys=SawyerDhandInHandObjectBaseEnv.ROD_DEFAULT_OBSERVATION_KEYS,
            include_phase_in_obs=False,
            **kwargs
    ):
        assert phase_envs_params, "Need to specify at least one phase"
        self.phase = 0
        self._perturb_off = perturb_off
        self._random_task_graph = random_task_graph
        self._stuck_reset_count = 0
        self.phase_env_params = copy.deepcopy(phase_envs_params)

        if include_phase_in_obs:
            if "phase" not in observation_keys:
                observation_keys = observation_keys + ("phase", )

        # Create an env from scratch for a shared sim
        env_params = dict(
            sim=None,
            task_name="Multi Phase Manipulate",
            reset_every_n_episodes=np.inf,
            readjust_to_object_in_reset=True,
            readjust_hand_euler=False,
            observation_keys=observation_keys,
            reset_robot=False,
        )
        env_params.update(kwargs)
        super().__init__(**env_params)

        self.active_env = None
        # Contains all phase environments.
        self._envs = []

        # Initialize all phase envs using current sim
        for env_params in phase_envs_params:
            assert (
                isinstance(env_params, dict)
                and "domain" in env_params
                and "task" in env_params), "Need to specify domain+task"
            env_params = env_params.copy()
            domain = env_params.pop("domain")
            task = env_params.pop("task")
            env_kwargs = env_params.pop("env_kwargs", {})
            # Replace envs with same sim as base environment
            env_kwargs["sim"] = self.sim
            env = gym.make(f"{domain}{task}", **env_kwargs)
            env.reset()
            self._envs.append(env)

        # Total number of phases
        self._num_phases = len(self._envs)
        # Number of failed attempts that will trigger an automatic phase switch
        self._max_episodes_in_phase = max_episodes_in_phase
        self._num_episodes_per_phase = np.zeros(self._num_phases)

        # Number of episodes where the object hasn't moved much will trigger a full reset
        self._max_episodes_stuck = max_episodes_stuck
        # Radius to determine whether or not the object hasn't moved in the past n episodes
        self._stuck_radius = stuck_radius
        self._obj_xyz_history = None
        if 0 < max_episodes_stuck < np.inf:
            self._obj_xyz_history = collections.deque(maxlen=self._max_episodes_stuck)
        self._object_xy_range = 0
        self._drop_object_on_pickup = drop_object_on_pickup

        # Whether or not the algorithm commands phases upon reset
        # (versus the environment itself assigning phase)
        self._commanded_phase_changes = commanded_phase_changes
        # Which phases to train (every other phase will use a trained policy to execute)
        # TODO: This is just defaulting to training all phases right now
        if training_phases is None:
            self._training_phases = list(range(self._num_phases))
        else:
            self._training_phases = training_phases

        self.configure_phase(self.phase)

        if shared_viewer_mode == "rgb_array" or shared_viewer_mode == "offscreen":
            self._mj_offscreen_viewer_setup()
            # Set all offscreen renderers to the same render context
            for env in self._envs:
                if hasattr(env, "_offscreen_renderer"):
                    env._offscreen_renderer = self._offscreen_renderer
        elif shared_viewer_mode == "human" or shared_viewer_mode == "onscreen":
            self.mujoco_render_frames=True
            for env in self._envs:
                env.mujoco_render_frames = self.mujoco_render_frames
        else:
            raise NotImplementedError(f"Invalid mode = {shared_viewer_mode}")

        # Load trained policies
        self._phase_policies = []
        if phase_policy_directories is None or not phase_policy_directories:
            self._phase_policies = [None] * self._num_phases
        else:
            if len(phase_policy_directories) == self._num_phases:
                for phase_policy_dir in phase_policy_directories:
                    if phase_policy_dir:
                        self._phase_policies.append(self._load_policy(phase_policy_dir))
                    else:
                        self._phase_policies.append(None)
            else:
                for ph_num in range(self._num_phases):
                    self._phase_policies.append(self._load_policy(phase_policy_directories[0], phase_num=ph_num))

    def turn_perturb_off(self):
        self._perturb_off = True

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_obs_dict()
        if self.initializing:
            phase_context = np.zeros((18,))
            object_xy_range = np.zeros((1,))
        else:
            obs_dict.update(self.active_env.get_obs_dict())
            phase_context = np.concatenate([
                self.active_env.sawyer_robot.config.mocap_range.mean(axis=1),
                self.active_env.sawyer_robot.config.mocap_velocity_lim,
            ])
            object_xy_range = np.array([self._object_xy_range])
            obs_dict.update({
                f"phase{i}/count": self._num_episodes_per_phase[i]
                for i in range(self._num_phases)
            })

        # Update with phase and phase context (in the form of pos + vel limits)
        obs_dict.update({
            "phase": np.array([self.phase]),
            # "context": phase_context,
            "stuck_reset_count": np.array([self._stuck_reset_count]),
            "object_xy_max_distance_moved": object_xy_range,
        })
        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self.initializing:
            return { "temp": np.zeros(1) }
        return self.active_env.get_reward_dict(action, obs_dict)

    def _get_total_reward(
        self,
        reward_dict: Dict[str, np.ndarray],
    ) -> float:
        return self.active_env._get_total_reward(reward_dict)

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if self.initializing:
            return { "temp": np.zeros(1) }
        return self.active_env.get_score_dict(obs_dict, reward_dict)

    def configure_phase(self, phase_idx):
        # Require int index input
        assert (
            isinstance(phase_idx, int)
            and phase_idx >= 0
            and phase_idx < self._num_phases
        )
        self.active_env = self._envs[phase_idx]
        self.phase = phase_idx
        self._onscreen_renderer = self._envs[phase_idx]._onscreen_renderer

        # TODO(justinvyu): Find a better way to keep track of target instead of
        # relying on updating in the sim
        if hasattr(self.active_env, "reset_target"):
            self.active_env.reset_target()

    @property
    def num_phases(self):
        return self._num_phases

    def task_graph(self) -> int:
        """
        Must return a next phase, given a task graph generated by the current
        phase and the state of the environment. To be implemented by subclasses.
        """
        raise NotImplementedError

    def next_phase(self):
        """
        Chooses the next phase:
        1. If the commanded_phase_changes flag is True, keep the same phase and
           rely on the algorithm to set the phase using configure_phase.
        2. If the class implements a task graph, then query that.
        """
        next_phase = None
        if not self._commanded_phase_changes:
            next_phase = self.task_graph()
            if self._verbose:
                print(f"TASK GRAPH RETURNED NEXT PHASE = {next_phase}")

        if next_phase is None:
            return self.phase
        return next_phase

    def update_phase(self):
        self.configure_phase(self.next_phase())
        self._num_episodes_per_phase[self.phase] += 1
        return self.phase

    def execute_phase(self, phase, phase_cnt=0, save_video=False):
        assert 0 <= phase < self._num_phases
        print(f"EXECUTING PHASE {phase} FROM LOADED POLICY")
        env = self._envs[phase]
        policy = self._phase_policies[phase]
        horizon = 100  # TODO: fix this
        env.reset_robot()
        path = {'observations': [], 'actions': [], 'rewards': [], 'dones': [], 'infos': [], 'images': []}
        for _ in range(horizon):
            o = env.get_obs_dict()
            action = policy(env.get_obs_dict())
            no, r, d, info = env.step(action)
            path['observations'].append(o)
            path['rewards'].append(r)
            path['actions'].append(action)
            path['dones'].append(d)
            path['infos'].append(info)
            if save_video:
                path['images'].append(np.flip(np.flip(env.sim.render(mode='offscreen', width=480, height=480), axis=0), axis=1))
        return path

    def step(self, action):
        # TODO: Is this correct logic? This would basically ignore the policy
        # that is generated by the policy. And there would be a mismatch
        # between the recorded action taken and the observations seen with this
        # trained policy.
        # Make sure to disable training for the skipped policy to save time.
        if self.phase not in self._training_phases:
            action = self._phase_policies[self.phase](self.active_env.get_obs_dict())

        obs, rew, done, info = self.active_env.step(action)
        # TEMP FIX TO ADD IN PHASE TO THE ACTIVE ENV's OBS
        if "phase" in self._observation_keys:
            obs.update({"phase": np.array([self.phase])})
        return obs, rew, done, info

    def _release_object(self):
        if self._verbose:
            print("Dropping object")
        self.do_simulation(np.zeros(16), 300)

    def get_count_bonuses(self, xys: np.ndarray):
        return self.active_env.get_count_bonuses(xys)

    def _reset(self):
        self.phase_old = self.phase
        # TODO: Check the hand is reset properly
        if self._should_reset():
            self.active_env.reset()
            # if self._reset_count > 0:
            self.update_phase()
            self.phase_cnt = 0
        else:
            # NOTE: Letting go of the object and allowing the simulation to step
            # is important to get the simulation to a steady state, ready for
            # the next `update_phase` call.
            if self._drop_object_on_pickup and self.get_obs_dict()["object_xyz"][2] > 0.85:
                self._release_object()

            # if self._reset_count > 0:
            self.update_phase()
            if self.phase != self.phase_old:
                self.phase_cnt = 0
                self.phase_old = self.phase
            else:
                # Count your days if you are stuck
                self.phase_cnt += 1

                # Move on with life, you are stuck in a phase or the object has been stuck
                # (even if you are solving the task, this is important to improve all phases)
                if self.phase_cnt >= self._max_episodes_in_phase:
                    if self._verbose:
                        print("Stuck in a phase, requesting forced reset")
                    # TODO: Not sure which reset to use here.
                    self.active_env.reset()
                    # TODO(justinvyu): I think this extra update_phase screws things up in logging
                    # self.update_phase()
                    self.phase_cnt = 0

            # Use a loaded policy to run through the phase, if we aren't
            # interested in training for it.
            # if self.phase not in self._training_phases:
            #     self.execute_phase(self.phase, self.phase_cnt)
            #     # Update the phase with the same logic as above
            #     if self.last_obs_dict["object_xyz"][2] > 0.85:  # drop object if picked
            #         if self._verbose:
            #             print("Dropping object")
            #         self.do_simulation(self.act_mid - 0.75 * self.act_rng, 300)
            #     self.phase_old = self.phase
            #     self.update_phase()
            #     self.phase_cnt = 0

            """
            If we specify a maximum number of episodes stuck, then keep track
            of the last `self._max_episodes_stuck` object positions. If the
            difference of the current position is less than a certain threshold
            `self._stuck_radius` relative to the some number of episodes ago,
            then do a full reset.
            """
            if 0 < self._max_episodes_stuck < np.inf:
                obj_xyz = self.get_obs_dict()["object_xyz"]
                self._obj_xyz_history.append(obj_xyz)
                # Need to have seen at least `_max_episodes_stuck` positions
                self._object_xy_range = 0
                if len(self._obj_xyz_history) == self._max_episodes_stuck:
                    xy_history = np.array(self._obj_xyz_history)[:, :2]
                    # Use the max L2 distance from latest timestep to some
                    # timestep in the last `max_episodes_stuck` timesteps
                    xy_range = self._object_xy_range = np.max(
                        np.linalg.norm(xy_history - xy_history[-1], axis=1)
                    )
                    # Use standard deviation to measure movement
                    # xyz_spread = np.array(self._obj_xyz_history)[:, :2].std(axis=0).mean()
                    if xy_range < self._stuck_radius:
                        self._stuck_reset_count += 1
                        self.active_env.reset()
                        self._obj_xyz_history.clear()

            # Run the reset policy if one exists
            if self._should_reset():
                self.active_env.run_reset_policy()

            # Always reset the robot, the active env will handle the case if it
            # shouldn't reset the robot
            self.active_env.reset_robot()

        if hasattr(self.active_env, "reset_target"):
            self.active_env.reset_target()

        if self._verbose:
            print("Current Phase: {}, Episodes since last phase change: {}".format(self.phase, self.phase_cnt))

    def set_goal(self, goal_index):
        """Consider the goal_index as the phase id."""
        self.configure_phase(goal_index)


