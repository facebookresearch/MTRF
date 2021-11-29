# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import logging
import numpy as np
import time
from typing import Any, Dict, Iterable, Optional, NewType, Sequence, Union, Tuple

from r3l.robot.robot import Robot
from r3l.robot.robot_config import DhandRobotConfig
from r3l.robot.default_configs import DEFAULT_DHAND_ROBOT_CONFIG
from r3l.robot.dynamixel_client import DynamixelClient

# Default tolerance for determining if the hardware has reached a state.
DEFAULT_ERROR_TOL = 5. * np.pi / 180


class DhandRobotState:
    def __init__(self, qpos=None, qvel=None):
        self.qpos = qpos
        self.qvel = qvel


class DhandRobot(Robot):
    def __init__(
            self,
            config_params: dict = DEFAULT_DHAND_ROBOT_CONFIG,
            env=None,
    ):
        super().__init__(env=env)

        config = self.config = DhandRobotConfig(sim=self._sim, **config_params)
        device_path = config.device_path
        motor_ids = config.motor_ids
        self._hardware = None
        if device_path and motor_ids is not None:
            self._hardware = DynamixelClient(
                motor_ids, port=device_path, lazy_connect=True)

        assert device_path or env, (
            "Need to specify either a hardware source or"
            " environment simulation"
        )

    @property
    def is_hardware(self):
        return self.config.device_path is not None

    def get_obs_dict(self):
        state = self.get_state()
        return collections.OrderedDict((
            ("dhand_qpos", state.qpos.copy()),
            ("dhand_qvel", state.qvel.copy()),
        ))

    def step(self, action: np.ndarray):
        if self.config.control_mode == "deltapos":
            action = np.clip(action, -1.0, 1.0)
            action = self.get_state().qpos + action * self.config.hand_velocity_lim
        elif self.config.control_mode == "pos":
            action = action
        else:
            raise NotImplementedError

        if self.is_hardware:
            self.set_state(DhandRobotState(qpos=action))
            # self._synchronize_timestep()
            # state = DhandRobotState(qpos=action)
            # self.set_state(state)
            # Synchronization for sim.
            # if state.qpos is not None:
            #     self._sim.data.qpos[self.config.qpos_indices] = state.qpos
            # if state.qvel is not None:
            #     self._sim.data.qvel[self.config.qvel_indices] = state.qvel
            # self._sim.forward()

        else:
            self._env.do_simulation(action, self._env.frame_skip)

    def hardware_to_sim(self, state):
        assert state.qpos is not None
        if self.config.calib_scale is not None:
            state.qpos *= self.config.calib_scale
            if state.qvel is not None:
                state.qvel *= self.config.calib_scale
        if self.config.calib_offset is not None:
            state.qpos += self.config.calib_offset
        return state

    def sim_to_hardware(self, state):
        assert state.qpos is not None
        if self.config.calib_offset is not None:
            state.qpos -= self.config.calib_offset
        if self.config.calib_scale is not None:
            state.qpos /= self.config.calib_scale
            if state.qvel is not None:
                state.qvel /= self.config.calib_scale
        return state

    def get_state(self):
        if self.is_hardware:
            qpos, qvel, _ = self._hardware.read_pos_vel_cur()
            state = DhandRobotState(qpos=qpos, qvel=qvel)
            # Return the hardware readings in sim space
            return self.hardware_to_sim(state)
        else:
            if self._env.initializing:
                return DhandRobotState(
                    qpos=np.zeros(len(self.config.qpos_indices)),
                    qvel=np.zeros(len(self.config.qvel_indices)))
            return DhandRobotState(
                qpos=self._sim.data.qpos[self.config.qpos_indices],
                qvel=self._sim.data.qvel[self.config.qvel_indices]
            )

    def set_state(self, state: DhandRobotState):
        config = self.config
        # Clip the position and velocity to the configured bounds.
        clipped_state = DhandRobotState(qpos=state.qpos, qvel=state.qvel)
        # qpos ranges are in the sim space
        if clipped_state.qpos is not None and config.qpos_range is not None:
            clipped_state.qpos = np.clip(clipped_state.qpos,
                                         config.qpos_range[:, 0],
                                         config.qpos_range[:, 1])
        if clipped_state.qvel is not None and config.qvel_range is not None:
            clipped_state.qvel = np.clip(clipped_state.qvel,
                                         config.qvel_range[:, 0],
                                         config.qvel_range[:, 1])

        if self.is_hardware:
            # Set the controls in hardware space
            clipped_state = self.sim_to_hardware(clipped_state)
            self._set_state_hardware(clipped_state)
        else:
            if state.qpos is not None:
                self._sim.data.qpos[self.config.qpos_indices] = clipped_state.qpos
            if state.qvel is not None:
                self._sim.data.qvel[self.config.qvel_indices] = clipped_state.qvel

    def reset(self):
        self._hardware.set_torque_enabled(self.config.motor_ids, True)
        reset_state = DhandRobotState(
            qpos=self._env.init_qpos[self.config.qpos_indices],
            qvel=self._env.init_qvel[self.config.qvel_indices],
        )
        self.set_state(reset_state)

    def _set_state_hardware(self, state: DhandRobotState, block: bool = True):
        control = state.qpos
        if control is not None:
            self._hardware.write_desired_pos(self.config.motor_ids, control)

        # # Block until we've reached the given states, also to achieve 10Hz
        if block:
            self._wait_for_desired_states(state, initial_sleep=None)
        #     # Reset the step time.
            # self.reset_time()

    def _wait_for_desired_states(
            self,
            desired_state: DhandRobotState,
            error_tol: float = DEFAULT_ERROR_TOL,
            timeout: float = 0.1,
            poll_interval: float = 0.015,
            initial_sleep: Optional[float] = 0.25,
            last_diff_tol: Optional[float] = DEFAULT_ERROR_TOL,
            last_diff_ticks: int = 2,
    ):
        """Polls the current state until it reaches the desired state.
        Args:
            desired_states: The desired states to wait for.
            error_tol: The maximum position difference within which the desired
                state is considered to have been reached.
            timeout: The maximum amount of time to wait, in seconds.
            poll_interval: The interval in seconds to poll the current state.
            initial_sleep: The initial time to sleep before polling.
            last_diff_tol: The maximum position difference between the current
                state and the last state at which motion is considered to be
                stopped, thus waiting will terminate early.
            last_diff_ticks: The number of cycles where the last difference
                tolerance check must pass for waiting to terminate early.
        """

        # Define helper function to compare two states.
        def all_close(state_a, state_b, tol):
            return np.allclose(state_a.qpos, state_b.qpos, atol=tol)

        # Poll for the hardware move command to complete.
        previous_state = None
        ticks_until_termination = last_diff_ticks
        start_time = time.time()

        if initial_sleep is not None and initial_sleep > 0:
            time.sleep(initial_sleep)

        while True:
            cur_state = self.get_state()
            # Terminate if the current states have reached the desired states.
            if all_close(cur_state, desired_state, tol=error_tol):
                return
            # Terminate if the current state and previous state are the same.
            # i.e. the robot is unable to move further.
            if previous_state is not None and all_close(
                    cur_state, previous_state, tol=last_diff_tol):
                if not ticks_until_termination:
                    logging.warning(
                        'Robot stopped motion; terminating wait early.')
                    return
                ticks_until_termination -= 1
            else:
                ticks_until_termination = last_diff_ticks

            if time.time() - start_time > timeout:
                logging.warning('Reset timed out after %1.1fs', timeout)
                return
            previous_state = cur_state
            time.sleep(poll_interval)