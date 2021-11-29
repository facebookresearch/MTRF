# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Union, Tuple
from enum import Enum
from r3l.utils.quatmath import quat2euler

class RobotConfig:
    """Stores configuration for a robot."""

    def __init__(
            self,
            sim,
            qpos_indices: Optional[Iterable[int]] = None,
            qvel_indices: Optional[Iterable[int]] = None,
            calib_offset = None,
            calib_scale = None,
            # actuator_indices: Optional[Iterable[int]] = None,
            qpos_range: Optional[Iterable[Tuple[float, float]]] = None,
            qvel_range: Optional[Iterable[Tuple[float, float]]] = None,
            # actuator_range: Optional[Iterable[Tuple[float, float]]] = None,
            control_mode: str = "deltapos",
            # sim_observation_noise: float = 0.0,
            # use_raw_actions: bool = False,
    ):
        """Initializes a new configuration for a RobotComponent group.

        Args:
            sim_scene: The simulation, used for validation purposes.
            control_mode: The control mode for the actuators.
            qpos_indices: The joint position indices that this group reads and
                writes to for the robot state.
            qvel_indices: The joint velocity indices that this group reads and
                writes to for the robot state. If not given, this defaults to
                `qpos_indices`.
            actuator_indices: The MuJoCo actuator indices that this group writes
                to for actions. If not provided, then will default one of the
                following depending on the control mode:
                - joint torque: None
                - joint position: `qpos_indices`
                - joint velocity: `qvel_indices`
                If None, this group will not write actions to the simulation.
            qpos_range: The position range, as a (lower, upper) tuple in
                generalized joint position space. This clamps the writable
                values for position state.
                e.g. For a hinge joint, this is measured in radians.
            qvel_range: The velocity range, as a (lower, upper) tuple in
                generalized joint velocity space. This bounds the writable
                values for position commands based on the difference from the
                current position.
                e.g. For a hinge joint, this is measured in radians/second.
            actuator_range: The actuator control range, as a (lower, upper)
                tuple in control space. The bounds the writable values for
                commands and defines the normalization range. If not provided,
                then will default to one of the following depending on control
                mode:
                - joint torque: None
                - joint position: `qpos_range`
                - joint velocity: `qvel_range`
                If None, the default range from the simulation is used.
            sim_observation_noise: The relative noise amplitude to add to
                state read from simulation.
            use_raw_actions: If True, doesn't denormalize actions from `step()`.
        """
        # self.control_mode = control_mode
        # self.use_raw_actions = use_raw_actions

        # Ensure that the qpos indices are valid.
        self.qpos_indices = None
        if qpos_indices is not None:
            if sim:
                nq = sim.model.nq
                assert all(-nq <= i < nq for i in qpos_indices), \
                    'All qpos indices must be in [-{}, {}]'.format(nq, nq - 1)
            self.qpos_indices = np.array(qpos_indices, dtype=int)

        if qvel_indices is None:
            qvel_indices = qpos_indices

        # Ensure that the qvel indices are valid.
        self.qvel_indices = None
        if qvel_indices is not None:
            if sim:
                nv = sim.model.nv
                assert all(-nv <= i < nv for i in qvel_indices), \
                    'All qvel indices must be in [-{}, {}]'.format(nv, nv - 1)
            self.qvel_indices = np.array(qvel_indices, dtype=int)

        # self.qpos_calib = qpos_calib
        self.calib_offset = calib_offset
        self.calib_scale = calib_scale
        # self.sim_observation_noise = sim_observation_noise

        # Convert the ranges to matrices.
        self.qpos_range = None
        if qpos_range is not None:
            assert all(lower <= upper for lower, upper in qpos_range), \
                'Items in qpos_range must follow (lower, upper)'
            self.qpos_range = np.array(qpos_range, dtype=np.float32)
            assert self.qpos_range.shape == (len(self.qpos_indices), 2), \
                'qpos_range must match the length of qpos_indices'

        self.qvel_range = None
        if qvel_range is not None:
            assert all(lower < upper for lower, upper in qvel_range), \
                'Items in qvel_range must follow (lower, upper)'
            self.qvel_range = np.array(qvel_range, dtype=np.float32)
            assert self.qvel_range.shape == (len(self.qvel_indices), 2), \
                'qvel_range must match the length of qpos_indices'

        self.control_mode = control_mode
        # if actuator_indices is None:
        #     if self.control_mode == ControlMode.JOINT_POSITION:
        #         actuator_indices = self.qpos_indices
        #     elif self.control_mode == ControlMode.JOINT_VELOCITY:
        #         actuator_indices = self.qvel_indices

        # Ensure that the actuator indices are valid.
        # self.actuator_indices = None
        # if actuator_indices is not None:
        #     nu = sim.model.nu
        #     assert all(-nu <= i < nu for i in actuator_indices), \
        #         'All actuator indices must be in [-{}, {}]'.format(
        #             nu, nu - 1)
        #     self.actuator_indices = np.array(actuator_indices, dtype=int)

        # if actuator_range is None:
        #     if self.control_mode == ControlMode.JOINT_POSITION:
        #         actuator_range = self.qpos_range
        #     elif self.control_mode == ControlMode.JOINT_VELOCITY:
        #         actuator_range = self.qvel_range
        #     # Default to use the simulation's control range.
        #     if actuator_range is None and actuator_indices is not None:
        #         actuator_range = sim.model.actuator_ctrlrange[
        #             actuator_indices, :]

        # self.actuator_range = None
        # if actuator_range is not None:
        #     assert all(lower < upper for lower, upper in actuator_range), \
        #         'Items in actuator_range must follow (lower, upper)'
        #     self.actuator_range = np.array(actuator_range, dtype=np.float32)
        #     assert (self.actuator_range.shape ==
        #             (len(self.actuator_indices), 2)), \
        #         'actuator_range must match the length of actuator_indices'

        # Calculate the denormalization center and range from the sim model.
        # self.denormalize_center = None
        # self.denormalize_range = None
        # if self.actuator_range is not None:
        #     self.denormalize_center = np.mean(self.actuator_range, axis=1)
        #     self.denormalize_range = 0.5 * (
        #         self.actuator_range[:, 1] - self.actuator_range[:, 0])

    # @property
    # def is_active(self) -> bool:
    #     """Returns True if the group is not in use."""
    #     return self.actuator_indices is not None


class DhandRobotConfig(RobotConfig):
    def __init__(
            self,
            *args,
            device_path: str = None,
            motor_ids: Sequence[int] = None,
            hand_velocity_lim: np.ndarray = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device_path = device_path
        self.motor_ids = motor_ids
        if motor_ids is not None:
            self.motor_ids = np.array(sorted(motor_ids), dtype=int)

        self.hand_velocity_lim = hand_velocity_lim
        if hand_velocity_lim is None:
            self.hand_velocity_lim = 0.8 * np.ones_like(self.qpos_indices)


class SawyerRobotConfig(RobotConfig):
    class ControlType(Enum):
        DeltaPosition = 0
        Position = 1

    def __init__(
            self,
            *args,
            mocap_name: str = "mocap",
            mocap_range: Tuple[np.ndarray, np.ndarray] = None,
            mocap_velocity_lim: np.ndarray = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mocap_name = mocap_name
        self.mocap_velocity_lim = mocap_velocity_lim

        if mocap_range is None:
            mean_pos = np.array([0.705, 0.185, 1.0])
            mean_quat = np.array([0, 1, 0, 0])
            mean_euler = quat2euler(mean_quat)
            one_side_range = np.array([.4, .4, .1, 0, 0, .02])
            mean = np.concatenate([mean_pos, mean_euler])
            self.mocap_range = np.array([
                mean - one_side_range, mean + one_side_range]).T
        else:
            if len(mocap_range) == 2:
                low, high = mocap_range[0], mocap_range[1]
                self.mocap_range = np.hstack([low.reshape(-1, 1), high.reshape(-1, 1)])
            else:
                self.mocap_range = np.array(mocap_range)

        if mocap_velocity_lim is None:
            self.mocap_velocity_lim = np.array([0.01, 0.01, 0.01, 0., 0., 0.02])


class ObjectConfig(DhandRobotConfig):
    pass
