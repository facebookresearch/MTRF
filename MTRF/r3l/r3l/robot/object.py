# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
import time

from r3l.robot.robot import Robot
from r3l.robot.robot_config import ObjectConfig
from r3l.robot.default_configs import DEFAULT_OBJECT_CONFIG
from r3l.robot.dynamixel_client import DynamixelClient


class ObjectState:
    def __init__(
            self, qpos=None, qvel=None, spool_qpos=None, spool_qvel=None, spool_curr=None):
        self.qpos = qpos
        self.qvel = qvel
        self.spool_qpos = spool_qpos
        self.spool_qvel = spool_qvel
        self.spool_curr = spool_curr


class ObjectListener:
    def __init__(self):
        import rospy
        from rospy_tutorials.msg import Floats
        from geometry_msgs.msg import Pose
        # NOTE: The object state will add on these offsets after reading
        # the OptiTrack data in order to align with the Sawyer axes

        """
        ### Calibration

        1. align the object such that the purple end at the middle of the reset wall
        2. Take a picture of optitrack terminal
        3. move the sawyer arm such that (palm as flat as possible) and (fingers touch the sides of reset wall)
        4. Take a picture of sawyer terminal
        5. enter the numbers below
        """


        sawyer_cali = np.array([0.77, 0.03, 0.10])
        object_cali = np.array([0.209, 0.069, 0.177])

        # swap the axis for object coord
        object_cali = np.array([object_cali[0],
                                -object_cali[2],
                                object_cali[1]
                                ])

        cali_offset = sawyer_cali - object_cali

        self.OFFSETS = np.array([
            # Sawyer coords - OptiTrack coords
            # i.e.: OptiTrack read coords + offset = Sawyer read coords
            # ROD OFFSETS
            # 0.539 - (0.128),
            # -0.05 - (-0.028),
            # 0.161 - (0.247 - 0.05),  # Z offsets
            # VALVE OFFSETS
            cali_offset[0],
            cali_offset[1],
            cali_offset[2],
            # TODO: Need to fix the OptiTrack axes?
            0, 0, 0, 0,
        ])
        self._object_state = ObjectState(
            qpos=np.zeros(7),
            qvel=np.zeros(6),
        )
        rospy.Subscriber("get_object_pose", Pose, self.update_qpos)
        rospy.sleep(1)

    def update_qpos(self, pose):
        self._object_state.qpos = np.array([
            pose.position.x,
            -pose.position.z,
            pose.position.y, # Note: this is for correcting the different axis permutation between optitrack and sawyer
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]) + self.OFFSETS

    def update_qvel(self, qvel):
        qvel_data = np.array(qvel.data)
        self._object_state.qvel = qvel_data

    def get_state(self):
        return self._object_state


class Object(Robot):
    def __init__(
            self,
            env=None,
            config_params: dict = DEFAULT_OBJECT_CONFIG,
    ):
        super().__init__(env=env)
        config = self.config = ObjectConfig(sim=self._sim, **config_params)
        device_path = config.device_path
        motor_ids = config.motor_ids
        self._hardware = None
        if device_path and motor_ids is not None:
            self._hardware = DynamixelClient(
                motor_ids, port=device_path, lazy_connect=True)
            self.listener = ObjectListener()

    def step(self, action):
        # Shouldn't be calling step on the object.
        pass

    @property
    def is_hardware(self):
        return self.config.device_path is not None

    def get_obs_dict(self):
        state = self.get_state()
        return collections.OrderedDict((
            ("object_qpos", state.qpos),
            ("object_qvel", state.qvel),
        ))

    def get_state(self):
        if self.is_hardware:
            # TODO(justinvyu): Create a copy instead so that nobody can modify
            # internal state
            state = self.listener.get_state()
            spool_qpos, spool_qvel, spool_curr = self._hardware.read_pos_vel_cur()
            return ObjectState(
                qpos=state.qpos,
                qvel=state.qvel,
                spool_qpos=spool_qpos,
                spool_qvel=spool_qvel,
                spool_curr=spool_curr,
            )
            return state
        else:
            if self._env.initializing:
                return ObjectState(
                    qpos=np.zeros_like(self.config.qpos_indices),
                    qvel=np.ones_like(self.config.qvel_indices)
                )
            return ObjectState(
                qpos=self._sim.data.qpos[self.config.qpos_indices].copy(),
                qvel=self._sim.data.qvel[self.config.qvel_indices].copy(),
            )

    def set_state(self, state: ObjectState):
        if self.is_hardware:
            # Usually, there is nothing to do here, but possible to
            # add instrumented reset mechanism here.
            pass
        else:
            if state.qpos is not None:
                self._sim.data.qpos[self.config.qpos_indices] = state.qpos
            if state.qvel is not None:
                self._sim.data.qvel[self.config.qvel_indices] = state.qvel
            # self._sim.forward()

    def reset(self):
        # import IPython
        # IPython.embed()

        if not self.is_hardware:
            return

        self._hardware.set_torque_enabled(self.config.motor_ids, True)
        CURRENT_LIM = [140]
        # CURRENT_LIM = [60, 60]

        vel = 20
        spool_vel = np.array([-vel]) # Tony: motor 2 not used

        print("Wire IN")
        done = [False, True] # Tony: motor 2 done
        # print("\n\n SPOOLING IN \n\n")
        while not all(done):
            state = self.get_state()
            cur = state.spool_curr
            # print(state.spool_qpos)
            if abs(cur[0]) >= CURRENT_LIM[0]:
                spool_vel[0] = 0
                done[0] = True
            self._hardware.write_desired_vel(self.config.motor_ids, spool_vel)
            time.sleep(0.05)

        print("Wire OUT")
        print("note: only partially released for now  to save time")
        unspool_pos = state.spool_qpos + np.array([40]) # Tony: 110 is all wire
        # print("\n\n SPOOLING OUT \n\n")
        spool_vel = np.array([23])
        done = [False, True] # Tony: motor 2 done
        while not all(done):
            state = self.get_state()
            spool_qp = state.spool_qpos
            # print(spool_qp)
            diff = spool_qp - unspool_pos
            if np.isclose(diff[0], 0, atol=2):
                # Stop changing the 1st motor qpos
                spool_vel[0] = 0
                done[0] = True
            self._hardware.write_desired_vel(self.config.motor_ids, spool_vel)
            time.sleep(0.05)

        self._hardware.set_torque_enabled(self.config.motor_ids, False)
        time.sleep(0.5)

