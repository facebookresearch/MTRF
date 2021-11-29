# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

import collections
import numpy as np
from r3l.utils.quatmath import quat2euler, euler2quat

from r3l.utils.mocap_utils import MocapActor
from r3l.robot.default_configs import DEFAULT_SAWYER_ROBOT_CONFIG
from r3l.robot.robot_config import SawyerRobotConfig
import time
from r3l.r3l_envs.base_env.mujoco_env import MujocoEnv
from r3l.robot.robot import Robot


class SawyerRobotState:
    def __init__(self, pos=None, quat=None, arm_qpos=None, arm_qvel=None):
        self.pos = pos
        self.quat = quat
        self.arm_qpos = arm_qpos
        self.arm_qvel = arm_qvel


class SawyerListener:
    """Listener node to ROS node that streams Sawyer position and angle.
    """
    def __init__(self):
        import rospy
        from std_msgs.msg import String
        from geometry_msgs.msg import Pose
        from rospy_tutorials.msg import Floats

        self._sawyer_state = SawyerRobotState(
            pos=np.zeros(3),
            quat=np.ones(4),
            arm_qpos=np.zeros(7),
            arm_qvel=np.zeros(7),
        )
        rospy.Subscriber("get_pose", Pose, self.update_pose)
        rospy.Subscriber("get_angles", Floats, self.update_angles)
        rospy.Subscriber("get_velocities", Floats, self.update_velocities)
        rospy.sleep(1)

    def update_pose(self, pose):
        self._sawyer_state.pos = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ])
        self._sawyer_state.quat = np.array([
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ])

    def update_angles(self, angles):
        angles_data = np.array(angles.data)
        self._sawyer_state.arm_qpos = angles_data

    def update_velocities(self, velocities):
        data = np.array(velocities.data)
        self._sawyer_state.arm_qvel = data

    def get_sawyer_state(self):
        return self._sawyer_state


class SawyerCommander:
    """Commander to ROS node to command sawyer positions.
    """
    def __init__(self):
        import rospy
        from std_msgs.msg import Bool
        from geometry_msgs.msg import Pose

        self.ros_node = rospy.init_node("Sawyer_robot")
        rate = rospy.Rate(13) # 10hz
        self.pub = rospy.Publisher("set_angles", Pose, queue_size=10)
        self.pub_type = rospy.Publisher("command_type", Bool, queue_size=11)
        rospy.sleep(1)

    def send_command(self, state: SawyerRobotState, command_angles=False):
        while self.pub.get_num_connections() == 0:
            # assert self.pub.get_num_connections() > 0, "No subscribers connected"
            print("No subscribers")
            time.sleep(0.5)
        self.pub_type.publish(command_angles)

        from geometry_msgs.msg import Pose
        pose = Pose()
        if command_angles:
            # 'right_j6': pose.position.x,
            # 'right_j5':pose.position.y,
            # 'right_j4':pose.position.z,
            # 'right_j3':pose.orientation.x,
            # 'right_j2':pose.orientation.y,
            # 'right_j1':pose.orientation.z,
            # 'right_j0':pose.orientation.w
            joint_angles = state.arm_qpos
            pose.position.x = joint_angles[0]
            pose.position.y = joint_angles[1]
            pose.position.z = joint_angles[2]
            pose.orientation.x = joint_angles[3]
            pose.orientation.y = joint_angles[4]
            pose.orientation.z = joint_angles[5]
            pose.orientation.w = joint_angles[6]
        else:
            pose.position.x, pose.position.y, pose.position.z = state.pos
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = state.quat

        self.pub.publish(pose)


class SawyerRobot(Robot):
    def __init__(
            self,
            config_params: dict = DEFAULT_SAWYER_ROBOT_CONFIG,
            env: MujocoEnv = None,
            is_hardware: bool = False,
    ):
        super(SawyerRobot, self).__init__(env=env)

        # Load config from params
        config = self.config = SawyerRobotConfig(sim=self._sim, **config_params)

        self._is_hardware = is_hardware
        self.listener = None
        self.commander = None
        if is_hardware:
            self.listener = SawyerListener()
            self.commander = SawyerCommander()

        self.reset_state = SawyerRobotState(
            pos=config.mocap_range[:3, :].mean(axis=1),
            quat=euler2quat(config.mocap_range[3:, :].mean(axis=1)),
            arm_qpos=env.init_qpos[:7],
            arm_qvel=np.zeros(7),
        )
        self.wrist_act = MocapActor(
            self._env,
            config.mocap_name,
            config.mocap_range,
            config.mocap_velocity_lim,
            config.control_mode
        )

    @property
    def is_hardware(self):
        return self._is_hardware

    def get_obs_dict(self):
        state = self.get_state()
        return collections.OrderedDict((
            ("mocap_pos", state.pos.copy()),
            ("mocap_quat", state.quat.copy()),
            ("mocap_euler", quat2euler(state.quat)),
            # TODO(justinvyu): These are not currently populated by hardware
            ("sawyer_arm_qpos", state.arm_qpos.copy()),
            ("sawyer_arm_qvel", state.arm_qvel.copy()),
        ))

    def get_state(self) -> SawyerRobotState:
        if self.is_hardware:
            state = self.listener.get_sawyer_state()
        else:
            if self._env.initializing:
                return SawyerRobotState(
                    pos=np.zeros(3),
                    quat=np.ones(4),
                    arm_qpos=np.zeros(len(self.config.qpos_indices)),
                    arm_qvel=np.zeros(len(self.config.qvel_indices))
                )
            state = SawyerRobotState(
                pos=self._sim.data.get_mocap_pos(self.config.mocap_name).copy(),
                quat=self._sim.data.get_mocap_quat(self.config.mocap_name).copy(),
                arm_qpos=self._sim.data.qpos[self.config.qpos_indices].copy(),
                arm_qvel=self._sim.data.qvel[self.config.qvel_indices].copy(),
            )
        return state

    def set_state(self, state: SawyerRobotState, command_angles=False):
        if self.is_hardware:
            self.commander.send_command(state, command_angles)
        else:
            if command_angles:
                if state.arm_qpos is not None:
                    self._sim.data.qpos[self.config.qpos_indices] = state.arm_qpos.copy()
                if state.arm_qvel is not None:
                    self._sim.data.qvel[self.config.qvel_indices] = state.arm_qvel.copy()
            else:
                # Clip the state to mocap limits
                sawyer_state = np.clip(
                    np.concatenate([state.pos, quat2euler(state.quat)]),
                    a_min=self.config.mocap_range[:, 0],
                    a_max=self.config.mocap_range[:, 1]
                )
                self._sim.data.set_mocap_pos(
                    self.config.mocap_name, sawyer_state[:3])
                self._sim.data.set_mocap_quat(
                    self.config.mocap_name, euler2quat(sawyer_state[3:]))

    def step(self, action: np.ndarray):
        assert len(action) == 6

        if self.is_hardware:
            action = np.clip(action, -1, 1)
            if self.config.control_mode == "pos":
                mid_sawyer = self.config.mocap_range.mean(axis=1)
                rng_sawyer = 0.5 * (
                    self.config.mocap_range[:, 1] - self.config.mocap_range[:, 0])
                action = mid_sawyer + action * rng_sawyer
            elif self.config.control_mode == "deltapos":
                state = self.get_state()
                xyz_euler = np.concatenate([state.pos, quat2euler(state.quat)])
                action = xyz_euler + action * self.config.mocap_velocity_lim
            else:
                raise NotImplementedError

            action = np.clip(
                action,
                a_min=self.config.mocap_range[:, 0],
                a_max=self.config.mocap_range[:, 1]
            )
            pos, euler = action[:3], action[3:]
            quat = euler2quat(euler)
            state = SawyerRobotState(pos=pos, quat=quat)
            self.set_state(state)

            # Synchronization for sim.
            # TODO: Get this right
            # offset = np.array([0.32635957, 0.22896217, 0.9134546])
            # self._sim.data.set_mocap_pos(self.config.mocap_name, state.pos + offset)
            # for _ in range(10):
            #     self._sim.forward()
            #     self._sim.step()

        else:
            self.wrist_act.update(action)

    def reset(self, command_angles=True):
        if self.is_hardware:
            if command_angles:
                reset_state = np.array([
                    3.20340419, -1.09228516, -0.10126562,
                    1.94024706, 0.14490137, -0.85471582, -0.26115918
                ])
                state = SawyerRobotState(arm_qpos=reset_state)
            else:
                assert False, "use angles to reset"
                state = self.reset_state
            self.set_state(state, command_angles=command_angles)
            # time.sleep(1)
            # self.set_state(self.reset_state)
        else:
            state = self.reset_state
            self.set_state(state, command_angles=command_angles)

        if self.is_hardware:
            time.sleep(6)

