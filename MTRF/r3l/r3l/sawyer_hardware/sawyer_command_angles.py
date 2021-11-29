# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

#!/usr/bin/env python
import os
# os.system('chmod +x sawyer_command_angles.py')
import rospy
from std_msgs.msg import String
import intera_interface

from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray
import numpy as np

rospy.init_node('Sawyer_commander')
limb = intera_interface.Limb('right')
limb.set_joint_position_speed(0.1)  # How fast the Sawyer joint position controller moves
curr_angles = limb.joint_angles()
command_angles_bool = False #False = pose, True = joint angles

J0_MAX = 0.12
J0_MIN = -1.0

J4_MAX = 100
J4_MIN = -0.1

J2_MAX = 0.8

J5_MAX = -1.4
J5_MIN = -2

def get_pose(pose):
    global curr_angles
    global command_angles_bool
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', pose)
    #angles = limb.ik_request(pose)
    if command_angles_bool:
        curr_angles = {
            'right_j6': pose.position.x,'right_j5':pose.position.y, 'right_j4':pose.position.z,'right_j3':pose.orientation.x,'right_j2':pose.orientation.y, 'right_j1':pose.orientation.z,'right_j0':pose.orientation.w}
    else:
        angles = limb.ik_request(pose, joint_seed=curr_angles)
        curr_angles = angles.copy()

    right_j0 = curr_angles["right_j0"]
    right_j4 = curr_angles["right_j4"]
    right_j2 = curr_angles["right_j2"]
    if curr_angles['right_j0'] > J0_MAX:
        print("clipping J0: original=%s, new=%s" % (right_j0, J0_MAX))
        curr_angles['right_j0'] = J0_MAX
    if curr_angles['right_j0'] < J0_MIN:
        print("clipping J0: original=%s, new=%s" % (right_j0, J0_MIN))
        curr_angles['right_j0'] = J0_MIN

    if curr_angles['right_j0'] > -0.05 and curr_angles['right_j4'] < J4_MIN:
        print("clipping J4: original=%s, new=%s" % (right_j4, J4_MIN))
        curr_angles['right_j4'] = J4_MIN

    if curr_angles['right_j2'] > J2_MAX:
        print("clipping J2: original=%s, new=%s" % (right_j2, J2_MAX))
        curr_angles['right_j2'] = J2_MAX

    # if curr_angles['right_j5'] > J5_MAX:
    #     curr_angles['right_j5'] = J5_MAX
    #limb.move_to_joint_positions(angles)
    #limb.set_joint_positions(angles)
    #limb.set_joint_positions({'right_j6':pose.position.x,'right_j5':pose.position.y, 'right_j4':pose.position.z,'right_j3':pose.orientation.x,'right_j2':pose.orientation.y, 'right_j1':pose.orientation.z,'right_j0':pose.orientation.w})
    #print("I'm Going To: ")
    #print(limb.fk_request(angles).pose_stamp[0].pose)

def set_type(type_in):
    global command_angles_bool
    command_angles_bool = type_in.data

def listener():
    global curr_angles
    global command_angles_bool
    curr_angles_last = curr_angles
    rospy.Subscriber("set_angles", Pose, get_pose)
    rospy.Subscriber("command_type", Bool, set_type)
    #rospy.Subscriber("set_joint_angles", Float32MultiArray, get_pose)
    # spin() simply keeps python from exiting until this node is stopped
    print('running')
    while True:
        #print(curr_angles)
        if not np.allclose(curr_angles.values(), curr_angles_last.values()):
            print(limb.fk_request(curr_angles).pose_stamp[0].pose)
        limb.set_joint_positions(curr_angles)
        curr_angles_last = curr_angles.copy()
        rospy.sleep(0.01)
    rospy.spin()

if __name__ == '__main__':
    # run with python2, intera env
    listener()

