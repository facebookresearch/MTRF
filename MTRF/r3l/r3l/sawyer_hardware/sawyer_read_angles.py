# Copyright (c) Facebook, Inc. and its affiliates
# Copyright (c) MTRF authors

#!/usr/bin/env python
import os
# os.system('chmod +x sawyer_read_angles.py')
import rospy
from std_msgs.msg import String
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from rospy_message_converter import message_converter
import intera_interface
import numpy as np

from geometry_msgs.msg import Pose

def talker():
    pub_pose = rospy.Publisher('get_pose', Pose, queue_size=10)
    pub_angles = rospy.Publisher('get_angles', Floats, queue_size=10)
    pub_velocities = rospy.Publisher('get_angle_velocities', Floats, queue_size=10)

    # pub = rospy.Publisher('chatter', Pose, queue_size=11)
    rospy.init_node('Sawyer_reader')
    limb = intera_interface.Limb('right')
    rate = rospy.Rate(13) # 10hz
    while not rospy.is_shutdown():
        angles = limb.joint_angles()
        fk_resp = limb.fk_request(angles)
        pos = fk_resp.pose_stamp[0].pose
        velocities = limb.joint_velocities()
        rospy.loginfo(pos)
	rospy.loginfo(list(angles.values()))
        # rospy.loginfo(velocities)
        pub_pose.publish(pos)
        pub_angles.publish(Floats(list(angles.values())))
        pub_velocities.publish(Floats(list(velocities.values())))

        rate.sleep()

if __name__ == '__main__':
    # run with python2, intera env
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
