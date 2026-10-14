#!/usr/bin/env python3
"""Broadcast the simulated UAV sensor pose for point-cloud ray integration."""

import rospy
import tf
from nav_msgs.msg import Odometry


class OdometryToTf:
    def __init__(self):
        self._broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber("/sim/odom", Odometry, self._callback, queue_size=10)

    def _callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self._broadcaster.sendTransform(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w),
            msg.header.stamp,
            "uav_sensor",
            "world",
        )


if __name__ == "__main__":
    rospy.init_node("odometry_to_tf")
    OdometryToTf()
    rospy.spin()
