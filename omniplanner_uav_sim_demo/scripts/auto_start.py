#!/usr/bin/env python3
"""Wait for the demo pipeline, then start autonomous exploration."""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger


if __name__ == "__main__":
    rospy.init_node("omniplanner_demo_auto_start")
    timeout = rospy.get_param("~timeout", 90.0)
    delay = rospy.get_param("~mapping_delay", 5.0)
    service_name = "/planner_control_interface/std_srvs/automatic_planning"
    rospy.loginfo("Waiting for odometry and simulated lidar point cloud")
    rospy.wait_for_message("/sim/odom", Odometry, timeout=timeout)
    rospy.wait_for_message("/lidar/points", PointCloud2, timeout=timeout)
    rospy.wait_for_service(service_name, timeout=timeout)
    rospy.sleep(delay)
    response = rospy.ServiceProxy(service_name, Trigger)()
    if response.success:
        rospy.loginfo("OmniPlanner autonomous exploration started: %s", response.message)
    else:
        rospy.logerr("OmniPlanner start request failed: %s", response.message)
