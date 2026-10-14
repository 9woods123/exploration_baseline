#!/usr/bin/env python3
"""Convert OmniPlanner PCI trajectories to uav_simulator position commands."""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from tf.transformations import euler_from_quaternion
from trajectory_msgs.msg import MultiDOFJointTrajectory


def _yaw(transform):
    q = transform.rotation
    return euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


def _angle_lerp(a, b, ratio):
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + ratio * delta


class TrajectoryAdapter:
    def __init__(self):
        self._lock = threading.Lock()
        self._trajectory = None
        self._start = rospy.Time(0)
        self._trajectory_id = 0
        topic_in = rospy.get_param("~trajectory_topic", "/command/trajectory")
        topic_out = rospy.get_param("~position_command_topic", "/position_cmd")
        odometry_topic = rospy.get_param("~odometry_topic", "/sim/odom")
        self._visualization_frame = rospy.get_param(
            "~visualization_frame", "world")
        self._executed_path_min_distance = rospy.get_param(
            "~executed_path_min_distance", 0.05)
        self._executed_path = Path()
        self._executed_path.header.frame_id = self._visualization_frame
        rate = rospy.get_param("~rate", 50.0)
        self._publisher = rospy.Publisher(topic_out, PositionCommand, queue_size=10)
        self._planned_path_publisher = rospy.Publisher(
            "~planned_trajectory", Path, queue_size=1, latch=True)
        self._executed_path_publisher = rospy.Publisher(
            "~executed_trajectory", Path, queue_size=1, latch=True)
        self._subscriber = rospy.Subscriber(
            topic_in, MultiDOFJointTrajectory, self._trajectory_callback,
            queue_size=1, tcp_nodelay=True)
        self._odometry_subscriber = rospy.Subscriber(
            odometry_topic, Odometry, self._odometry_callback,
            queue_size=20, tcp_nodelay=True)
        self._timer = rospy.Timer(rospy.Duration(1.0 / rate), self._timer_callback)

    def _trajectory_callback(self, msg):
        if not msg.points:
            return
        now = rospy.Time.now()
        planned_path = Path()
        planned_path.header.stamp = now
        planned_path.header.frame_id = (
            msg.header.frame_id.lstrip("/") if msg.header.frame_id
            else self._visualization_frame)
        for point in msg.points:
            if not point.transforms:
                continue
            transform = point.transforms[0]
            pose = PoseStamped()
            pose.header = planned_path.header
            pose.pose.position.x = transform.translation.x
            pose.pose.position.y = transform.translation.y
            pose.pose.position.z = transform.translation.z
            pose.pose.orientation = transform.rotation
            planned_path.poses.append(pose)
        with self._lock:
            self._trajectory_id += 1
            self._trajectory = msg
            self._start = now
            trajectory_id = self._trajectory_id
        self._planned_path_publisher.publish(planned_path)
        rospy.loginfo("Received trajectory %d (%d points)",
                      trajectory_id, len(msg.points))

    def _odometry_callback(self, msg):
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self._visualization_frame
        pose.pose = msg.pose.pose

        with self._lock:
            if self._executed_path.poses:
                previous = self._executed_path.poses[-1].pose.position
                current = pose.pose.position
                distance = math.sqrt(
                    (current.x - previous.x) ** 2
                    + (current.y - previous.y) ** 2
                    + (current.z - previous.z) ** 2)
                if distance < self._executed_path_min_distance:
                    return
            self._executed_path.header.stamp = rospy.Time.now()
            self._executed_path.poses.append(pose)
            executed_path = self._executed_path

        self._executed_path_publisher.publish(executed_path)

    @staticmethod
    def _vector(points, field, index):
        values = getattr(points, field)
        if not values:
            return (0.0, 0.0, 0.0)
        value = values[index if index < len(values) else 0]
        return (value.linear.x, value.linear.y, value.linear.z)

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self._lock:
            trajectory = self._trajectory
            start = self._start
            trajectory_id = self._trajectory_id
        if trajectory is None:
            return

        elapsed = max(0.0, (now - start).to_sec())
        points = trajectory.points
        upper = len(points) - 1
        for index, point in enumerate(points):
            if point.time_from_start.to_sec() >= elapsed:
                upper = index
                break
        lower = max(0, upper - 1)
        t0 = points[lower].time_from_start.to_sec()
        t1 = points[upper].time_from_start.to_sec()
        ratio = 1.0 if upper == lower or t1 <= t0 else min(1.0, max(0.0, (elapsed - t0) / (t1 - t0)))

        if not points[lower].transforms or not points[upper].transforms:
            return
        transform0 = points[lower].transforms[0]
        transform1 = points[upper].transforms[0]
        command = PositionCommand()
        command.header.stamp = rospy.Time.now()
        command.header.frame_id = self._visualization_frame
        command.position.x = transform0.translation.x + ratio * (transform1.translation.x - transform0.translation.x)
        command.position.y = transform0.translation.y + ratio * (transform1.translation.y - transform0.translation.y)
        command.position.z = transform0.translation.z + ratio * (transform1.translation.z - transform0.translation.z)

        velocity0 = self._vector(points[lower], "velocities", 0)
        velocity1 = self._vector(points[upper], "velocities", 0)
        acceleration0 = self._vector(points[lower], "accelerations", 0)
        acceleration1 = self._vector(points[upper], "accelerations", 0)
        command.velocity.x = velocity0[0] + ratio * (velocity1[0] - velocity0[0])
        command.velocity.y = velocity0[1] + ratio * (velocity1[1] - velocity0[1])
        command.velocity.z = velocity0[2] + ratio * (velocity1[2] - velocity0[2])
        command.acceleration.x = acceleration0[0] + ratio * (acceleration1[0] - acceleration0[0])
        command.acceleration.y = acceleration0[1] + ratio * (acceleration1[1] - acceleration0[1])
        command.acceleration.z = acceleration0[2] + ratio * (acceleration1[2] - acceleration0[2])
        command.yaw = _angle_lerp(_yaw(transform0), _yaw(transform1), ratio)
        command.trajectory_id = trajectory_id
        command.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        self._publisher.publish(command)


if __name__ == "__main__":
    rospy.init_node("trajectory_to_position_cmd")
    TrajectoryAdapter()
    rospy.spin()
