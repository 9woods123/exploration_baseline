#pragma once

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <memory>
#include <string>
#include <vector>

#include "map_manager/map_manager.h"

class TrajectoryOptimizer {
 public:
  TrajectoryOptimizer(const ros::NodeHandle& private_nh,
                      MapManager* map_manager);

  bool enabled() const { return enabled_; }
  bool optimize(const std::vector<geometry_msgs::Pose>& path,
                const geometry_msgs::Pose& initial_pose,
                const geometry_msgs::Twist& initial_twist,
                const ros::Time& start_time,
                const std::string& frame_id,
                trajectory_msgs::MultiDOFJointTrajectory* trajectory) const;

 private:
  struct ObjectiveData;

  static double objective(unsigned count, const double* x,
                          double* gradient, void* data);
  std::vector<geometry_msgs::Pose> densify(
      const std::vector<geometry_msgs::Pose>& path,
      const geometry_msgs::Pose& initial_pose) const;
  bool buildTrajectory(const std::vector<Eigen::Vector3d>& positions,
                       const std::vector<double>& yaws,
                       const geometry_msgs::Twist& initial_twist,
                       const ros::Time& start_time,
                       const std::string& frame_id,
                       trajectory_msgs::MultiDOFJointTrajectory* trajectory) const;

  MapManager* map_manager_;
  bool enabled_ = true;
  bool unknown_is_obstacle_ = false;
  double reference_spacing_ = 0.35;
  double max_deviation_ = 0.8;
  double safety_distance_ = 0.7;
  double smooth_weight_ = 8.0;
  double reference_weight_ = 2.0;
  double obstacle_weight_ = 30.0;
  double max_solve_time_ = 0.8;
  int max_evaluations_ = 160;
  double v_max_ = 1.8;
  double a_max_ = 4.0;
  double yaw_rate_max_ = 0.8;
};
