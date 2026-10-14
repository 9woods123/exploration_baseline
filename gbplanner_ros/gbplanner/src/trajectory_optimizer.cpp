#include "gbplanner/trajectory_optimizer.h"

#include <mav_msgs/conversions.h>
#include <nlopt/nlopt.h>
#include <tf/tf.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double shortestYawDelta(double from, double to) {
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

geometry_msgs::Pose interpolatePose(const geometry_msgs::Pose& a,
                                    const geometry_msgs::Pose& b,
                                    double ratio) {
  geometry_msgs::Pose result;
  result.position.x = a.position.x + ratio * (b.position.x - a.position.x);
  result.position.y = a.position.y + ratio * (b.position.y - a.position.y);
  result.position.z = a.position.z + ratio * (b.position.z - a.position.z);
  const double yaw = tf::getYaw(a.orientation) +
      ratio * shortestYawDelta(tf::getYaw(a.orientation),
                               tf::getYaw(b.orientation));
  result.orientation = tf::createQuaternionMsgFromYaw(yaw);
  return result;
}

Eigen::Vector3d positionOf(const geometry_msgs::Pose& pose) {
  return Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
}

}  // namespace

struct TrajectoryOptimizer::ObjectiveData {
  std::vector<Eigen::Vector3d> reference;
  std::shared_ptr<const EsdfSnapshot> esdf;
  double smooth_weight;
  double reference_weight;
  double obstacle_weight;
  double safety_distance;
  bool unknown_is_obstacle;
  std::size_t evaluations = 0;
};

TrajectoryOptimizer::TrajectoryOptimizer(const ros::NodeHandle& private_nh,
                                         MapManager* map_manager)
    : map_manager_(map_manager) {
  private_nh.param("trajectory_optimization/enable", enabled_, enabled_);
  private_nh.param("trajectory_optimization/reference_spacing",
                   reference_spacing_, reference_spacing_);
  private_nh.param("trajectory_optimization/max_deviation", max_deviation_,
                   max_deviation_);
  private_nh.param("trajectory_optimization/safety_distance", safety_distance_,
                   safety_distance_);
  private_nh.param("trajectory_optimization/smooth_weight", smooth_weight_,
                   smooth_weight_);
  private_nh.param("trajectory_optimization/reference_weight",
                   reference_weight_, reference_weight_);
  private_nh.param("trajectory_optimization/obstacle_weight",
                   obstacle_weight_, obstacle_weight_);
  private_nh.param("trajectory_optimization/max_solve_time", max_solve_time_,
                   max_solve_time_);
  private_nh.param("trajectory_optimization/max_evaluations", max_evaluations_,
                   max_evaluations_);
  private_nh.param("trajectory_optimization/unknown_is_obstacle",
                   unknown_is_obstacle_, unknown_is_obstacle_);
  private_nh.param("RobotDynamics/v_max", v_max_, v_max_);
  private_nh.param("RobotDynamics/a_max", a_max_, a_max_);
  private_nh.param("RobotDynamics/yaw_rate_max", yaw_rate_max_, yaw_rate_max_);
}

std::vector<geometry_msgs::Pose> TrajectoryOptimizer::densify(
    const std::vector<geometry_msgs::Pose>& path,
    const geometry_msgs::Pose& initial_pose) const {
  std::vector<geometry_msgs::Pose> source = path;
  if (source.empty()) return source;
  source.front() = initial_pose;

  std::vector<geometry_msgs::Pose> dense;
  dense.push_back(source.front());
  const double spacing = std::max(0.05, reference_spacing_);
  for (std::size_t i = 0; i + 1 < source.size(); ++i) {
    const double length = (positionOf(source[i + 1]) - positionOf(source[i])).norm();
    const int steps = std::max(1, static_cast<int>(std::ceil(length / spacing)));
    for (int step = 1; step <= steps; ++step) {
      dense.push_back(interpolatePose(source[i], source[i + 1],
                                      static_cast<double>(step) / steps));
    }
  }
  return dense;
}

double TrajectoryOptimizer::objective(unsigned count, const double* x,
                                      double*, void* raw_data) {
  auto& data = *static_cast<ObjectiveData*>(raw_data);
  if (data.evaluations++ == 0) {
    ROS_INFO("Trajectory optimizer entered the first objective evaluation (%zu values).",
             static_cast<std::size_t>(count));
  }
  std::vector<Eigen::Vector3d> points = data.reference;
  for (std::size_t i = 1; i + 1 < points.size(); ++i) {
    points[i] = Eigen::Vector3d(x[3 * (i - 1)], x[3 * (i - 1) + 1],
                               x[3 * (i - 1) + 2]);
  }

  double cost = 0.0;
  for (std::size_t i = 1; i + 1 < points.size(); ++i) {
    cost += data.reference_weight *
            (points[i] - data.reference[i]).squaredNorm();
    const Eigen::Vector3d second = points[i - 1] - 2.0 * points[i] + points[i + 1];
    cost += data.smooth_weight * second.squaredNorm();

    if (data.esdf && data.esdf->valid()) {
      const double distance = data.esdf->getPointDistance(points[i]);
      const double effective_distance =
          distance < 0.0 && data.unknown_is_obstacle ? 0.0 : distance;
      if (effective_distance >= 0.0 &&
          effective_distance < data.safety_distance) {
        const double violation = data.safety_distance - effective_distance;
        cost += data.obstacle_weight * violation * violation;
      }
    }
  }
  // Penalize segment midpoints as well as optimized knots so a segment cannot
  // cut through a thin obstacle between two collision-free samples.
  if (data.esdf && data.esdf->valid()) {
    for (std::size_t i = 1; i < points.size(); ++i) {
      const Eigen::Vector3d midpoint = 0.5 * (points[i - 1] + points[i]);
      const double distance = data.esdf->getPointDistance(midpoint);
      const double effective_distance =
          distance < 0.0 && data.unknown_is_obstacle ? 0.0 : distance;
      if (effective_distance >= 0.0 &&
          effective_distance < data.safety_distance) {
        const double violation = data.safety_distance - effective_distance;
        cost += data.obstacle_weight * violation * violation;
      }
    }
  }
  return cost;
}

bool TrajectoryOptimizer::optimize(
    const std::vector<geometry_msgs::Pose>& path,
    const geometry_msgs::Pose& initial_pose,
    const geometry_msgs::Twist& initial_twist,
    const ros::Time& start_time, const std::string& frame_id,
    trajectory_msgs::MultiDOFJointTrajectory* trajectory) const {
  if (!trajectory || path.size() < 2) return false;
  const auto dense = densify(path, initial_pose);
  if (dense.size() < 2) return false;
  ROS_INFO("Trajectory optimizer densified %zu waypoints to %zu samples.",
           path.size(), dense.size());

  std::vector<Eigen::Vector3d> reference;
  std::vector<double> yaws;
  reference.reserve(dense.size());
  yaws.reserve(dense.size());
  for (const auto& pose : dense) {
    reference.push_back(positionOf(pose));
    double yaw = tf::getYaw(pose.orientation);
    if (!yaws.empty()) yaw = yaws.back() + shortestYawDelta(yaws.back(), yaw);
    yaws.push_back(yaw);
  }

  std::vector<Eigen::Vector3d> optimized = reference;
  if (enabled_ && dense.size() > 2) {
    ROS_INFO("Trajectory optimizer acquiring ESDF snapshot.");
    ObjectiveData data{reference, map_manager_->getEsdfSnapshot(), smooth_weight_,
                       reference_weight_, obstacle_weight_, safety_distance_,
                       unknown_is_obstacle_, 0};
    if (!data.esdf || !data.esdf->valid()) {
      ROS_WARN("Trajectory optimization skipped: no ESDF snapshot is ready.");
      return false;
    }

    const std::size_t variables = 3 * (dense.size() - 2);
    ROS_INFO("Trajectory optimizer starting NLopt with %zu variables.", variables);
    std::vector<double> x(variables), lower(variables), upper(variables);
    for (std::size_t i = 1; i + 1 < dense.size(); ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        const std::size_t index = 3 * (i - 1) + axis;
        x[index] = reference[i][axis];
        lower[index] = x[index] - max_deviation_;
        upper[index] = x[index] + max_deviation_;
      }
    }

    // COBYLA has robust bounded, derivative-free behavior for this cost.
    nlopt_opt optimizer = nlopt_create(NLOPT_LN_COBYLA, variables);
    if (!optimizer) {
      ROS_ERROR("NLopt trajectory optimization failed to create a solver.");
      return false;
    }
    nlopt_set_lower_bounds(optimizer, lower.data());
    nlopt_set_upper_bounds(optimizer, upper.data());
    // The Noetic 2.1.21 package has a broken forwarding call in
    // nlopt_set_min_objective().  The preconditioned form with a null
    // preconditioner is exactly the same objective API and avoids it.
    nlopt_set_precond_min_objective(optimizer, &TrajectoryOptimizer::objective,
                                    nullptr, &data);
    nlopt_set_maxeval(optimizer, max_evaluations_);
    nlopt_set_maxtime(optimizer, max_solve_time_);
    nlopt_set_xtol_rel(optimizer, 1e-3);
    double minimum = std::numeric_limits<double>::infinity();
    const nlopt_result result = nlopt_optimize(optimizer, x.data(), &minimum);
    nlopt_destroy(optimizer);
    if (result < 0 && result != NLOPT_ROUNDOFF_LIMITED) {
      ROS_ERROR("NLopt trajectory optimization failed with result %d.",
                static_cast<int>(result));
      return false;
    }
    if (result == NLOPT_ROUNDOFF_LIMITED) {
      ROS_WARN("NLopt stopped at its roundoff limit; validating the best iterate.");
    }
    for (std::size_t i = 1; i + 1 < dense.size(); ++i) {
      optimized[i] = Eigen::Vector3d(x[3 * (i - 1)], x[3 * (i - 1) + 1],
                                     x[3 * (i - 1) + 2]);
    }

    for (std::size_t i = 0; i + 1 < optimized.size(); ++i) {
      const double segment_length = (optimized[i + 1] - optimized[i]).norm();
      const int checks = std::max(
          1, static_cast<int>(std::ceil(segment_length /
                                        std::max(0.05, 0.5 * reference_spacing_))));
      for (int check = 0; check <= checks; ++check) {
        const double ratio = static_cast<double>(check) / checks;
        const Eigen::Vector3d point =
            optimized[i] + ratio * (optimized[i + 1] - optimized[i]);
        const double distance = data.esdf->getPointDistance(point);
        if ((distance < 0.0 && unknown_is_obstacle_) ||
            (distance >= 0.0 && distance < safety_distance_)) {
          ROS_WARN("Optimized trajectory failed ESDF validation (distance %.3f).",
                   distance);
          return false;
        }
      }
    }
  }

  // Planning can take longer than a path's original request timestamp (for
  // example while the first ESDF is built).  A command trajectory must never
  // begin in the past: preserve a future request time, otherwise start now.
  const ros::Time execution_start = std::max(start_time, ros::Time::now());
  return buildTrajectory(optimized, yaws, initial_twist, execution_start,
                         frame_id, trajectory);
}

bool TrajectoryOptimizer::buildTrajectory(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<double>& yaws,
    const geometry_msgs::Twist& initial_twist,
    const ros::Time& start_time, const std::string& frame_id,
    trajectory_msgs::MultiDOFJointTrajectory* trajectory) const {
  if (positions.size() < 2 || positions.size() != yaws.size()) return false;
  trajectory->header.stamp = start_time;
  trajectory->header.frame_id = frame_id;
  trajectory->points.clear();
  trajectory->points.resize(positions.size());

  const std::size_t count = positions.size();
  std::vector<double> speed(count, v_max_);
  const Eigen::Vector3d initial_velocity(initial_twist.linear.x,
                                         initial_twist.linear.y,
                                         initial_twist.linear.z);
  const Eigen::Vector3d first_segment = positions[1] - positions[0];
  speed.front() = first_segment.norm() > 1e-9
      ? std::min(v_max_, std::max(0.0, initial_velocity.dot(
                                      first_segment.normalized())))
      : 0.0;
  for (std::size_t i = 1; i < count; ++i) {
    const double ds = (positions[i] - positions[i - 1]).norm();
    speed[i] = std::min(v_max_, std::sqrt(speed[i - 1] * speed[i - 1] +
                                         2.0 * a_max_ * ds));
  }
  speed.back() = 0.0;
  for (std::size_t i = count - 1; i > 0; --i) {
    const double ds = (positions[i] - positions[i - 1]).norm();
    speed[i - 1] = std::min(speed[i - 1],
                            std::sqrt(speed[i] * speed[i] + 2.0 * a_max_ * ds));
  }

  double elapsed = 0.0;
  std::vector<double> times(count, 0.0);
  for (std::size_t i = 1; i < count; ++i) {
    const double ds = (positions[i] - positions[i - 1]).norm();
    const double velocity_sum = speed[i - 1] + speed[i];
    double translation_time = velocity_sum > 1e-6
        ? 2.0 * ds / velocity_sum
        : 2.0 * std::sqrt(ds / std::max(1e-3, a_max_));
    const double yaw_time =
        std::abs(yaws[i] - yaws[i - 1]) / std::max(1e-3, yaw_rate_max_);
    elapsed += std::max({translation_time, yaw_time, 0.001});
    times[i] = elapsed;
  }

  for (std::size_t i = 0; i < count; ++i) {
    mav_msgs::EigenTrajectoryPoint point;
    point.position_W = positions[i];
    point.setFromYaw(std::atan2(std::sin(yaws[i]), std::cos(yaws[i])));
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    if (i + 1 < count && (positions[i + 1] - positions[i]).norm() > 1e-9)
      direction = (positions[i + 1] - positions[i]).normalized();
    else if (i > 0 && (positions[i] - positions[i - 1]).norm() > 1e-9)
      direction = (positions[i] - positions[i - 1]).normalized();
    point.velocity_W = i == 0 ? initial_velocity : direction * speed[i];
    trajectory_msgs::MultiDOFJointTrajectoryPoint message;
    mav_msgs::msgMultiDofJointTrajectoryPointFromEigen(point, &message);
    message.time_from_start = ros::Duration(times[i]);
    trajectory->points[i] = message;
  }
  return true;
}
