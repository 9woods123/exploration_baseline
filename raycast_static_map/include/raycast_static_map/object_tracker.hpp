#pragma once

#include <Eigen/Core>

#include <vector>

#include "raycast_static_map/dynamic_object.hpp"

namespace raycast_static_map {

struct TrackerConfig {
  // Gating distance for centroid-based data association [m]. Clusters farther
  // than this from any track prediction are treated as new objects.
  double max_association_distance = 1.0;
  // Number of consecutive frames a track must be observed before it is reported
  // as a confirmed object (matches Dynablox's min_track_duration idea).
  int min_track_duration = 2;
  // Consecutive frames without an association before a track is dropped.
  int max_misses = 5;
  // Kalman filter process noise, expressed as the standard deviation of the
  // unmodelled acceleration [m/s^2]. Larger = trust the measurements more.
  double kalman_process_noise = 2.0;
  // Kalman filter measurement noise, expressed as the standard deviation of the
  // measured centroid position [m]. Larger = trust the model more.
  double kalman_measurement_noise = 0.5;
  // Prediction horizon for the reported predicted position [s].
  double velocity_horizon = 1.0;
  // Hard clamp on the estimated speed [m/s].
  double max_velocity = 20.0;
  // Speeds below this are snapped to zero to suppress stationary jitter [m/s].
  double min_velocity = 0.10;
  // Cap on the number of simultaneously tracked objects.
  int max_tracks = 500;
};

struct TrackedObject {
  int id = -1;
  bool confirmed = false;
  // Filtered (posterior) centroid position.
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  // Filtered world-frame velocity [m/s].
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d min_corner = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_corner = Eigen::Vector3d::Zero();
  // centroid + velocity * velocity_horizon.
  Eigen::Vector3d predicted_position = Eigen::Vector3d::Zero();
  int age = 0;     // frames observed since creation.
  int misses = 0;  // consecutive frames without association.
  double speed = 0.0;
  std::size_t point_count = 0;
};

// Constant-velocity multi-object tracker. Clusters are associated to existing
// tracks by greedy nearest-neighbour matching against the Kalman prediction, and
// each track runs a linear Kalman filter with a constant-velocity motion model
// (state = [position, velocity], measurement = position) to estimate a smoothed
// position and velocity and to extrapolate the next position.
class ObjectTracker {
 public:
  explicit ObjectTracker(const TrackerConfig& config);

  // Update the tracker with the clusters of the current frame. `stamp` is the
  // frame time in seconds and is used as the Kalman time delta.
  std::vector<TrackedObject> update(const std::vector<DynamicCluster>& clusters,
                                    double stamp);
  void reset();

  const TrackerConfig& config() const { return config_; }
  std::size_t trackCount() const { return tracks_.size(); }

 private:
  struct Track {
    int id = -1;
    // Kalman state estimate x = [px, py, pz, vx, vy, vz].
    Eigen::Matrix<double, 6, 1> state = Eigen::Matrix<double, 6, 1>::Zero();
    // State covariance P.
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Vector3d min_corner = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_corner = Eigen::Vector3d::Zero();
    double stamp = 0.0;
    int age = 0;
    int misses = 0;
    bool confirmed = false;
    std::size_t point_count = 0;
  };

  // Constant-velocity Kalman prediction to time `stamp`.
  void predict(Track* track, double stamp) const;
  // Kalman measurement update with the measured centroid position.
  void correct(Track* track, const Eigen::Vector3d& measurement) const;

  TrackerConfig config_;
  std::vector<Track> tracks_;
  int next_id_ = 0;
};

}  // namespace raycast_static_map
