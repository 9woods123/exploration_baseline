#include "raycast_static_map/object_tracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace raycast_static_map {
namespace {

using StateMatrix = Eigen::Matrix<double, 6, 6>;

// State covariance for a freshly created track: position uncertainty equals the
// measurement noise, velocity is essentially unknown.
StateMatrix initialCovariance(const double position_variance,
                              const double velocity_variance) {
  StateMatrix covariance = StateMatrix::Zero();
  covariance.block<3, 3>(0, 0) = position_variance * Eigen::Matrix3d::Identity();
  covariance.block<3, 3>(3, 3) = velocity_variance * Eigen::Matrix3d::Identity();
  return covariance;
}

// Discrete white-noise-acceleration process noise for a constant-velocity model.
StateMatrix processNoise(const double dt, const double acceleration_variance) {
  StateMatrix noise = StateMatrix::Zero();
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  noise.block<3, 3>(0, 0) = (dt3 / 3.0) * acceleration_variance * identity;
  noise.block<3, 3>(0, 3) = (dt2 / 2.0) * acceleration_variance * identity;
  noise.block<3, 3>(3, 0) = (dt2 / 2.0) * acceleration_variance * identity;
  noise.block<3, 3>(3, 3) = dt * acceleration_variance * identity;
  return noise;
}

}  // namespace

ObjectTracker::ObjectTracker(const TrackerConfig& config) : config_(config) {
  if (config_.max_association_distance <= 0.0) {
    throw std::invalid_argument("max_association_distance must be positive");
  }
  if (config_.min_track_duration < 1) {
    throw std::invalid_argument("min_track_duration must be at least 1");
  }
  if (config_.max_misses < 0) {
    throw std::invalid_argument("max_misses must be non-negative");
  }
  if (config_.kalman_process_noise <= 0.0) {
    throw std::invalid_argument("kalman_process_noise must be positive");
  }
  if (config_.kalman_measurement_noise <= 0.0) {
    throw std::invalid_argument("kalman_measurement_noise must be positive");
  }
  if (config_.velocity_horizon < 0.0) {
    throw std::invalid_argument("velocity_horizon must be non-negative");
  }
  if (config_.max_velocity <= 0.0 || config_.min_velocity < 0.0 ||
      config_.min_velocity > config_.max_velocity) {
    throw std::invalid_argument("invalid velocity clamp configuration");
  }
  if (config_.max_tracks < 1) {
    throw std::invalid_argument("max_tracks must be at least 1");
  }
}

void ObjectTracker::reset() {
  tracks_.clear();
  next_id_ = 0;
}

void ObjectTracker::predict(Track* track, const double stamp) const {
  const double dt = stamp - track->stamp;
  if (dt <= 1e-6) {
    track->stamp = stamp;
    return;
  }
  // Constant-velocity transition F = [[I, dt I], [0, I]].
  StateMatrix transition = StateMatrix::Identity();
  transition.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
  const double acceleration_variance =
      config_.kalman_process_noise * config_.kalman_process_noise;
  track->state = transition * track->state;
  track->covariance =
      transition * track->covariance * transition.transpose() +
      processNoise(dt, acceleration_variance);
  track->stamp = stamp;
}

void ObjectTracker::correct(Track* track,
                            const Eigen::Vector3d& measurement) const {
  // Measurement model z = H x with H = [I 0] and R = measurement_variance * I.
  const double measurement_variance =
      config_.kalman_measurement_noise * config_.kalman_measurement_noise;
  const Eigen::Matrix3d position_covariance = track->covariance.block<3, 3>(0, 0);
  const Eigen::Matrix3d innovation_covariance =
      position_covariance + measurement_variance * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d innovation_covariance_inverse =
      innovation_covariance.inverse();
  const Eigen::Vector3d innovation = measurement - track->state.head<3>();

  // Kalman gain K = P H^T S^-1, split into the position and velocity blocks.
  const Eigen::Matrix3d gain_position =
      position_covariance * innovation_covariance_inverse;
  const Eigen::Matrix3d gain_velocity =
      track->covariance.block<3, 3>(3, 0) * innovation_covariance_inverse;

  track->state.head<3>() += gain_position * innovation;
  track->state.tail<3>() += gain_velocity * innovation;

  // Covariance update P = (I - K H) P. H P is exactly the top three rows of P,
  // and the 6x3 gain is stacked from its position and velocity blocks.
  Eigen::Matrix<double, 6, 3> gain;
  gain.topRows<3>() = gain_position;
  gain.bottomRows<3>() = gain_velocity;
  track->covariance -= gain * track->covariance.topRows<3>();

  // Keep the covariance symmetric to avoid drift from floating point error.
  track->covariance =
      0.5 * (track->covariance + track->covariance.transpose());

  // Clamp the estimated velocity and suppress stationary jitter.
  Eigen::Vector3d velocity = track->state.tail<3>();
  const double speed = velocity.norm();
  if (speed > config_.max_velocity) {
    velocity *= config_.max_velocity / speed;
  } else if (speed < config_.min_velocity) {
    velocity.setZero();
  }
  track->state.tail<3>() = velocity;
}

std::vector<TrackedObject> ObjectTracker::update(
    const std::vector<DynamicCluster>& clusters, const double stamp) {
  // Predict all existing tracks to the measurement time so gating uses the
  // Kalman predictions.
  for (Track& track : tracks_) {
    predict(&track, stamp);
  }

  const std::size_t num_tracks = tracks_.size();
  const std::size_t num_clusters = clusters.size();
  std::vector<char> track_matched(num_tracks, 0);
  std::vector<char> cluster_matched(num_clusters, 0);

  // Build all gated track/cluster pairs and match greedily from closest to
  // farthest. This is O(n*m) per frame, which is fine for the object counts of
  // a single LiDAR scan.
  struct Pair {
    double distance;
    std::size_t track;
    std::size_t cluster;
  };
  std::vector<Pair> pairs;
  pairs.reserve(num_tracks * num_clusters);
  for (std::size_t t = 0; t < num_tracks; ++t) {
    const Eigen::Vector3d predicted = tracks_[t].state.head<3>();
    for (std::size_t c = 0; c < num_clusters; ++c) {
      const double distance = (predicted - clusters[c].centroid).norm();
      if (distance <= config_.max_association_distance) {
        pairs.push_back({distance, t, c});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.distance < b.distance; });

  for (const Pair& pair : pairs) {
    if (track_matched[pair.track] || cluster_matched[pair.cluster]) continue;
    track_matched[pair.track] = 1;
    cluster_matched[pair.cluster] = 1;

    Track& track = tracks_[pair.track];
    const DynamicCluster& cluster = clusters[pair.cluster];
    correct(&track, cluster.centroid);
    track.min_corner = cluster.min_corner;
    track.max_corner = cluster.max_corner;
    track.point_count = cluster.point_count;
    ++track.age;
    track.misses = 0;
    if (track.age >= config_.min_track_duration) track.confirmed = true;
  }

  // Age unmatched tracks; drop them after too many consecutive misses.
  for (std::size_t t = 0; t < num_tracks; ++t) {
    if (!track_matched[t]) ++tracks_[t].misses;
  }

  // Spawn new tracks from unmatched clusters, up to max_tracks. New tracks start
  // unconfirmed with zero velocity (a single observation carries no motion
  // information); their position uncertainty is the measurement variance.
  const double measurement_variance =
      config_.kalman_measurement_noise * config_.kalman_measurement_noise;
  const double velocity_variance = config_.max_velocity * config_.max_velocity;
  for (std::size_t c = 0; c < num_clusters; ++c) {
    if (cluster_matched[c]) continue;
    if (static_cast<int>(tracks_.size()) >= config_.max_tracks) break;
    Track track;
    track.id = next_id_++;
    track.state.head<3>() = clusters[c].centroid;
    track.covariance = initialCovariance(measurement_variance, velocity_variance);
    track.min_corner = clusters[c].min_corner;
    track.max_corner = clusters[c].max_corner;
    track.stamp = stamp;
    track.age = 1;
    track.misses = 0;
    track.point_count = clusters[c].point_count;
    track.confirmed = config_.min_track_duration <= 1;
    tracks_.push_back(track);
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& track) {
                                 return track.misses > config_.max_misses;
                               }),
                tracks_.end());

  std::vector<TrackedObject> objects;
  objects.reserve(tracks_.size());
  for (const Track& track : tracks_) {
    TrackedObject object;
    object.id = track.id;
    object.confirmed = track.confirmed;
    object.centroid = track.state.head<3>();
    object.velocity = track.state.tail<3>();
    object.min_corner = track.min_corner;
    object.max_corner = track.max_corner;
    object.predicted_position =
        object.centroid + object.velocity * config_.velocity_horizon;
    object.age = track.age;
    object.misses = track.misses;
    object.speed = object.velocity.norm();
    object.point_count = track.point_count;
    objects.push_back(object);
  }
  return objects;
}

}  // namespace raycast_static_map

