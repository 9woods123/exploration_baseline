#pragma once

#include <Eigen/Core>

#include <cstddef>

namespace raycast_static_map {

// One dynamic cluster detected in a single frame, produced by the mapper and
// consumed by the tracker. The bounding box is axis-aligned in the map frame.
struct DynamicCluster {
  // Mean position of the cluster points [m]. Used for data association.
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  // Axis-aligned bounding box corners [m].
  Eigen::Vector3d min_corner = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_corner = Eigen::Vector3d::Zero();
  std::size_t point_count = 0;
};

inline Eigen::Vector3d clusterBoxCenter(const DynamicCluster& cluster) {
  return 0.5 * (cluster.min_corner + cluster.max_corner);
}

inline Eigen::Vector3d clusterBoxSize(const DynamicCluster& cluster) {
  return cluster.max_corner - cluster.min_corner;
}

}  // namespace raycast_static_map
