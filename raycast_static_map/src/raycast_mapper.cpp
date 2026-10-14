#include "raycast_static_map/raycast_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>

namespace raycast_static_map {
namespace {

double logit(double probability) {
  if (!(probability > 0.0 && probability < 1.0)) {
    throw std::invalid_argument("occupancy probabilities must be in (0, 1)");
  }
  return std::log(probability / (1.0 - probability));
}

}  // namespace

std::size_t VoxelKeyHash::operator()(const VoxelKey& key) const {
  std::size_t value = static_cast<std::size_t>(key.x) * 73856093u;
  value ^= static_cast<std::size_t>(key.y) * 19349663u;
  value ^= static_cast<std::size_t>(key.z) * 83492791u;
  return value;
}

RaycastMapper::RaycastMapper(const MapperConfig& config)
    : config_(config),
      hit_log_odds_(logit(config.prob_hit)),
      miss_log_odds_(logit(config.prob_miss)),
      min_log_odds_(logit(config.prob_min)),
      max_log_odds_(logit(config.prob_max)),
      occupied_log_odds_(logit(config.occupied_probability)) {
  if (config_.resolution <= 0.0 || config_.min_range < 0.0 ||
      config_.max_range <= config_.min_range || config_.esdf_max_distance <= 0.0 ||
      config_.esdf_update_every_n_frames <= 0) {
    throw std::invalid_argument("invalid resolution or range configuration");
  }
  if (config_.tsdf_truncation <= 0.0 || config_.tsdf_surface_band < 0.0 ||
      config_.tsdf_surface_band > config_.tsdf_truncation) {
    throw std::invalid_argument("invalid TSDF truncation or surface band configuration");
  }
  if (config_.tsdf_max_weight <= 0.0) {
    throw std::invalid_argument("tsdf_max_weight must be positive");
  }
  if (config_.cluster_connectivity != 6 &&
      config_.cluster_connectivity != 18 &&
      config_.cluster_connectivity != 26) {
    throw std::invalid_argument("cluster_connectivity must be 6, 18, or 26");
  }
  if (config_.ever_free_neighbor_connectivity != 6 &&
      config_.ever_free_neighbor_connectivity != 18 &&
      config_.ever_free_neighbor_connectivity != 26) {
    throw std::invalid_argument("ever_free_neighbor_connectivity must be 6, 18, or 26");
  }
  if (config_.ever_free_burn_in_period < 0 ||
      config_.ever_free_temporal_buffer_frames < 0 ||
      config_.ever_free_reset_occupied_frames < 1) {
    throw std::invalid_argument("invalid ever-free frame counts");
  }
  if (config_.ever_free_max_free_probability < 0.0 ||
      config_.ever_free_max_free_probability >= 1.0) {
    throw std::invalid_argument("ever_free_max_free_probability must be in [0, 1)");
  }
  if (config_.cluster_min_separation < 0.0 || config_.cluster_min_extent < 0.0 ||
      config_.cluster_max_extent < config_.cluster_min_extent) {
    throw std::invalid_argument("invalid cluster separation or extent configuration");
  }
}

VoxelKey RaycastMapper::key(const Eigen::Vector3d& point) const {
  return {static_cast<int>(std::floor(point.x() / config_.resolution)),
          static_cast<int>(std::floor(point.y() / config_.resolution)),
          static_cast<int>(std::floor(point.z() / config_.resolution))};
}

Eigen::Vector3d RaycastMapper::center(const VoxelKey& voxel) const {
  return config_.resolution *
         Eigen::Vector3d(voxel.x + 0.5, voxel.y + 0.5, voxel.z + 0.5);
}

bool RaycastMapper::occupied(const Cell& cell) const {
  if (config_.use_tsdf) {
    // The surface band around the TSDF zero crossing defines occupied space.
    // Voxels whose signed distance exceeds the truncation are never marked
    // observed, so free space stays free.
    return cell.tsdf_observed && cell.tsdf <= config_.tsdf_surface_band;
  }
  return cell.log_odds >= occupied_log_odds_;
}

double RaycastMapper::probability(const Cell& cell) const {
  return 1.0 / (1.0 + std::exp(-cell.log_odds));
}

std::vector<VoxelKey> RaycastMapper::neighborOffsets(int connectivity) {
  std::vector<VoxelKey> offsets;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        const int nonzero = (dx != 0) + (dy != 0) + (dz != 0);
        if (nonzero == 0 ||
            (connectivity == 6 && nonzero != 1) ||
            (connectivity == 18 && nonzero > 2)) {
          continue;
        }
        offsets.push_back({dx, dy, dz});
      }
    }
  }
  return offsets;
}

void RaycastMapper::traceRay(const Eigen::Vector3d& origin,
                             const Eigen::Vector3d& endpoint,
                             std::vector<VoxelKey>* traversed) const {
  traversed->clear();
  VoxelKey current = key(origin);
  const VoxelKey end = key(endpoint);
  if (current == end) return;

  const Eigen::Vector3d direction = endpoint - origin;
  int current_axis[3] = {current.x, current.y, current.z};
  int step[3];
  double t_max[3];
  double t_delta[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (direction[axis] > 0.0) {
      step[axis] = 1;
      const double boundary = (current_axis[axis] + 1) * config_.resolution;
      t_max[axis] = (boundary - origin[axis]) / direction[axis];
      t_delta[axis] = config_.resolution / direction[axis];
    } else if (direction[axis] < 0.0) {
      step[axis] = -1;
      const double boundary = current_axis[axis] * config_.resolution;
      t_max[axis] = (boundary - origin[axis]) / direction[axis];
      t_delta[axis] = -config_.resolution / direction[axis];
    } else {
      step[axis] = 0;
      t_max[axis] = std::numeric_limits<double>::infinity();
      t_delta[axis] = std::numeric_limits<double>::infinity();
    }
  }

  traversed->push_back(current);
  const int max_steps = std::abs(end.x - current.x) +
                        std::abs(end.y - current.y) +
                        std::abs(end.z - current.z) + 3;
  for (int count = 0; count < max_steps; ++count) {
    int axis = 0;
    if (t_max[1] < t_max[axis]) axis = 1;
    if (t_max[2] < t_max[axis]) axis = 2;
    current_axis[axis] += step[axis];
    t_max[axis] += t_delta[axis];
    current = {current_axis[0], current_axis[1], current_axis[2]};
    if (current == end) break;  // Endpoint is occupied, never free.
    traversed->push_back(current);
  }
}

void RaycastMapper::integrateTsdfAlongRay(const Eigen::Vector3d& origin,
                                          const Eigen::Vector3d& endpoint,
                                          const std::vector<VoxelKey>& ray) {
  const Eigen::Vector3d segment = endpoint - origin;
  const double surface_distance = segment.norm();
  if (surface_distance <= 1e-9) return;
  const Eigen::Vector3d direction = segment / surface_distance;

  const auto fuse = [&](const VoxelKey& voxel) {
    // Projective signed distance: project the voxel center on the ray and take
    // the difference to the measured surface range. Positive in front of the
    // surface (free), negative behind it (occluded/solid).
    const double projection = (center(voxel) - origin).dot(direction);
    const double signed_distance = surface_distance - projection;
    if (signed_distance > config_.tsdf_truncation) {
      // The voxel lies clearly in front of the measured surface, i.e. in open
      // free space. If a surface was previously fused here it can only belong to
      // an obstacle that has since disappeared (typically a dynamic object), so
      // carve it away. Without this, dynamic geometry would leave a permanent
      // stale surface and poison the Dynablox ever-free test, which in turn
      // makes later dynamic points be accepted as static.
      auto existing = cells_.find(voxel);
      if (existing != cells_.end() && existing->second.tsdf_observed &&
          existing->second.tsdf <= config_.tsdf_surface_band) {
        existing->second.tsdf = config_.tsdf_truncation;
        existing->second.tsdf_weight = 0.0;
        existing->second.tsdf_observed = false;
      }
      return;
    }
    double truncated = signed_distance;
    if (truncated < -config_.tsdf_truncation) {
      truncated = -config_.tsdf_truncation;
    }
    auto insertion = cells_.emplace(voxel, Cell{});
    Cell& cell = insertion.first->second;
    if (insertion.second) cell.esdf_distance = config_.esdf_max_distance;
    // Saturating running average: the weight is capped so surfaces remain
    // adaptive and stale geometry can be overwritten by fresh observations.
    cell.tsdf = (cell.tsdf * cell.tsdf_weight + truncated) /
                (cell.tsdf_weight + 1.0);
    cell.tsdf_weight = std::min(cell.tsdf_weight + 1.0, config_.tsdf_max_weight);
    cell.tsdf_observed = true;
  };

  // traceRay omits the endpoint voxel, so fuse it explicitly.
  for (const auto& voxel : ray) fuse(voxel);
  fuse(key(endpoint));
}

void RaycastMapper::clearTsdfAlongRay(const std::vector<VoxelKey>& ray) {
  if (!config_.use_tsdf) return;
  for (const auto& voxel : ray) {
    auto insertion = cells_.emplace(voxel, Cell{});
    Cell& cell = insertion.first->second;
    if (insertion.second) cell.esdf_distance = config_.esdf_max_distance;
    cell.tsdf = config_.tsdf_truncation;
    cell.tsdf_weight = 0.0;
    cell.tsdf_observed = false;
  }
}

void RaycastMapper::classifyDynamic(
    const std::vector<Eigen::Vector3d>& points_world,
    const std::vector<Eigen::Vector3d>& points_sensor,
    const std::vector<std::size_t>& valid_indices,
    std::vector<DynamicCluster>* clusters_out,
    std::vector<std::size_t>* dynamic_indices) const {
  if (!config_.use_dynablox) return;

  std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash> point_map;
  for (const std::size_t index : valid_indices) {
    const double z = config_.dynamic_z_use_map_frame
                         ? points_world[index].z()
                         : points_sensor[index].z();
    if (z < config_.dynamic_z_min || z > config_.dynamic_z_max) {
      continue;
    }
    point_map[key(points_world[index])].push_back(index);
  }

  const auto neighbors = neighborOffsets(config_.cluster_connectivity);

  // Dynablox-style voxel clustering. Seeds are currently occupied voxels that
  // are ever-free. A cluster only grows *through* ever-free voxels; occupied
  // but not ever-free neighbours are attached as leaves and do not propagate
  // further. With cluster_grow_clusters_twice the cluster also grows one extra
  // ring beyond ever-free space.
  struct Cluster {
    KeySet voxels;
    Eigen::Vector3d min_corner =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d max_corner =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    std::vector<std::size_t> points;
  };
  const auto absorb = [&](Cluster* cluster, const VoxelKey& voxel,
                          const std::vector<std::size_t>& indices) {
    cluster->voxels.insert(voxel);
    const Eigen::Vector3d position = center(voxel);
    cluster->min_corner = cluster->min_corner.cwiseMin(position);
    cluster->max_corner = cluster->max_corner.cwiseMax(position);
    cluster->points.insert(cluster->points.end(), indices.begin(), indices.end());
  };
  const auto ever_free_voxel = [&](const VoxelKey& voxel) {
    const auto it = cells_.find(voxel);
    return it != cells_.end() && it->second.ever_free;
  };

  std::vector<Cluster> clusters;
  KeySet processed;  // Every voxel belongs to at most one cluster.
  for (const auto& entry : point_map) {
    const VoxelKey seed = entry.first;
    if (!ever_free_voxel(seed) || processed.count(seed) != 0) {
      continue;
    }

    Cluster cluster;
    std::vector<VoxelKey> stack{seed};
    while (!stack.empty()) {
      const VoxelKey voxel = stack.back();
      stack.pop_back();
      if (!processed.insert(voxel).second) {
        continue;
      }
      const auto point_it = point_map.find(voxel);
      if (point_it == point_map.end()) {
        continue;
      }
      absorb(&cluster, voxel, point_it->second);

      const bool voxel_ever_free = ever_free_voxel(voxel);
      for (const auto& offset : neighbors) {
        const VoxelKey neighbor{voxel.x + offset.x, voxel.y + offset.y,
                                voxel.z + offset.z};
        if (processed.count(neighbor) != 0) {
          continue;
        }
        const auto neighbor_points = point_map.find(neighbor);
        if (neighbor_points == point_map.end()) {
          continue;  // Only voxels occupied by the current scan can join.
        }
        if (ever_free_voxel(neighbor) ||
            (voxel_ever_free && config_.cluster_grow_clusters_twice)) {
          stack.push_back(neighbor);
        } else if (voxel_ever_free) {
          // Leaf: reachable from ever-free space but not ever-free itself.
          processed.insert(neighbor);
          absorb(&cluster, neighbor, neighbor_points->second);
        }
      }
    }
    clusters.push_back(std::move(cluster));
  }

  // Merge clusters that touch or come closer than min_cluster_separation.
  if (config_.cluster_min_separation > 0.0 && clusters.size() > 1) {
    const auto clusters_close = [&](const Cluster& a, const Cluster& b) {
      const double separation = config_.cluster_min_separation;
      // Bounding boxes are exact outer bounds, so a larger gap rules out any
      // close voxel pair without an O(n*m) scan.
      if ((b.min_corner - a.max_corner).maxCoeff() > separation ||
          (a.min_corner - b.max_corner).maxCoeff() > separation) {
        return false;
      }
      for (const auto& voxel_a : a.voxels) {
        const Eigen::Vector3d center_a = center(voxel_a);
        for (const auto& voxel_b : b.voxels) {
          if ((center_a - center(voxel_b)).norm() <= separation) {
            return true;
          }
        }
      }
      return false;
    };
    bool merged = true;
    while (merged) {
      merged = false;
      for (std::size_t i = 0; i < clusters.size() && !merged; ++i) {
        for (std::size_t j = i + 1; j < clusters.size(); ++j) {
          if (!clusters_close(clusters[i], clusters[j])) {
            continue;
          }
          clusters[i].voxels.insert(clusters[j].voxels.begin(),
                                    clusters[j].voxels.end());
          clusters[i].min_corner =
              clusters[i].min_corner.cwiseMin(clusters[j].min_corner);
          clusters[i].max_corner =
              clusters[i].max_corner.cwiseMax(clusters[j].max_corner);
          clusters[i].points.insert(clusters[i].points.end(),
                                    clusters[j].points.begin(),
                                    clusters[j].points.end());
          clusters.erase(clusters.begin() + static_cast<std::ptrdiff_t>(j));
          merged = true;
          break;
        }
      }
    }
  }

  // Cluster-level filters, then emit surviving clusters with their centroid,
  // bounding box and point ids.
  for (const auto& cluster : clusters) {
    const std::size_t count = cluster.points.size();
    if (count < static_cast<std::size_t>(std::max(0, config_.cluster_min_points)) ||
        count > static_cast<std::size_t>(std::max(0, config_.cluster_max_points))) {
      continue;
    }
    const Eigen::Vector3d extent = cluster.max_corner - cluster.min_corner;
    if (extent.minCoeff() < config_.cluster_min_extent ||
        extent.maxCoeff() > config_.cluster_max_extent) {
      continue;
    }
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const std::size_t index : cluster.points) {
      centroid += points_world[index];
    }
    centroid /= static_cast<double>(count);

    DynamicCluster output;
    output.centroid = centroid;
    output.min_corner = cluster.min_corner;
    output.max_corner = cluster.max_corner;
    output.point_count = count;
    clusters_out->push_back(output);
    dynamic_indices->insert(dynamic_indices->end(), cluster.points.begin(),
                            cluster.points.end());
  }
}

FrameResult RaycastMapper::integrate(
    const std::vector<Eigen::Vector3d>& points_world,
    const std::vector<Eigen::Vector3d>& points_sensor,
    const Eigen::Vector3d& sensor_origin_world,
    const std::vector<Eigen::Vector3d>& free_endpoints_world,
    bool update_esdf) {
  if (points_world.size() != points_sensor.size()) {
    throw std::invalid_argument("world and sensor point arrays have different sizes");
  }

  std::vector<std::size_t> valid_indices;
  valid_indices.reserve(points_world.size());
  for (std::size_t i = 0; i < points_world.size(); ++i) {
    const double range = points_sensor[i].norm();
    if (points_world[i].allFinite() && points_sensor[i].allFinite() &&
        range >= config_.min_range && range <= config_.max_range) {
      valid_indices.push_back(i);
    }
  }

  std::vector<DynamicCluster> dynamic_clusters;
  std::vector<std::size_t> dynamic_indices;
  classifyDynamic(points_world, points_sensor, valid_indices, &dynamic_clusters,
                  &dynamic_indices);
  std::vector<uint8_t> is_dynamic(points_world.size(), uint8_t{0});
  for (const auto index : dynamic_indices) is_dynamic[index] = 1;

  KeySet free_voxels;
  KeySet all_endpoints;
  KeySet static_endpoints;
  std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> unique_endpoints;
  std::vector<VoxelKey> ray;
  for (const auto index : valid_indices) {
    const VoxelKey endpoint = key(points_world[index]);
    all_endpoints.insert(endpoint);
    unique_endpoints.emplace(endpoint, index);
    if (!is_dynamic[index]) static_endpoints.insert(endpoint);
  }
  for (const auto& endpoint : unique_endpoints) {
    traceRay(sensor_origin_world, points_world[endpoint.second], &ray);
    free_voxels.insert(ray.begin(), ray.end());
    // Only static returns contribute to the TSDF surface; dynamic returns are
    // excluded so moving objects do not imprint a surface into the map.
    if (config_.use_tsdf && !is_dynamic[endpoint.second]) {
      integrateTsdfAlongRay(sensor_origin_world, points_world[endpoint.second], ray);
    }
  }
  for (const auto& endpoint : free_endpoints_world) {
    if (!endpoint.allFinite()) continue;
    traceRay(sensor_origin_world, endpoint, &ray);
    ray.push_back(key(endpoint));
    free_voxels.insert(ray.begin(), ray.end());
    clearTsdfAlongRay(ray);
  }
  for (const auto& endpoint : all_endpoints) free_voxels.erase(endpoint);

  for (const auto& voxel : free_voxels) {
    auto insertion = cells_.emplace(voxel, Cell{});
    Cell& cell = insertion.first->second;
    if (insertion.second) cell.esdf_distance = config_.esdf_max_distance;
    cell.log_odds = std::max(min_log_odds_, cell.log_odds + miss_log_odds_);
    ++cell.free_observations;
  }
  for (const auto& voxel : static_endpoints) {
    auto insertion = cells_.emplace(voxel, Cell{});
    Cell& cell = insertion.first->second;
    if (insertion.second) cell.esdf_distance = config_.esdf_max_distance;
    cell.log_odds = std::min(max_log_odds_, cell.log_odds + hit_log_odds_);
  }
  const auto ever_free_neighbors =
      neighborOffsets(config_.ever_free_neighbor_connectivity);
  for (const auto& voxel : all_endpoints) {
    auto insertion = cells_.emplace(voxel, Cell{});
    Cell& cell = insertion.first->second;
    if (insertion.second) cell.esdf_distance = config_.esdf_max_distance;
    if (cell.last_occupied_frame >=
        frame_index_ - config_.ever_free_temporal_buffer_frames) {
      ++cell.occupancy_streak;
    } else {
      cell.occupancy_streak = 1;
    }
    cell.last_occupied_frame = frame_index_;
    if (cell.occupancy_streak >= config_.ever_free_reset_occupied_frames) {
      cell.ever_free = false;
      for (const auto& offset : ever_free_neighbors) {
        const VoxelKey neighbor{voxel.x + offset.x, voxel.y + offset.y,
                                voxel.z + offset.z};
        const auto neighbor_it = cells_.find(neighbor);
        if (neighbor_it != cells_.end()) neighbor_it->second.ever_free = false;
      }
    }
  }

  // Dynablox-style high-confidence free space: a voxel becomes ever-free only
  // when it has been observed free, is not occupied, has not been occupied for
  // the last ever_free_burn_in_period frames, and the same holds for its whole
  // neighbourhood. The neighbourhood check suppresses occlusion-boundary seeds.
  // A voxel counts as free enough to seed ever-free space. In TSDF mode the
  // surface band already provides the margin; in probability mode an optional
  // stricter threshold keeps marginal voxels (e.g. obliquely grazed facades)
  // from being labeled ever-free.
  const auto ever_free_eligible = [&](const Cell& cell) {
    if (config_.use_tsdf) return !occupied(cell);
    if (config_.ever_free_max_free_probability > 0.0) {
      return probability(cell) <= config_.ever_free_max_free_probability;
    }
    return !occupied(cell);
  };

  for (const auto& voxel : free_voxels) {
    Cell& candidate = cells_.at(voxel);
    if (candidate.ever_free || candidate.free_observations == 0) {
      continue;
    }
    bool high_confidence =
        ever_free_eligible(candidate) &&
        frame_index_ - candidate.last_occupied_frame >=
            config_.ever_free_burn_in_period;
    for (const auto& offset : ever_free_neighbors) {
      if (!high_confidence) break;
      const VoxelKey neighbor{voxel.x + offset.x, voxel.y + offset.y,
                              voxel.z + offset.z};
      const auto it = cells_.find(neighbor);
      high_confidence = it != cells_.end() &&
          it->second.free_observations > 0 && ever_free_eligible(it->second) &&
          frame_index_ - it->second.last_occupied_frame >=
              config_.ever_free_burn_in_period;
    }
    if (high_confidence) candidate.ever_free = true;
  }

  if (update_esdf &&
      (frame_index_ + 1) % config_.esdf_update_every_n_frames == 0) {
    updateEsdf();
  }

  FrameResult result;
  result.valid_points = valid_indices.size();
  result.dynamic_clusters = std::move(dynamic_clusters);
  result.static_points_world.reserve(valid_indices.size());
  result.dynamic_points_world.reserve(dynamic_indices.size());
  for (const auto index : valid_indices) {
    (is_dynamic[index] ? result.dynamic_points_world : result.static_points_world)
        .push_back(points_world[index]);
  }
  ++frame_index_;
  return result;
}

void RaycastMapper::updateEsdfNow() { updateEsdf(); }

VoxelState RaycastMapper::voxelState(const Eigen::Vector3d& position) const {
  const auto it = cells_.find(key(position));
  if (it == cells_.end()) return VoxelState::kUnknown;
  return occupied(it->second) ? VoxelState::kOccupied : VoxelState::kFree;
}

double RaycastMapper::distance(const Eigen::Vector3d& position) const {
  const auto it = cells_.find(key(position));
  if (it == cells_.end()) return -1.0;
  if (occupied(it->second)) return 0.0;
  return it->second.esdf_distance;
}

Eigen::Vector3d RaycastMapper::distanceGradient(
    const Eigen::Vector3d& position) const {
  const auto it = cells_.find(key(position));
  if (it == cells_.end() || !it->second.has_closest_obstacle) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d delta =
      position - center(it->second.closest_obstacle);
  return delta.norm() > 1e-9 ? delta.normalized() : Eigen::Vector3d::Zero();
}

void RaycastMapper::clearBox(const Eigen::Vector3d& box_center,
                             const Eigen::Vector3d& size) {
  const VoxelKey min_key = key(box_center - 0.5 * size);
  const VoxelKey max_key = key(box_center + 0.5 * size);
  for (int x = min_key.x; x <= max_key.x; ++x) {
    for (int y = min_key.y; y <= max_key.y; ++y) {
      for (int z = min_key.z; z <= max_key.z; ++z) {
        Cell& cell = cells_[VoxelKey{x, y, z}];
        cell.log_odds = min_log_odds_;
        ++cell.free_observations;
        cell.esdf_distance = config_.esdf_max_distance;
        cell.has_closest_obstacle = false;
        cell.tsdf = config_.tsdf_truncation;
        cell.tsdf_weight = 0.0;
        cell.tsdf_observed = false;
      }
    }
  }
}

void RaycastMapper::updateEsdf() {
  struct QueueEntry {
    double distance;
    VoxelKey voxel;
  };
  struct Greater {
    bool operator()(const QueueEntry& a, const QueueEntry& b) const {
      return a.distance > b.distance;
    }
  };
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, Greater> queue;

  for (auto& entry : cells_) {
    Cell& cell = entry.second;
    cell.esdf_distance = config_.esdf_max_distance;
    cell.has_closest_obstacle = false;
    if (occupied(cell)) {
      cell.esdf_distance = 0.0;
      cell.closest_obstacle = entry.first;
      cell.has_closest_obstacle = true;
      queue.push({0.0, entry.first});
    }
  }

  // Propagate the obstacle site, rather than accumulating edge lengths. This
  // keeps the stored value Euclidean while operating directly on sparse hash
  // voxels, in the same spirit as a voxel-block ESDF integrator.
  const auto offsets = [&]() {
    std::vector<VoxelKey> result;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
          if (dx != 0 || dy != 0 || dz != 0) result.push_back({dx, dy, dz});
    return result;
  }();

  while (!queue.empty()) {
    const QueueEntry current = queue.top();
    queue.pop();
    auto current_it = cells_.find(current.voxel);
    if (current_it == cells_.end() ||
        current.distance > current_it->second.esdf_distance + 1e-9 ||
        !current_it->second.has_closest_obstacle) {
      continue;
    }
    const VoxelKey obstacle = current_it->second.closest_obstacle;
    for (const auto& offset : offsets) {
      const VoxelKey neighbor{current.voxel.x + offset.x,
                              current.voxel.y + offset.y,
                              current.voxel.z + offset.z};
      auto neighbor_it = cells_.find(neighbor);
      if (neighbor_it == cells_.end()) continue;  // Unknown space is not allocated.
      const double dx = static_cast<double>(neighbor.x - obstacle.x);
      const double dy = static_cast<double>(neighbor.y - obstacle.y);
      const double dz = static_cast<double>(neighbor.z - obstacle.z);
      const double distance = config_.resolution * std::sqrt(dx * dx + dy * dy + dz * dz);
      if (distance <= config_.esdf_max_distance &&
          distance + 1e-9 < neighbor_it->second.esdf_distance) {
        neighbor_it->second.esdf_distance = distance;
        neighbor_it->second.closest_obstacle = obstacle;
        neighbor_it->second.has_closest_obstacle = true;
        queue.push({distance, neighbor});
      }
    }
  }
}

std::vector<MapPoint> RaycastMapper::occupiedVoxels() const {
  std::vector<MapPoint> result;
  result.reserve(cells_.size() / 4);
  for (const auto& entry : cells_) {
    if (occupied(entry.second)) {
      result.push_back({center(entry.first), probability(entry.second),
                        entry.second.esdf_distance, entry.second.tsdf});
    }
  }
  return result;
}

std::vector<MapPoint> RaycastMapper::knownVoxels() const {
  std::vector<MapPoint> result;
  result.reserve(cells_.size());
  for (const auto& entry : cells_) {
    // Voxels never hit by a TSDF ray read as free (+truncation) rather than 0,
    // so consumers can treat "tsdf <= tsdf_surface_band" as the surface test.
    const double tsdf = entry.second.tsdf_observed ? entry.second.tsdf
                                                   : config_.tsdf_truncation;
    result.push_back({center(entry.first), probability(entry.second),
                      entry.second.esdf_distance, tsdf});
  }
  return result;
}

void RaycastMapper::reset() {
  cells_.clear();
  frame_index_ = 0;
}

}  // namespace raycast_static_map
