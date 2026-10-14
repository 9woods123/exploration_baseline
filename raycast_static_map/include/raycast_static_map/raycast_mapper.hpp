#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "raycast_static_map/dynamic_object.hpp"

namespace raycast_static_map {

struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;
  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const;
};

struct MapperConfig {
  double resolution = 0.20;
  double min_range = 1.0;
  double max_range = 80.0;
  double prob_hit = 0.70;
  double prob_miss = 0.40;
  double prob_min = 0.12;
  double prob_max = 0.97;
  double occupied_probability = 0.60;
  // Occupancy criterion selection. When false (default) occupancy is decided by
  // the fused log-odds probability threshold; when true the map is built from a
  // truncated signed distance field (TSDF) fused along each ray cast. The
  // log-odds probability update is always maintained, so the criterion can be
  // switched without changing the rest of the pipeline.
  bool use_tsdf = false;
  // TSDF truncation distance in meters, applied on both sides of the surface.
  double tsdf_truncation = 0.60;
  // Surface band half-width in meters. A TSDF voxel is treated as occupied when
  // its fused signed distance is <= this value. Keep it comfortably below
  // tsdf_truncation (typically about half the map resolution).
  double tsdf_surface_band = 0.10;
  // Upper bound on the per-voxel TSDF fusion weight. The running average
  // saturates here so surfaces stay adaptive instead of becoming permanently
  // rigid; it also bounds how many samples a stale surface needs to be
  // overwritten by fresh observations.
  double tsdf_max_weight = 30.0;
  bool use_dynablox = true;
  // Height band used to restrict dynamic detection. By default the band is
  // interpreted in the sensor frame (z relative to the LiDAR), which is how the
  // original pipeline consumed it. Set dynamic_z_use_map_frame = true to apply
  // the band in the map frame instead: for flat ground this is much more
  // robust, because a moving or pitching platform no longer shifts the ground
  // out of the band. Ground returns are a classical false positive for
  // ever-free based detection, so the band is the first knob to tune.
  bool dynamic_z_use_map_frame = false;
  double dynamic_z_min = -2.0;
  double dynamic_z_max = 2.5;
  // Dynablox EverFreeIntegrator parameters. Names and semantics intentionally
  // follow the original implementation (ethz-asl/dynablox).
  // A voxel may only become ever-free when it has been observed free, is not
  // occupied, and has not been occupied for the last ever_free_burn_in_period
  // frames. The same recency check is applied to the whole neighbourhood.
  int ever_free_burn_in_period = 5;
  // Frames a voxel may be free in between occupied observations without losing
  // its occupancy counter (compensates for LiDAR sparsity).
  int ever_free_temporal_buffer_frames = 2;
  // After this many accumulated occupied observations an ever-free voxel is
  // reset to never-free and re-admitted as static.
  int ever_free_reset_occupied_frames = 150;
  int ever_free_neighbor_connectivity = 26;
  // Optional free margin for the ever-free test (probability mode only). A voxel
  // may only become ever-free when its fused probability is at or below this
  // value. This keeps voxels that merely dip under the occupancy threshold - for
  // example a flat facade grazed by oblique rays - from seeding false dynamic
  // clusters. 0 disables the extra margin and uses the plain occupancy test.
  double ever_free_max_free_probability = 0.0;
  // Dynablox Clustering parameters, following the original implementation.
  int cluster_connectivity = 6;
  // Grow the cluster one extra voxel ring beyond ever-free space.
  bool cluster_grow_clusters_twice = false;
  // Merge clusters whose voxels are closer than this distance [m].
  double cluster_min_separation = 0.0;
  // Filter clusters by the extent [m] of their axis-aligned bounding box.
  double cluster_min_extent = 0.0;
  double cluster_max_extent = 200000.0;
  int cluster_min_points = 20;
  int cluster_max_points = 200000;
  double esdf_max_distance = 5.0;
  int esdf_update_every_n_frames = 1;
};

struct MapPoint {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double probability = 0.5;
  double esdf_distance = 0.0;
  // Fused truncated signed distance. Positive in free space, negative behind
  // the surface. Voxels never observed by a TSDF ray report +tsdf_truncation so
  // that non-negative always means "in front of / far from a surface".
  double tsdf = 0.0;
};

struct FrameResult {
  std::vector<Eigen::Vector3d> static_points_world;
  std::vector<Eigen::Vector3d> dynamic_points_world;
  // One entry per dynamic cluster after cluster-level filtering, carrying the
  // centroid and axis-aligned bounding box for downstream tracking.
  std::vector<DynamicCluster> dynamic_clusters;
  std::size_t valid_points = 0;
};

enum class VoxelState { kUnknown = 0, kFree, kOccupied };

class RaycastMapper {
 public:
  explicit RaycastMapper(const MapperConfig& config);

  FrameResult integrate(const std::vector<Eigen::Vector3d>& points_world,
                        const std::vector<Eigen::Vector3d>& points_sensor,
                        const Eigen::Vector3d& sensor_origin_world,
                        const std::vector<Eigen::Vector3d>& free_endpoints_world = {},
                        bool update_esdf = true);
  // Explicit entry point used by the asynchronous map backend. A copied
  // mapper can rebuild its ESDF without holding the live occupancy-map lock.
  void updateEsdfNow();
  VoxelState voxelState(const Eigen::Vector3d& position) const;
  double distance(const Eigen::Vector3d& position) const;
  Eigen::Vector3d distanceGradient(const Eigen::Vector3d& position) const;
  void clearBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size);
  std::vector<MapPoint> occupiedVoxels() const;
  std::vector<MapPoint> knownVoxels() const;
  void reset();
  std::size_t voxelCount() const { return cells_.size(); }
  int frameIndex() const { return frame_index_; }
  const MapperConfig& config() const { return config_; }

 private:
  struct Cell {
    double log_odds = 0.0;
    int free_observations = 0;
    // Frame index at which the voxel was last occupied. Starts at 0 so the
    // burn-in period also applies from the first frame, matching Dynablox.
    int last_occupied_frame = 0;
    int occupancy_streak = 0;
    bool ever_free = false;
    // Truncated Euclidean distance to the nearest occupied voxel.
    double esdf_distance = 0.0;
    VoxelKey closest_obstacle;
    bool has_closest_obstacle = false;
    // Truncated signed distance fused as a weighted running average. Positive
    // in free space, negative behind the surface. Only maintained when
    // MapperConfig::use_tsdf is enabled; tsdf_observed marks that at least one
    // TSDF ray sample fell inside the truncation band.
    double tsdf = 0.0;
    double tsdf_weight = 0.0;
    bool tsdf_observed = false;
  };
  // Fuse the truncated signed distance of one ray into the voxels it covers.
  void integrateTsdfAlongRay(const Eigen::Vector3d& origin,
                             const Eigen::Vector3d& endpoint,
                             const std::vector<VoxelKey>& ray);
  void clearTsdfAlongRay(const std::vector<VoxelKey>& ray);
  using KeySet = std::unordered_set<VoxelKey, VoxelKeyHash>;
  using CellMap = std::unordered_map<VoxelKey, Cell, VoxelKeyHash>;

  VoxelKey key(const Eigen::Vector3d& point) const;
  Eigen::Vector3d center(const VoxelKey& key) const;
  void traceRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& endpoint,
                std::vector<VoxelKey>* traversed) const;
  // Cluster currently occupied ever-free voxels (Dynablox-style), returning the
  // surviving clusters and all their point indices.
  void classifyDynamic(const std::vector<Eigen::Vector3d>& points_world,
                       const std::vector<Eigen::Vector3d>& points_sensor,
                       const std::vector<std::size_t>& valid_indices,
                       std::vector<DynamicCluster>* clusters,
                       std::vector<std::size_t>* dynamic_indices) const;
  static std::vector<VoxelKey> neighborOffsets(int connectivity);
  bool occupied(const Cell& cell) const;
  double probability(const Cell& cell) const;
  void updateEsdf();

  MapperConfig config_;
  double hit_log_odds_ = 0.0;
  double miss_log_odds_ = 0.0;
  double min_log_odds_ = 0.0;
  double max_log_odds_ = 0.0;
  double occupied_log_odds_ = 0.0;
  int frame_index_ = 0;
  CellMap cells_;
};

}  // namespace raycast_static_map
