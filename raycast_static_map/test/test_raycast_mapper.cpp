#include "raycast_static_map/raycast_mapper.hpp"
#include "raycast_static_map/object_tracker.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

namespace raycast_static_map {

TEST(RaycastMapper, IntegratesRayAndComputesEsdf) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 10.0;
  config.use_dynablox = false;
  config.esdf_max_distance = 10.0;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const std::vector<Eigen::Vector3d> points{Eigen::Vector3d(3.2, 0.1, 0.1)};
  mapper.integrate(points, points, Eigen::Vector3d::Zero());
  const auto occupied = mapper.occupiedVoxels();
  ASSERT_EQ(occupied.size(), 1u);
  EXPECT_NEAR(occupied.front().position.x(), 3.5, 1e-9);

  bool found_preceding_voxel = false;
  for (const auto& voxel : mapper.knownVoxels()) {
    if (std::abs(voxel.position.x() - 2.5) < 1e-9) {
      EXPECT_NEAR(voxel.esdf_distance, 1.0, 1e-9);
      found_preceding_voxel = true;
    }
  }
  EXPECT_TRUE(found_preceding_voxel);
}

TEST(RaycastMapper, DynabloxMarksReturnInEverFreeSpace) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 10.0;
  config.use_dynablox = true;
  config.ever_free_burn_in_period = 0;
  config.ever_free_temporal_buffer_frames = 0;
  config.ever_free_reset_occupied_frames = 3;
  config.ever_free_neighbor_connectivity = 6;
  config.cluster_connectivity = 6;
  config.cluster_grow_clusters_twice = false;
  config.cluster_min_separation = 0.0;
  config.cluster_min_extent = 0.0;
  config.cluster_max_extent = 1e9;
  config.cluster_min_points = 1;
  config.cluster_max_points = 100;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const std::vector<Eigen::Vector3d> background{
      Eigen::Vector3d(4.2, 0.1, 0.1),
      Eigen::Vector3d(4.2, 2.2, 0.1),
      Eigen::Vector3d(4.2, -1.8, 0.1),
      Eigen::Vector3d(4.2, 0.1, 2.2),
      Eigen::Vector3d(4.2, 0.1, -1.8)};
  mapper.integrate(background, background, Eigen::Vector3d::Zero());
  const std::vector<Eigen::Vector3d> moving{Eigen::Vector3d(2.2, 0.1, 0.1)};
  const FrameResult result = mapper.integrate(moving, moving, Eigen::Vector3d::Zero());
  EXPECT_EQ(result.dynamic_points_world.size(), 1u);
  EXPECT_TRUE(result.static_points_world.empty());
}

TEST(RaycastMapper, TsdfCriterionMarksSurfaceVoxelOnly) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 10.0;
  config.use_dynablox = false;
  config.use_tsdf = true;
  config.tsdf_truncation = 0.6;
  config.tsdf_surface_band = 0.1;
  config.esdf_max_distance = 10.0;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const std::vector<Eigen::Vector3d> points{Eigen::Vector3d(3.2, 0.1, 0.1)};
  mapper.integrate(points, points, Eigen::Vector3d::Zero());
  const auto occupied = mapper.occupiedVoxels();
  ASSERT_EQ(occupied.size(), 1u);
  EXPECT_NEAR(occupied.front().position.x(), 3.5, 1e-9);
  EXPECT_LE(occupied.front().tsdf, config.tsdf_surface_band);

  // Free voxels along the ray stay unobserved by the TSDF and therefore free,
  // even though the probability log-odds update is still running.
  for (const auto& voxel : mapper.knownVoxels()) {
    if (std::abs(voxel.position.x() - 2.5) < 1e-9) {
      EXPECT_GT(voxel.tsdf, config.tsdf_surface_band);
    }
  }
}

namespace {
bool hasVoxelAtX(const std::vector<MapPoint>& voxels, double x) {
  return std::any_of(voxels.begin(), voxels.end(), [x](const MapPoint& voxel) {
    return std::abs(voxel.position.x() - x) < 1e-9;
  });
}
}  // namespace

TEST(RaycastMapper, TsdfCarvesStaleSurfaceAfterObstacleRemoved) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 20.0;
  config.use_dynablox = false;
  config.use_tsdf = true;
  config.tsdf_truncation = 0.6;
  config.tsdf_surface_band = 0.1;
  config.tsdf_max_weight = 30.0;
  config.esdf_max_distance = 20.0;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  // Frame 1: a "dynamic" obstacle sits at 3.2 m and is fused into the map.
  const std::vector<Eigen::Vector3d> obstacle{Eigen::Vector3d(3.2, 0.0, 0.0)};
  mapper.integrate(obstacle, obstacle, origin);
  EXPECT_TRUE(hasVoxelAtX(mapper.occupiedVoxels(), 3.5));

  // Frame 2: the obstacle is gone and the ray now reaches the wall at 6.2 m,
  // passing far in front of the stale surface. It must be carved away.
  const std::vector<Eigen::Vector3d> wall{Eigen::Vector3d(6.2, 0.0, 0.0)};
  mapper.integrate(wall, wall, origin);
  const auto occupied = mapper.occupiedVoxels();
  EXPECT_FALSE(hasVoxelAtX(occupied, 3.5));
  EXPECT_TRUE(hasVoxelAtX(occupied, 6.5));
}

TEST(RaycastMapper, ExposesPlannerVoxelQueriesAndClearBox) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 10.0;
  config.use_dynablox = false;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const std::vector<Eigen::Vector3d> obstacle{Eigen::Vector3d(3.2, 0.1, 0.1)};
  mapper.integrate(obstacle, obstacle, Eigen::Vector3d::Zero());
  EXPECT_EQ(mapper.voxelState(Eigen::Vector3d(3.2, 0.1, 0.1)),
            VoxelState::kOccupied);
  EXPECT_EQ(mapper.voxelState(Eigen::Vector3d(2.2, 0.1, 0.1)),
            VoxelState::kFree);
  EXPECT_EQ(mapper.voxelState(Eigen::Vector3d(8.2, 0.1, 0.1)),
            VoxelState::kUnknown);
  EXPECT_NEAR(mapper.distance(Eigen::Vector3d(2.2, 0.1, 0.1)), 1.0, 1e-9);

  mapper.clearBox(Eigen::Vector3d(3.5, 0.5, 0.5),
                  Eigen::Vector3d(0.5, 0.5, 0.5));
  EXPECT_EQ(mapper.voxelState(Eigen::Vector3d(3.2, 0.1, 0.1)),
            VoxelState::kFree);
}

TEST(RaycastMapper, FreespaceRayDoesNotCreateTsdfSurface) {
  MapperConfig config;
  config.resolution = 1.0;
  config.min_range = 0.0;
  config.max_range = 10.0;
  config.use_dynablox = false;
  config.use_tsdf = true;
  config.esdf_update_every_n_frames = 1;
  RaycastMapper mapper(config);

  const std::vector<Eigen::Vector3d> free_endpoints{
      Eigen::Vector3d(6.2, 0.1, 0.1)};
  mapper.integrate({}, {}, Eigen::Vector3d::Zero(), free_endpoints);
  EXPECT_TRUE(mapper.occupiedVoxels().empty());
  EXPECT_EQ(mapper.voxelState(Eigen::Vector3d(6.2, 0.1, 0.1)),
            VoxelState::kFree);
}

namespace {
DynamicCluster makeCluster(double x, double y, double z) {
  DynamicCluster cluster;
  cluster.centroid = Eigen::Vector3d(x, y, z);
  cluster.min_corner = Eigen::Vector3d(x - 0.5, y - 0.5, z - 0.5);
  cluster.max_corner = Eigen::Vector3d(x + 0.5, y + 0.5, z + 0.5);
  cluster.point_count = 100;
  return cluster;
}
}  // namespace

TEST(ObjectTracker, ConfirmsTrackAndKeepsId) {
  TrackerConfig config;
  config.min_track_duration = 2;
  ObjectTracker tracker(config);
  const auto first = tracker.update({makeCluster(0.0, 0.0, 0.0)}, 0.0);
  ASSERT_EQ(first.size(), 1u);
  EXPECT_FALSE(first.front().confirmed);
  const auto second = tracker.update({makeCluster(0.1, 0.0, 0.0)}, 0.1);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_TRUE(second.front().confirmed);
  EXPECT_EQ(second.front().id, first.front().id);
}

TEST(ObjectTracker, EstimatesVelocityAndPredicts) {
  TrackerConfig config;
  config.kalman_process_noise = 4.0;
  config.kalman_measurement_noise = 0.2;
  config.velocity_horizon = 1.0;
  config.min_track_duration = 1;
  config.min_velocity = 0.0;
  ObjectTracker tracker(config);
  tracker.update({makeCluster(0.0, 0.0, 0.0)}, 0.0);
  tracker.update({makeCluster(1.0, 0.0, 0.0)}, 1.0);
  const auto objects = tracker.update({makeCluster(2.0, 0.0, 0.0)}, 2.0);
  ASSERT_EQ(objects.size(), 1u);
  // The constant-velocity Kalman filter should converge close to 1 m/s.
  EXPECT_NEAR(objects.front().velocity.x(), 1.0, 0.2);
  EXPECT_NEAR(objects.front().speed, 1.0, 0.2);
  EXPECT_NEAR(objects.front().predicted_position.x(), 3.0, 0.5);
}

TEST(ObjectTracker, DropsTrackAfterMisses) {
  TrackerConfig config;
  config.max_misses = 0;
  ObjectTracker tracker(config);
  tracker.update({makeCluster(0.0, 0.0, 0.0)}, 0.0);
  const auto objects = tracker.update({}, 0.1);
  EXPECT_TRUE(objects.empty());
}

TEST(ObjectTracker, SeparatesDistantClusters) {
  TrackerConfig config;
  config.max_association_distance = 1.0;
  ObjectTracker tracker(config);
  const auto objects = tracker.update(
      {makeCluster(0.0, 0.0, 0.0), makeCluster(10.0, 0.0, 0.0)}, 0.0);
  ASSERT_EQ(objects.size(), 2u);
  EXPECT_NE(objects[0].id, objects[1].id);
}

}  // namespace raycast_static_map

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
