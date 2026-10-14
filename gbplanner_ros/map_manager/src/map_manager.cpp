#include "map_manager/map_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/bind.hpp>
#include <geometry_msgs/Point.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <raycast_static_map/raycast_mapper.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace {

VoxelStatus convertState(raycast_static_map::VoxelState state) {
  if (state == raycast_static_map::VoxelState::kOccupied) return kOccupied;
  if (state == raycast_static_map::VoxelState::kFree) return kFree;
  return kUnknown;
}

std::size_t voxelHash(const Eigen::Vector3d& point, double resolution) {
  const int64_t x = static_cast<int64_t>(std::floor(point.x() / resolution));
  const int64_t y = static_cast<int64_t>(std::floor(point.y() / resolution));
  const int64_t z = static_cast<int64_t>(std::floor(point.z() / resolution));
  std::size_t seed = std::hash<int64_t>{}(x);
  seed ^= std::hash<int64_t>{}(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<int64_t>{}(z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

}  // namespace

class MapManager::RaycastBackend {
 public:
  using Cloud = sensor_msgs::PointCloud2;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Cloud, Cloud>;

  RaycastBackend(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
      : nh_(nh), private_nh_(private_nh), mapper_(loadConfig(private_nh_)),
        hit_sub_(nh_, pointcloudTopic(private_nh_), 3),
        free_sub_(nh_, freespaceTopic(private_nh_), 3),
        sync_(SyncPolicy(5), hit_sub_, free_sub_) {
    private_nh_.param<std::string>("raycast_map/world_frame", world_frame_, "world");
    private_nh_.param("raycast_map/publish_every_n_frames", publish_every_n_frames_, 2);
    private_nh_.param("raycast_map/sync_tolerance", sync_tolerance_, 0.05);
    sync_.setMaxIntervalDuration(ros::Duration(sync_tolerance_));
    sync_.registerCallback(boost::bind(&RaycastBackend::cloudCallback, this, _1, _2));
    surface_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("surface_pointcloud", 1, true);
    occupied_pub_ = private_nh_.advertise<visualization_msgs::MarkerArray>("occupied_nodes", 1, true);
    dynamic_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("dynamic_pointcloud", 1, true);
    ROS_INFO("Raycast map backend: hits=%s free=%s frame=%s resolution=%.2f",
             pointcloudTopic(private_nh_).c_str(), freespaceTopic(private_nh_).c_str(),
             world_frame_.c_str(), mapper_.config().resolution);
  }

  double getResolution() const { return mapper_.config().resolution; }
  bool getStatus() const { return received_cloud_; }

  VoxelStatus getVoxelStatus(const Eigen::Vector3d& position) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return convertState(mapper_.voxelState(position));
  }

  double getDistance(const Eigen::Vector3d& position) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return mapper_.distance(position);
  }

  Eigen::Vector3d getGradient(const Eigen::Vector3d& position) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return mapper_.distanceGradient(position);
  }

  VoxelStatus getRayStatus(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                           bool stop_at_unknown, Eigen::Vector3d* end_voxel = nullptr,
                           double* distance = nullptr) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const Eigen::Vector3d ray = end - start;
    const double length = ray.norm();
    if (length < 1e-9) return getVoxelStatus(end);
    const Eigen::Vector3d direction = ray / length;
    const double step_size = getResolution() * ray_step_multiplier_;
    for (double step = 0.0; step <= length + 1e-9; step += step_size) {
      const Eigen::Vector3d point = start + std::min(step, length) * direction;
      const VoxelStatus status = getVoxelStatus(point);
      if (status == kOccupied || (status == kUnknown && stop_at_unknown)) {
        if (end_voxel) *end_voxel = point;
        if (distance) *distance = mapper_.distance(point);
        return status;
      }
    }
    if (end_voxel) *end_voxel = end;
    if (distance) *distance = mapper_.distance(end);
    return kFree;
  }

  VoxelStatus getBoxStatus(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                           bool stop_at_unknown) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const double resolution = getResolution();
    const Eigen::Vector3d half = size.cwiseAbs() * 0.5;
    bool unknown = false;
    for (double x = -half.x(); x <= half.x() + 1e-9; x += resolution) {
      for (double y = -half.y(); y <= half.y() + 1e-9; y += resolution) {
        for (double z = -half.z(); z <= half.z() + 1e-9; z += resolution) {
          const VoxelStatus status = getVoxelStatus(center + Eigen::Vector3d(x, y, z));
          if (status == kOccupied) return kOccupied;
          unknown = unknown || status == kUnknown;
        }
      }
    }
    return stop_at_unknown && unknown ? kUnknown : kFree;
  }

  VoxelStatus getPathStatus(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                            const Eigen::Vector3d& size, bool stop_at_unknown) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    const Eigen::Vector3d ray = end - start;
    const double length = ray.norm();
    const int steps = std::max(1, static_cast<int>(std::ceil(length / getResolution())));
    for (int i = 0; i <= steps; ++i) {
      const VoxelStatus status = getBoxStatus(start + ray * (static_cast<double>(i) / steps),
                                              size, stop_at_unknown);
      if (status != kFree) return status;
    }
    return kFree;
  }

  bool augmentFreeBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    mapper_.clearBox(center, size);
    return true;
  }

  void getScanStatus(Eigen::Vector3d& position, std::vector<Eigen::Vector3d>& endpoints,
                     std::tuple<int, int, int>& gain,
                     std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& log) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    int unknown = 0, free = 0, occupied = 0;
    for (const Eigen::Vector3d& endpoint : endpoints) {
      const Eigen::Vector3d ray = endpoint - position;
      const double length = ray.norm();
      if (length < 1e-9) continue;
      const Eigen::Vector3d direction = ray / length;
      for (double d = 0.0; d <= length; d += getResolution() * ray_step_multiplier_) {
        const Eigen::Vector3d point = position + d * direction;
        const VoxelStatus status = getVoxelStatus(point);
        log.emplace_back(point, status);
        if (status == kUnknown) ++unknown;
        else if (status == kFree) ++free;
        else { ++occupied; break; }
      }
    }
    gain = std::make_tuple(unknown, free, occupied);
  }

  void extractLocalMap(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                       const Eigen::Quaterniond& rotation,
                       std::vector<Eigen::Vector3d>& occupied,
                       std::vector<Eigen::Vector3d>& free) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    occupied.clear();
    free.clear();
    const Eigen::Vector3d half = size.cwiseAbs() * 0.5;
    for (double x = -half.x(); x <= half.x(); x += getResolution())
      for (double y = -half.y(); y <= half.y(); y += getResolution())
        for (double z = -half.z(); z <= half.z(); z += getResolution()) {
          const Eigen::Vector3d point = center + rotation * Eigen::Vector3d(x, y, z);
          const VoxelStatus status = getVoxelStatus(point);
          if (status == kOccupied) occupied.push_back(point);
          else if (status == kFree) free.push_back(point);
        }
  }

  void getLocalPointcloud(const Eigen::Vector3d& center, double range,
                          const Eigen::Vector2d* z_limits,
                          pcl::PointCloud<pcl::PointXYZI>& cloud,
                          bool include_unknown) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    cloud.clear();
    const auto known = mapper_.knownVoxels();
    for (const auto& voxel : known) {
      const Eigen::Vector3d delta = voxel.position - center;
      if (delta.norm() > range) continue;
      if (z_limits && (delta.z() < (*z_limits)[0] || delta.z() > (*z_limits)[1])) continue;
      if (getVoxelStatus(voxel.position) != kOccupied) continue;
      pcl::PointXYZI point;
      point.x = voxel.position.x(); point.y = voxel.position.y(); point.z = voxel.position.z();
      point.intensity = 250.0f;
      cloud.push_back(point);
    }
    (void)include_unknown;  // Unknown space is unbounded in this sparse map.
  }

  void reset() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    mapper_.reset();
    received_cloud_ = false;
    publishMap({});
  }

  void setRaycastingParams(bool nonuniform, double multiplier) {
    nonuniform_ray_cast_ = nonuniform;
    ray_step_multiplier_ = std::max(0.25, multiplier);
  }

 private:
  static std::string pointcloudTopic(ros::NodeHandle& nh) {
    std::string topic;
    nh.param<std::string>("raycast_map/pointcloud_topic", topic, "/pointcloud");
    return topic;
  }

  static std::string freespaceTopic(ros::NodeHandle& nh) {
    std::string topic;
    nh.param<std::string>("raycast_map/freespace_topic", topic, "/freespace_pointcloud");
    return topic;
  }

  static raycast_static_map::MapperConfig loadConfig(ros::NodeHandle& nh) {
    raycast_static_map::MapperConfig config;
    nh.param("raycast_map/resolution", config.resolution, config.resolution);
    nh.param("raycast_map/min_range", config.min_range, config.min_range);
    nh.param("raycast_map/max_range", config.max_range, config.max_range);
    nh.param("raycast_map/prob_hit", config.prob_hit, config.prob_hit);
    nh.param("raycast_map/prob_miss", config.prob_miss, config.prob_miss);
    nh.param("raycast_map/occupied_probability", config.occupied_probability, config.occupied_probability);
    nh.param("raycast_map/use_tsdf", config.use_tsdf, config.use_tsdf);
    nh.param("raycast_map/tsdf_truncation", config.tsdf_truncation, config.tsdf_truncation);
    nh.param("raycast_map/tsdf_surface_band", config.tsdf_surface_band, config.tsdf_surface_band);
    nh.param("raycast_map/use_dynablox", config.use_dynablox, config.use_dynablox);
    nh.param("raycast_map/esdf_max_distance", config.esdf_max_distance, config.esdf_max_distance);
    nh.param("raycast_map/esdf_update_every_n_frames", config.esdf_update_every_n_frames,
             config.esdf_update_every_n_frames);
    return config;
  }

  void cloudCallback(const Cloud::ConstPtr& hits, const Cloud::ConstPtr& freespace) {
    tf::StampedTransform transform;
    try {
      tf_listener_.lookupTransform(world_frame_, hits->header.frame_id, hits->header.stamp, transform);
    } catch (const tf::TransformException& error) {
      ROS_WARN_THROTTLE(2.0, "Raycast map waiting for point-cloud TF: %s", error.what());
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> hit_cloud, free_cloud;
    pcl::fromROSMsg(*hits, hit_cloud);
    pcl::fromROSMsg(*freespace, free_cloud);
    std::vector<Eigen::Vector3d> points_sensor, points_world, free_world;
    points_sensor.reserve(hit_cloud.size());
    points_world.reserve(hit_cloud.size());
    free_world.reserve(free_cloud.size());
    for (const pcl::PointXYZ& point : hit_cloud) {
      if (!pcl::isFinite(point)) continue;
      const tf::Vector3 sensor(point.x, point.y, point.z);
      const tf::Vector3 world = transform * sensor;
      points_sensor.emplace_back(point.x, point.y, point.z);
      points_world.emplace_back(world.x(), world.y(), world.z());
    }
    for (const pcl::PointXYZ& point : free_cloud) {
      if (!pcl::isFinite(point)) continue;
      const tf::Vector3 world = transform * tf::Vector3(point.x, point.y, point.z);
      free_world.emplace_back(world.x(), world.y(), world.z());
    }
    const tf::Vector3 origin_tf = transform.getOrigin();
    const Eigen::Vector3d origin(origin_tf.x(), origin_tf.y(), origin_tf.z());

    raycast_static_map::FrameResult result;
    {
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      result = mapper_.integrate(points_world, points_sensor, origin, free_world);
      received_cloud_ = true;
      if (mapper_.frameIndex() % std::max(1, publish_every_n_frames_) == 0)
        publishMap(result.dynamic_points_world);
    }
  }

  void publishMap(const std::vector<Eigen::Vector3d>& dynamic_points) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    const auto occupied = mapper_.occupiedVoxels();
    cloud.reserve(occupied.size());
    for (const auto& voxel : occupied)
      cloud.emplace_back(voxel.position.x(), voxel.position.y(), voxel.position.z());
    cloud.width = cloud.size(); cloud.height = 1; cloud.is_dense = true;
    sensor_msgs::PointCloud2 message;
    pcl::toROSMsg(cloud, message);
    message.header.stamp = ros::Time::now(); message.header.frame_id = world_frame_;
    surface_pub_.publish(message);

    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker marker;
    marker.header = message.header; marker.ns = "raycast_occupied"; marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE_LIST; marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = getResolution();
    marker.color.r = 0.9f; marker.color.g = 0.75f; marker.color.b = 0.1f; marker.color.a = 0.85f;
    marker.points.reserve(occupied.size());
    for (const auto& voxel : occupied) {
      geometry_msgs::Point point;
      point.x = voxel.position.x(); point.y = voxel.position.y(); point.z = voxel.position.z();
      marker.points.push_back(point);
    }
    markers.markers.push_back(marker); occupied_pub_.publish(markers);

    pcl::PointCloud<pcl::PointXYZ> dynamic_cloud;
    for (const auto& point : dynamic_points)
      dynamic_cloud.emplace_back(point.x(), point.y(), point.z());
    dynamic_cloud.width = dynamic_cloud.size(); dynamic_cloud.height = 1; dynamic_cloud.is_dense = true;
    pcl::toROSMsg(dynamic_cloud, message);
    message.header.stamp = ros::Time::now(); message.header.frame_id = world_frame_;
    dynamic_pub_.publish(message);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  mutable std::recursive_mutex mutex_;
  raycast_static_map::RaycastMapper mapper_;
  message_filters::Subscriber<Cloud> hit_sub_;
  message_filters::Subscriber<Cloud> free_sub_;
  message_filters::Synchronizer<SyncPolicy> sync_;
  tf::TransformListener tf_listener_;
  ros::Publisher surface_pub_, occupied_pub_, dynamic_pub_;
  std::string world_frame_;
  int publish_every_n_frames_ = 2;
  double sync_tolerance_ = 0.05;
  double ray_step_multiplier_ = 1.0;
  bool nonuniform_ray_cast_ = true;
  bool received_cloud_ = false;
};

MapManager::MapManager(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), nh_private_(private_nh), map_manager_impl_(new RaycastBackend(nh_, nh_private_)) {}
MapManager::~MapManager() = default;
double MapManager::getResolution() { return map_manager_impl_->getResolution(); }
bool MapManager::getStatus() { return map_manager_impl_->getStatus(); }
VoxelStatus MapManager::getVoxelStatus(const Eigen::Vector3d& p) { return map_manager_impl_->getVoxelStatus(p); }
float MapManager::getVoxelDistance(const Eigen::Vector3d& p) { return map_manager_impl_->getDistance(p); }
double MapManager::getPointDistance(const Eigen::Vector3d& p) { return map_manager_impl_->getDistance(p); }
Eigen::Vector3d MapManager::getPointGradient(const Eigen::Vector3d& p) { return map_manager_impl_->getGradient(p); }
VoxelStatus MapManager::getRayStatus(const Eigen::Vector3d& a, const Eigen::Vector3d& b, bool u) {
  return map_manager_impl_->getRayStatus(a, b, u);
}
VoxelStatus MapManager::getRayStatus(const Eigen::Vector3d& a, const Eigen::Vector3d& b, bool u,
                                     Eigen::Vector3d& end, double& distance) {
  return map_manager_impl_->getRayStatus(a, b, u, &end, &distance);
}
VoxelStatus MapManager::getBoxStatus(const Eigen::Vector3d& c, const Eigen::Vector3d& s, bool u) {
  return map_manager_impl_->getBoxStatus(c, s, u);
}
VoxelStatus MapManager::getPathStatus(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                      const Eigen::Vector3d& s, bool u) {
  return map_manager_impl_->getPathStatus(a, b, s, u);
}
bool MapManager::augmentFreeBox(const Eigen::Vector3d& c, const Eigen::Vector3d& s) {
  return map_manager_impl_->augmentFreeBox(c, s);
}
void MapManager::getScanStatus(Eigen::Vector3d& p, std::vector<Eigen::Vector3d>& e,
                               std::tuple<int, int, int>& g,
                               std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& l,
                               SensorParamsBase&) { map_manager_impl_->getScanStatus(p, e, g, l); }
void MapManager::annotateCameraVoxels(Eigen::Vector3d&, std::vector<Eigen::Vector3d>&) {}
void MapManager::getCameraScanStatus(Eigen::Vector3d& p, std::vector<Eigen::Vector3d>& e,
                                     std::tuple<int, int, int>& g,
                                     std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& l,
                                     SensorParamsBase&) { map_manager_impl_->getScanStatus(p, e, g, l); }
void MapManager::getCameraScanStatus(StateVec& state, SensorParamsBase& sensor,
                                     std::vector<VoxelLog>& logs) {
  std::vector<Eigen::Vector3d> endpoints;
  sensor.getFrustumEndpoints(state, endpoints);
  Eigen::Vector3d position = state.head(3);
  std::tuple<int, int, int> gain;
  std::vector<std::pair<Eigen::Vector3d, VoxelStatus>> voxels;
  map_manager_impl_->getScanStatus(position, endpoints, gain, voxels);
  for (const auto& voxel : voxels) {
    if (voxel.second != kUnknown && (voxel.first - position).norm() >= sensor.min_range)
      logs.emplace_back(voxel.first, voxelHash(voxel.first, getResolution()));
  }
}
void MapManager::augmentFreeFrustum() {
  ROS_WARN_THROTTLE(5.0, "Raycast map: augmentFreeFrustum is not required");
}
void MapManager::getFreeSpacePointCloud(std::vector<Eigen::Vector3d> endpoints, StateVec state,
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  const Eigen::Vector3d origin = state.head(3);
  const Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(state[3], Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(state[4], Eigen::Vector3d::UnitY()))
          .inverse()
          .toRotationMatrix();
  for (const auto& endpoint : endpoints) {
    if (map_manager_impl_->getRayStatus(origin, endpoint, false) == kOccupied) continue;
    const Eigen::Vector3d body = rotation * (endpoint - origin);
    cloud->emplace_back(body.x(), body.y(), body.z());
  }
}
void MapManager::extractLocalMap(const Eigen::Vector3d& c, const Eigen::Vector3d& s,
                                 std::vector<Eigen::Vector3d>& o, std::vector<Eigen::Vector3d>& f) {
  map_manager_impl_->extractLocalMap(c, s, Eigen::Quaterniond::Identity(), o, f);
}
void MapManager::extractLocalMapAlongAxis(const Eigen::Vector3d& c, const Eigen::Vector3d& axis,
                                          const Eigen::Vector3d& s,
                                          std::vector<Eigen::Vector3d>& o,
                                          std::vector<Eigen::Vector3d>& f) {
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  if (axis.norm() > 1e-9)
    rotation.setFromTwoVectors(Eigen::Vector3d::UnitX(), axis.normalized());
  map_manager_impl_->extractLocalMap(c, s, rotation, o, f);
}
void MapManager::getLocalPointcloud(const Eigen::Vector3d& c, const double& r, const double&,
                                    pcl::PointCloud<pcl::PointXYZI>& cloud, bool unknown) {
  map_manager_impl_->getLocalPointcloud(c, r, nullptr, cloud, unknown);
}
void MapManager::getLocalPointcloud(const Eigen::Vector3d& c, const double& r, const double&,
                                    const Eigen::Vector2d& z,
                                    pcl::PointCloud<pcl::PointXYZI>& cloud, bool unknown) {
  map_manager_impl_->getLocalPointcloud(c, r, &z, cloud, unknown);
}
void MapManager::resetMap() { map_manager_impl_->reset(); }
void MapManager::setRaycastingParams(bool n, double m) { map_manager_impl_->setRaycastingParams(n, m); }
void MapManager::setRobotRadius(double) {}
void MapManager::setBoxCheckMethod(int) {}
void MapManager::setLineCheckMethod(int) {}
