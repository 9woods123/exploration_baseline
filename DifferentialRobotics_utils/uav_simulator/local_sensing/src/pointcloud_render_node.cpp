#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#ifdef LOCAL_SENSING_ENABLE_CUDA
#include "lidar_dda_cuda.h"
#endif

using namespace std;
using namespace Eigen;

ros::Publisher pub_cloud;
ros::Publisher pub_freespace_cloud;

sensor_msgs::PointCloud2 local_map_pcl;
sensor_msgs::PointCloud2 local_depth_pcl;

ros::Subscriber odom_sub;
ros::Subscriber global_map_sub, local_map_sub;

ros::Timer local_sensing_timer;

bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);

nav_msgs::Odometry _odom;

double sensing_horizon, sensing_rate, estimation_rate;
double horizontal_resolution_deg, vertical_resolution_deg;
double vertical_fov_min_deg, vertical_fov_max_deg, min_ray_length;
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution, _inv_resolution;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;

inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i& index) {
  Eigen::Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
  pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
  pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;

  return pt;
};

inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d& pt) {
  Eigen::Vector3i idx;
  idx(0) = std::min(std::max(int((pt(0) - _gl_xl) * _inv_resolution), 0),
                    _GLX_SIZE - 1);
  idx(1) = std::min(std::max(int((pt(1) - _gl_yl) * _inv_resolution), 0),
                    _GLY_SIZE - 1);
  idx(2) = std::min(std::max(int((pt(2) - _gl_zl) * _inv_resolution), 0),
                    _GLZ_SIZE - 1);

  return idx;
};

void rcvOdometryCallbck(const nav_msgs::Odometry& odom) {
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  _odom = odom;
}

pcl::PointCloud<pcl::PointXYZ> _local_map;
sensor_msgs::PointCloud2 _local_map_pcd;
vector<uint8_t> _occupancy;
vector<Eigen::Vector3d> _ray_directions_sensor;
bool use_cuda = true;
#ifdef LOCAL_SENSING_ENABLE_CUDA
CudaDdaContext* cuda_dda = nullptr;
#endif

// Simulated lidar sensor noise. Range noise grows with distance
// (constant + per-metre term) and, optionally, a fraction of returns is
// dropped to emulate missing/outlier measurements.
bool lidar_noise_enable = true;
double lidar_noise_stddev = 0.01;         // m, distance-independent term
double lidar_noise_stddev_per_m = 0.001;  // m/m, range-proportional term
double lidar_noise_dropout_prob = 0.0;    // [0,1], fraction of dropped returns
std::mt19937 lidar_rng(std::random_device{}());
inline double applyLidarNoise(double range) {
  if (!lidar_noise_enable) return range;
  if (lidar_noise_dropout_prob > 0.0) {
    std::uniform_real_distribution<double> drop(0.0, 1.0);
    if (drop(lidar_rng) < lidar_noise_dropout_prob) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  const double sigma = lidar_noise_stddev + lidar_noise_stddev_per_m * range;
  if (sigma <= 0.0) return range;
  std::normal_distribution<double> gaussian(0.0, sigma);
  return std::max(0.0, range + gaussian(lidar_rng));
}

inline bool isInside(const Eigen::Vector3i& idx) {
  return idx.x() >= 0 && idx.x() < _GLX_SIZE && idx.y() >= 0 &&
         idx.y() < _GLY_SIZE && idx.z() >= 0 && idx.z() < _GLZ_SIZE;
}

inline size_t occupancyAddress(const Eigen::Vector3i& idx) {
  return (static_cast<size_t>(idx.x()) * _GLY_SIZE + idx.y()) * _GLZ_SIZE +
         idx.z();
}

// Amanatides-Woo voxel traversal. Returns only the first occupied voxel along
// the ray, which gives the simulated lidar proper occlusion behavior.
bool castRayDda(const Eigen::Vector3d& origin,
                const Eigen::Vector3d& direction,
                double max_range,
                double* hit_range) {
  Eigen::Vector3i voxel;
  voxel.x() = static_cast<int>(std::floor((origin.x() - _gl_xl) * _inv_resolution));
  voxel.y() = static_cast<int>(std::floor((origin.y() - _gl_yl) * _inv_resolution));
  voxel.z() = static_cast<int>(std::floor((origin.z() - _gl_zl) * _inv_resolution));
  if (!isInside(voxel)) return false;

  Eigen::Vector3i step = Eigen::Vector3i::Zero();
  Eigen::Vector3d t_max;
  Eigen::Vector3d t_delta;
  const double inf = std::numeric_limits<double>::infinity();
  const double lower[3] = {_gl_xl, _gl_yl, _gl_zl};

  for (int axis = 0; axis < 3; ++axis) {
    const double component = direction[axis];
    if (component > 1e-12) {
      step[axis] = 1;
      const double boundary =
          lower[axis] + (static_cast<double>(voxel[axis]) + 1.0) * _resolution;
      t_max[axis] = (boundary - origin[axis]) / component;
      t_delta[axis] = _resolution / component;
    } else if (component < -1e-12) {
      step[axis] = -1;
      const double boundary =
          lower[axis] + static_cast<double>(voxel[axis]) * _resolution;
      t_max[axis] = (boundary - origin[axis]) / component;
      t_delta[axis] = -_resolution / component;
    } else {
      t_max[axis] = inf;
      t_delta[axis] = inf;
    }
  }

  double distance = 0.0;
  while (distance <= max_range && isInside(voxel)) {
    if (distance >= min_ray_length && _occupancy[occupancyAddress(voxel)]) {
      // Move a tiny amount inside the occupied voxel so the mapper reliably
      // assigns the endpoint to the hit cell instead of the preceding free cell.
      *hit_range = std::min(distance + 1e-3, max_range);
      return true;
    }

    const double next = std::min(t_max.x(), std::min(t_max.y(), t_max.z()));
    if (!std::isfinite(next) || next > max_range) break;

    // Advance all tied axes together. This avoids falsely hitting a voxel
    // which the ray only touches at an edge or corner.
    for (int axis = 0; axis < 3; ++axis) {
      if (t_max[axis] <= next + 1e-10) {
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];
      }
    }
    distance = next;
  }
  return false;
}

void rcvGlobalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  if (has_global_map) return;

  ROS_WARN("Global Pointcloud received..");

  pcl::PointCloud<pcl::PointXYZ> cloud_input;
  pcl::fromROSMsg(pointcloud_map, cloud_input);
  _occupancy.assign(static_cast<size_t>(_GLX_SIZE) * _GLY_SIZE * _GLZ_SIZE, 0);

  size_t occupied_voxels = 0;
  for (const pcl::PointXYZ& point : cloud_input.points) {
    Eigen::Vector3i idx;
    idx.x() = static_cast<int>(std::floor((point.x - _gl_xl) * _inv_resolution));
    idx.y() = static_cast<int>(std::floor((point.y - _gl_yl) * _inv_resolution));
    idx.z() = static_cast<int>(std::floor((point.z - _gl_zl) * _inv_resolution));
    if (!isInside(idx)) continue;
    uint8_t& cell = _occupancy[occupancyAddress(idx)];
    if (!cell) {
      cell = 1;
      ++occupied_voxels;
    }
  }

  has_global_map = true;
#ifdef LOCAL_SENSING_ENABLE_CUDA
  if (cuda_dda) {
    std::string error;
    if (!cudaDdaUploadOccupancy(cuda_dda, _occupancy, &error)) {
      ROS_ERROR("CUDA occupancy upload failed (%s); using CPU DDA", error.c_str());
      cudaDdaDestroy(cuda_dda);
      cuda_dda = nullptr;
    }
  }
#endif
  ROS_INFO("Built lidar occupancy grid: %zu occupied voxels at %.3f m",
           occupied_voxels, _resolution);
}

void renderSensedPoints(const ros::TimerEvent& event) {
  if (!has_global_map || !has_odom) return;

  Eigen::Quaterniond q;
  q.x() = _odom.pose.pose.orientation.x;
  q.y() = _odom.pose.pose.orientation.y;
  q.z() = _odom.pose.pose.orientation.z;
  q.w() = _odom.pose.pose.orientation.w;

  const Eigen::Matrix3d rot = q.toRotationMatrix();
  const Eigen::Vector3d origin(_odom.pose.pose.position.x,
                               _odom.pose.pose.position.y,
                               _odom.pose.pose.position.z);

  _local_map.points.clear();
  pcl::PointCloud<pcl::PointXYZ> freespace_cloud;
  vector<float> cuda_ranges;
  vector<uint8_t> cuda_hits;
  bool cuda_scan_valid = false;
#ifdef LOCAL_SENSING_ENABLE_CUDA
  if (cuda_dda) {
    const float cuda_origin[3] = {static_cast<float>(origin.x()),
                                  static_cast<float>(origin.y()),
                                  static_cast<float>(origin.z())};
    float cuda_rotation[9];
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
        cuda_rotation[3 * row + col] = static_cast<float>(rot(row, col));
    float elapsed_ms = 0.0f;
    std::string error;
    cuda_scan_valid = cudaDdaCast(
        cuda_dda, cuda_origin, cuda_rotation, static_cast<float>(min_ray_length),
        static_cast<float>(sensing_horizon), &cuda_ranges, &cuda_hits,
        &elapsed_ms, &error);
    if (!cuda_scan_valid) {
      ROS_ERROR("CUDA DDA scan failed (%s); switching to CPU DDA", error.c_str());
      cudaDdaDestroy(cuda_dda);
      cuda_dda = nullptr;
    } else {
      const size_t hit_count =
          std::count(cuda_hits.begin(), cuda_hits.end(), uint8_t{1});
      ROS_INFO_THROTTLE(2.0,
                        "CUDA DDA: %zu rays (%zu hits, %zu misses), kernel %.3f ms",
                        _ray_directions_sensor.size(), hit_count,
                        cuda_hits.size() - hit_count, elapsed_ms);
    }
  }
#endif

  for (size_t ray = 0; ray < _ray_directions_sensor.size(); ++ray) {
      const Eigen::Vector3d& direction_sensor = _ray_directions_sensor[ray];
      bool hit = false;
      double hit_range = sensing_horizon;
      if (cuda_scan_valid) {
        hit = cuda_hits[ray] != 0;
        hit_range = cuda_ranges[ray];
      } else {
        const Eigen::Vector3d direction_world = rot * direction_sensor;
        hit = castRayDda(origin, direction_world, sensing_horizon, &hit_range);
      }
      if (hit) {
        hit_range = applyLidarNoise(hit_range);
        if (!std::isfinite(hit_range)) {
          hit = false;  // dropped return, treated as free space
        } else {
          hit_range = std::min(hit_range, sensing_horizon);
        }
      }
      pcl::PointXYZ endpoint;
      if (hit) {
        endpoint.x = direction_sensor.x() * hit_range;
        endpoint.y = direction_sensor.y() * hit_range;
        endpoint.z = direction_sensor.z() * hit_range;
        _local_map.points.push_back(endpoint);
      } else {
        endpoint.x = direction_sensor.x() * sensing_horizon;
        endpoint.y = direction_sensor.y() * sensing_horizon;
        endpoint.z = direction_sensor.z() * sensing_horizon;
        freespace_cloud.points.push_back(endpoint);
      }
  }

  _local_map.width = _local_map.points.size();
  _local_map.height = 1;
  _local_map.is_dense = true;

  pcl::toROSMsg(_local_map, _local_map_pcd);
  _local_map_pcd.header.frame_id = "uav_sensor";
  _local_map_pcd.header.stamp = _odom.header.stamp;

  pub_cloud.publish(_local_map_pcd);

  freespace_cloud.width = freespace_cloud.points.size();
  freespace_cloud.height = 1;
  freespace_cloud.is_dense = true;
  sensor_msgs::PointCloud2 freespace_msg;
  pcl::toROSMsg(freespace_cloud, freespace_msg);
  freespace_msg.header.frame_id = "uav_sensor";
  freespace_msg.header.stamp = _odom.header.stamp;
  pub_freespace_cloud.publish(freespace_msg);
}

void rcvLocalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  // do nothing, fix later
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);
  nh.param("horizontal_resolution_deg", horizontal_resolution_deg, 5.0);
  nh.param("vertical_resolution_deg", vertical_resolution_deg, 5.0);
  nh.param("vertical_fov_min_deg", vertical_fov_min_deg, -30.0);
  nh.param("vertical_fov_max_deg", vertical_fov_max_deg, 30.0);
  nh.param("min_ray_length", min_ray_length, 0.25);
  nh.param("use_cuda", use_cuda, true);
  nh.param("lidar_noise_enable", lidar_noise_enable, true);
  nh.param("lidar_noise_stddev", lidar_noise_stddev, 0.01);
  nh.param("lidar_noise_stddev_per_m", lidar_noise_stddev_per_m, 0.001);
  nh.param("lidar_noise_dropout_prob", lidar_noise_dropout_prob, 0.0);
  ROS_ASSERT(sensing_rate > 0.0);
  ROS_ASSERT(horizontal_resolution_deg > 0.0);
  ROS_ASSERT(vertical_resolution_deg > 0.0);
  ROS_ASSERT(vertical_fov_max_deg >= vertical_fov_min_deg);

  nh.getParam("map/x_size", _x_size);
  nh.getParam("map/y_size", _y_size);
  nh.getParam("map/z_size", _z_size);
  nh.param("map/resolution", _resolution, 0.2);

  // subscribe point cloud
  global_map_sub = nh.subscribe("global_map", 1, rcvGlobalPointCloudCallBack);
  local_map_sub = nh.subscribe("local_map", 1, rcvLocalPointCloudCallBack);
  odom_sub = nh.subscribe("odometry", 50, rcvOdometryCallbck);

  // publisher depth image and color image
  pub_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("pcl_render_node/cloud", 10);
  pub_freespace_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("freespace_pointcloud", 10);

  const double sensing_duration = 1.0 / sensing_rate;

  local_sensing_timer =
      nh.createTimer(ros::Duration(sensing_duration), renderSensedPoints);

  _inv_resolution = 1.0 / _resolution;

  _gl_xl = -_x_size / 2.0;
  _gl_yl = -_y_size / 2.0;
  _gl_zl = 0.0;

  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  vector<float> cuda_directions;
  // Generate rays by integer index instead of repeatedly adding floating-point
  // angles. This guarantees the configured Mid-360 layout is exactly
  // 500 x 40 = 20,000 rays rather than occasionally gaining an endpoint ray.
  const int horizontal_ray_count =
      std::max(1, static_cast<int>(std::lround(360.0 /
                                               horizontal_resolution_deg)));
  const int vertical_ray_count = std::max(
      1, static_cast<int>(std::lround(
             (vertical_fov_max_deg - vertical_fov_min_deg) /
             vertical_resolution_deg)) +
             1);
  for (int pitch_index = 0; pitch_index < vertical_ray_count; ++pitch_index) {
    const double pitch =
        vertical_ray_count == 1
            ? 0.5 * (vertical_fov_min_deg + vertical_fov_max_deg)
            : vertical_fov_min_deg +
                  (vertical_fov_max_deg - vertical_fov_min_deg) * pitch_index /
                      (vertical_ray_count - 1);
    const double pitch_rad = pitch * M_PI / 180.0;
    for (int yaw_index = 0; yaw_index < horizontal_ray_count; ++yaw_index) {
      const double yaw =
          -180.0 + 360.0 * yaw_index / horizontal_ray_count;
      const double yaw_rad = yaw * M_PI / 180.0;
      _ray_directions_sensor.emplace_back(
          std::cos(pitch_rad) * std::cos(yaw_rad),
          std::cos(pitch_rad) * std::sin(yaw_rad), std::sin(pitch_rad));
      const Eigen::Vector3d& direction = _ray_directions_sensor.back();
      cuda_directions.push_back(static_cast<float>(direction.x()));
      cuda_directions.push_back(static_cast<float>(direction.y()));
      cuda_directions.push_back(static_cast<float>(direction.z()));
    }
  }

#ifdef LOCAL_SENSING_ENABLE_CUDA
  if (use_cuda) {
    std::string error;
    cuda_dda = cudaDdaCreate(_GLX_SIZE, _GLY_SIZE, _GLZ_SIZE,
                            static_cast<float>(_gl_xl),
                            static_cast<float>(_gl_yl),
                            static_cast<float>(_gl_zl),
                            static_cast<float>(_resolution), cuda_directions,
                            &error);
    if (cuda_dda) {
      ROS_INFO("CUDA 3D DDA enabled for %zu lidar rays",
               _ray_directions_sensor.size());
    } else {
      ROS_ERROR("CUDA initialization failed (%s); using CPU DDA", error.c_str());
    }
  } else {
    ROS_WARN("CUDA disabled by ~use_cuda; using CPU DDA");
  }
#else
  if (use_cuda) ROS_WARN("CUDA support was not compiled; using CPU DDA");
#endif

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }
#ifdef LOCAL_SENSING_ENABLE_CUDA
  cudaDdaDestroy(cuda_dda);
#endif
}
