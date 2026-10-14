# OmniPlanner + DifferentialRobotics UAV simulator demo

Build and run from `explore_last_ws`:

```bash
catkin build gbplanner pci_general so3_quadrotor_simulator so3_control \
  mockamap local_sensing_node omniplanner_uav_sim_demo
source devel/setup.bash
roslaunch omniplanner_uav_sim_demo demo.launch
```

The default launch opens a ready-to-use RViz view for a deterministic
50 x 50 x 5 m, 16 x 16-cell recursive-division maze with a solid ground
surface. It shows SO(3) UAV odometry,
`/lidar/points`, the incrementally accumulated sparse occupied map, planning
graphs, selected path, the latest planned trajectory in green, and the full
executed trajectory in blue. The complete `/mock_map`
ground truth is available as `Ground Truth Maze (optional)` but is disabled by
default, so the explored map visibly grows as the UAV flies. No RViz topic
setup is required.

For a headless smoke test, pass `rviz:=false`. Automatic exploration starts
after odometry, the simulated lidar, and the PCI service are available. The
main data path is:

```text
Lightweight SO(3) simulator /sim/odom
  -> local_sensing_node /lidar/points (PointCloud2, uav_sensor frame)
  -> GBPlanner -> explore_last_ws/src/raycast_static_map (hash occupancy + ESDF)
  -> PCI /command/trajectory -> /position_cmd -> SO(3) controller
```

The local sensing node also publishes `/lidar/freespace_points`. The embedded
mapper integrates these no-return rays as free space without creating false
surfaces at maximum range. It publishes the incrementally growing map on both
`/gbplanner_node/surface_pointcloud` and the compatibility MarkerArray topic
`/gbplanner_node/occupied_nodes`.

The simulated lidar uses a CUDA Amanatides-Woo 3D DDA kernel over the 0.25 m
occupancy grid. The grid and sensor-frame ray directions remain resident on
the GPU; one CUDA thread traverses one ray and returns only the first occupied
voxel. First hits are published on `/lidar/points`; only rays with no hit
contribute a max-range endpoint to `/lidar/freespace_points`. The default scan
uses a simplified Livox Mid-360 layout: 360-degree horizontal and -7 to +52
degree vertical field of view, 20,000 regularly distributed rays per frame,
15 m range, and 10 Hz output (200,000 rays/s). It reproduces the Mid-360 field
of view and point count, but not its non-repetitive scan trajectory or noise.

CUDA is enabled by default. If a CUDA device or kernel call is unavailable,
the node reports the error and automatically falls back to the equivalent CPU
DDA. The fallback can also be selected explicitly with:

```bash
roslaunch omniplanner_uav_sim_demo demo.launch use_cuda:=false
```

Set `auto_start:=false` to start exploration manually with:

```bash
rosservice call /planner_control_interface/std_srvs/automatic_planning
```
