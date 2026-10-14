PROJECT(local_sensing_node)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )

option(ENABLE_CUDA "Build the CUDA 3D DDA lidar" ON)

if(ENABLE_CUDA)
  # CUDA 10.1 does not understand Ada's sm_89 target. Ship forward-compatible
  # compute_75 PTX and let the installed NVIDIA driver JIT it for the GPU.
  set(CUDA_HOST_COMPILER /usr/bin/g++-8 CACHE FILEPATH
      "Host compiler used by nvcc" FORCE)
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-gencode;arch=compute_75,code=compute_75)

  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(Boost REQUIRED COMPONENTS system filesystem)

  find_package(catkin REQUIRED COMPONENTS
      roscpp roslib cmake_modules cv_bridge image_transport pcl_ros sensor_msgs geometry_msgs nav_msgs quadrotor_msgs dynamic_reconfigure)
  generate_dynamic_reconfigure_options(
    cfg/local_sensing_node.cfg
  )
  catkin_package(
      DEPENDS OpenCV Eigen Boost
      CATKIN_DEPENDS roscpp roslib image_transport pcl_ros
  #INCLUDE_DIRS include
      LIBRARIES lidar_dda_cuda
  )

  include_directories(
    SYSTEM 
    #include 
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
  )

  CUDA_ADD_LIBRARY( lidar_dda_cuda
      src/lidar_dda_cuda.cu
  )

  add_executable(
    pcl_render_node
    src/pointcloud_render_node.cpp
  )
  target_compile_definitions(pcl_render_node PRIVATE LOCAL_SENSING_ENABLE_CUDA=1)
  target_link_libraries( pcl_render_node
    lidar_dda_cuda
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
  )
else(ENABLE_CUDA)
  find_package(Eigen3 REQUIRED)
  find_package(catkin REQUIRED COMPONENTS
      roscpp roslib cmake_modules pcl_ros sensor_msgs geometry_msgs nav_msgs quadrotor_msgs)
  
  catkin_package(
      DEPENDS Eigen
      CATKIN_DEPENDS roscpp roslib pcl_ros
  )

  include_directories(
    SYSTEM 
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}.
  )

  add_executable(
    pcl_render_node
    src/pointcloud_render_node.cpp
  )

  target_link_libraries( pcl_render_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
  )
endif(ENABLE_CUDA)
