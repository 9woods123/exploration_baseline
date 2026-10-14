cmake_minimum_required(VERSION 3.0.2)
project(raycast_static_map)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES raycast_mapper
)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_library(raycast_mapper
  src/raycast_mapper.cpp
  src/object_tracker.cpp
)
target_compile_features(raycast_mapper PUBLIC cxx_std_14)
target_link_libraries(raycast_mapper ${catkin_LIBRARIES})

install(TARGETS raycast_mapper
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_raycast_mapper test/test_raycast_mapper.cpp)
  if(TARGET test_raycast_mapper)
    target_link_libraries(test_raycast_mapper raycast_mapper ${catkin_LIBRARIES})
  endif()
endif()
