cmake_minimum_required(VERSION 3.8)
project(traffic_light_plugin)

find_package(ament_cmake REQUIRED)
find_package(ignition-gazebo6 REQUIRED) # Uses 'fortress' version (gazebo6)
find_package(ignition-plugin1 REQUIRED)

add_library(TrafficLightPlugin SHARED src/TrafficLightPlugin.cpp)
target_link_libraries(TrafficLightPlugin
  ignition-gazebo6::core
  ignition-plugin1::register
)

install(TARGETS TrafficLightPlugin DESTINATION lib/${PROJECT_NAME})
ament_package()
