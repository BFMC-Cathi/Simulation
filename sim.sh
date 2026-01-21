#!/bin/bash
# Move to the workspace directory first
cd /home/ros_dev/BFMC_workspace/files/ros || exit

# Reset Environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/humble/lib
export IGN_GAZEBO_RESOURCE_PATH=/home/ros_dev/BFMC_workspace/files/simulation/models_pkg

# --- ADD THIS NEW LINE ---
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/home/ros_dev/BFMC_workspace/files/ros/install/traffic_light_plugin/lib/traffic_light_plugin
# -------------------------

# Use the engine that worked for your car/track
export IGN_GAZEBO_RENDER_ENGINE_GUI=ogre

# Clear cache to prevent sign texture glitches
rm -rf ~/.ignition/gazebo/

# Source and Launch
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Environment ready. Launching BFMC Competition World..."
ros2 launch car_brain sim_launch.py
