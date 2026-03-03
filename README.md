# BFMC Simulation Environment

A comprehensive ROS 2 Gazebo-based simulation for BFMC (Bosch Future Mobility Challenge) autonomous vehicle competition. This project provides a containerized environment for developing and testing autonomous driving algorithms.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Running the Simulation](#running-the-simulation)
- [Common Commands](#common-commands)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## 📦 Requirements

Before getting started, ensure you have:

- **Docker** (version 20.10+)
- **Docker Compose** (optional, for easier management)
- **Git**
- **X11 server** (for GUI support on Linux)
- Adequate disk space (~5GB for Docker image and dependencies)

### System Requirements

- Linux system with Docker support
- GPU recommended for smooth simulation performance
- Minimum 4GB RAM (8GB+ recommended)

## 🚀 Quick Start

If you already have the Docker image built, here's the fastest way to get running:

```bash
# 1. Allow Docker to access your X11 display
xhost +local:docker

# 2. Start the existing container
docker start bfmc_sim

# 3. Enter the container
docker exec -it bfmc_sim bash

# 4. Inside the container, run the simulation
./sim.sh
```

## 📝 Detailed Setup

### 1. Clone the Project

```bash
git clone https://github.com/BFMC-Cathi/Simulation
cd Simulation
```

### 2. Build the Docker Image

The Dockerfile includes all dependencies for ROS 2 Humble, Gazebo, and the BFMC environment.

```bash
docker build -t team_cathi/bfmc_env .
```

**Expected output:** Docker image `team_cathi/bfmc_env` created successfully (this may take 5-10 minutes)

### 3. Run the Container for the First Time

First, allow Docker access to your X11 display for GUI rendering:

```bash
xhost +local:docker
```

Then, create and start the container:

```bash
docker run -it \
  --name bfmc_sim \
  --network=bridge \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --device /dev/dri:/dev/dri \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume $(pwd):/home/ros_dev/BFMC_workspace \
  team_cathi/bfmc_env bash
```

**Explanation of flags:**
- `-it`: Interactive terminal mode
- `--name bfmc_sim`: Names the container for easy reference
- `--env DISPLAY=$DISPLAY`: Enables GUI rendering
- `--device /dev/dri:/dev/dri`: Enables GPU acceleration
- `--volume`: Mounts your project folder into the container

## 🎮 Running the Simulation

Once inside the container, run:

```bash
# Execute the simulation script
./sim.sh
```

This script will:
1. Set up the ROS 2 environment variables
2. Configure Gazebo resource paths
3. Load the traffic light plugin
4. Launch the BFMC Competition World

**Expected behavior:** Gazebo window opens with the simulated track and vehicle.

## 🔧 Common Commands

### Container Management

```bash
# Start an existing container
docker start bfmc_sim

# Enter a running container
docker exec -it bfmc_sim bash

# Stop the container
docker stop bfmc_sim

# Remove the container (if needed)
docker rm bfmc_sim

# View running containers
docker ps

# View all containers (including stopped ones)
docker ps -a
```

### Inside the Container (ROS 2 Commands)

```bash
# Navigate to workspace
cd /home/ros_dev/BFMC_workspace/files/ros

# Source the ROS environment (already done in .bashrc)
source /opt/ros/humble/setup.bash
source install/setup.bash

# Build the ROS packages (if you made changes)
cd /home/ros_dev/BFMC_workspace/files/ros
colcon build

# Run the simulation
./sim.sh

# List available ROS topics
ros2 topic list

# Echo a specific topic (for debugging)
ros2 topic echo /topic_name

# View the ROS node graph
ros2 node list
```

### Simulation Troubleshooting Commands

```bash
# Clear Gazebo cache (useful if you see texture glitches)
rm -rf ~/.ignition/gazebo/

# Reset library paths before launching
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/humble/lib
export IGN_GAZEBO_RESOURCE_PATH=/home/ros_dev/BFMC_workspace/files/simulation/models_pkg
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/home/ros_dev/BFMC_workspace/files/ros/install/traffic_light_plugin/lib/traffic_light_plugin

# Set the rendering engine
export IGN_GAZEBO_RENDER_ENGINE_GUI=ogre
```

## 📂 Project Structure

```
Simulation/
├── Dockerfile                 # Docker configuration
├── sim.sh                      # Main simulation launch script
├── README.md                   # This file
├── files/
│   ├── ros/                    # ROS 2 workspace
│   │   ├── src/               # Source code
│   │   │   ├── car_brain/     # Vehicle control package
│   │   │   └── traffic_light_plugin/  # Traffic light simulation
│   │   ├── build/             # Build artifacts
│   │   └── install/           # Installed packages
│   └── simulation/            # Gazebo simulation assets
│       ├── world.sdf          # World definition
│       └── models_pkg/        # 3D models for tracks and objects
```

## 🐛 Troubleshooting

### Problem: GUI not displaying

**Solution:**
```bash
# Restart the X11 authorization
xhost +local:docker

# Then start/restart the container
docker stop bfmc_sim
docker start bfmc_sim
docker exec -it bfmc_sim bash
```

### Problem: "Permission denied" when running sim.sh

**Solution:**
```bash
# Make the script executable
chmod +x sim.sh

# Then run it
./sim.sh
```

### Problem: Gazebo shows texture errors or models not loading

**Solution:**
```bash
# Clear the Gazebo cache
rm -rf ~/.ignition/gazebo/

# Verify resource paths are set
echo $IGN_GAZEBO_RESOURCE_PATH
```

### Problem: GPU acceleration not working

**Solution:**
```bash
# Check if GPU is accessible
glxinfo | grep "OpenGL"

# Alternatively, try software rendering
export IGN_GAZEBO_RENDER_ENGINE_GUI=ogre2
```

### Problem: "Package not found" errors

**Solution:**
```bash
# Rebuild the packages
cd /home/ros_dev/BFMC_workspace/files/ros
colcon build --symlink-install

# Make sure environment is sourced
source install/setup.bash
```
**Happy simulating! 🚗**
