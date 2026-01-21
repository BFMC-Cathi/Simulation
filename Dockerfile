FROM osrf/ros:humble-desktop-full

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    nano \
    vim \
    mesa-utils \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    ros-humble-ros-gz \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /root/BFMC_workspace
