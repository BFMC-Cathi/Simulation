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
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=ros_dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG video,render $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc

WORKDIR /home/$USERNAME/BFMC_workspace
