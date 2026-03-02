FROM osrf/ros:humble-desktop-full

# 1. Install basic tools + sudo
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    nano \
    vim \
    mesa-utils \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-pip \
    ros-humble-ros-gz \
    ros-humble-cv-bridge \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 1b. Install Python ML / CV packages
RUN python3 -m pip install --upgrade pip

# Install numpy + headless OpenCV first (smaller, no GUI libs needed in container)
RUN python3 -m pip install --default-timeout=1000 --retries 10 --no-cache-dir \
    numpy opencv-python-headless

# Install ultralytics (--retries handles transient network timeouts on large deps)
RUN python3 -m pip install --default-timeout=1000 --retries 10 --no-cache-dir \
    ultralytics

# 2. Create a non-root user
ARG USERNAME=ros_dev
ARG USER_UID=1000
ARG USER_GID=1000

# --- FIX IS HERE ---
# We use 'groupadd -f -r render' to ensure the group exists before assigning it
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && groupadd -f -r render \
    && usermod -aG video,render $USERNAME

# 3. Switch to the new user
USER $USERNAME

# 4. Source ROS automatically for this user
RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc

# 5. Set the working directory to the user's home
WORKDIR /home/$USERNAME/BFMC_workspace
