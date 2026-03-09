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

# Pin numpy<2 FIRST, then install ultralytics + opencv with --no-deps to prevent
# numpy 2.x from sneaking back in. Finally install remaining ultralytics deps.
RUN python3 -m pip install --default-timeout=1000 --retries 10 --no-cache-dir \
    "numpy>=1.24,<2" \
    && python3 -m pip install --default-timeout=1000 --retries 10 --no-cache-dir \
    --ignore-installed sympy \
    "opencv-python-headless<4.11" \
    ultralytics \
    && python3 -m pip install --default-timeout=1000 --retries 10 --no-cache-dir \
    "numpy>=1.24,<2" --force-reinstall

# 1c. Install mesa software rendering utils (for headless / weak GPU fallback)
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgl1-mesa-dri libglx-mesa0 libegl-mesa0 \
    && sudo rm -rf /var/lib/apt/lists/*

# 2. Create a non-root user
ARG USERNAME=ros_dev
ARG USER_UID=1000
ARG USER_GID=1000

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