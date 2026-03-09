#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Team Cathı – BFMC Full Stack Startup Script
# ═══════════════════════════════════════════════════════════════════
#  This script does everything in order:
#    1. Builds the Docker image (if needed) 
#    2. Starts the container with GPU + display forwarding
#    3. Builds the ROS 2 workspace inside the container
#    4. Launches the Gazebo simulation + ROS bridge
#    5. Launches the YOLOv8 autonomous driving node
#
#  Usage:
#    chmod +x start.sh
#    ./start.sh              # full startup (sim + autonomy)
#    ./start.sh sim          # only simulation (no autonomy node)
#    ./start.sh autonomy     # only autonomy node (attach to running container)
#    ./start.sh build        # only build the workspace
#    ./start.sh shell        # just open a bash shell in the container
# ═══════════════════════════════════════════════════════════════════

set -e  # exit on any error

# ── Configuration ────────────────────────────────────────────────
IMAGE_NAME="team_cathi/bfmc_env"
CONTAINER_NAME="bfmc_sim"
WORKSPACE_HOST="$(cd "$(dirname "$0")" && pwd)"   # repo root
WORKSPACE_CONTAINER="/home/ros_dev/BFMC_workspace"
ROS_WS="${WORKSPACE_CONTAINER}/files/ros"

# Colours for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

# ── Helper functions ─────────────────────────────────────────────
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR ]${NC}  $*"; }

# ═════════════════════════════════════════════════════════════════
#  STEP 1 — Allow X11 display forwarding (so Gazebo GUI shows up)
# ═════════════════════════════════════════════════════════════════
setup_display() {
    info "Allowing X11 connections for Docker …"
    xhost +local:docker > /dev/null 2>&1 || warn "xhost not found – GUI may not work"
    ok "Display forwarding ready (DISPLAY=${DISPLAY})"
}

# ═════════════════════════════════════════════════════════════════
#  STEP 2 — Build Docker image (only if it doesn't exist)
# ═════════════════════════════════════════════════════════════════
build_image() {
    if docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
        ok "Docker image '${IMAGE_NAME}' already exists — skipping build."
    else
        info "Building Docker image '${IMAGE_NAME}' …"
        docker build -t "${IMAGE_NAME}" "${WORKSPACE_HOST}"
        ok "Docker image built."
    fi
}

# ═════════════════════════════════════════════════════════════════
#  STEP 3 — Start (or re-use) the Docker container
# ═════════════════════════════════════════════════════════════════
start_container() {
    # Check if the container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        # Container exists — is it running?
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            ok "Container '${CONTAINER_NAME}' is already running."
        else
            info "Starting existing container '${CONTAINER_NAME}' …"
            docker start "${CONTAINER_NAME}"
            ok "Container started."
        fi
    else
        info "Creating and starting container '${CONTAINER_NAME}' …"
        docker run -dit \
            --name "${CONTAINER_NAME}" \
            --privileged \
            --net=host \
            --ipc=host \
            --pid=host \
            -e DISPLAY="${DISPLAY}" \
            -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
            -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
            -e QT_X11_NO_MITSHM=1 \
            -e IGN_GAZEBO_RENDER_ENGINE_GUI=ogre2 \
            -e IGN_GAZEBO_RENDER_ENGINE_SERVER=ogre2 \
            -e IGN_GAZEBO_RESOURCE_PATH=${WORKSPACE_CONTAINER}/files/simulation/models_pkg \
            -e SDF_PATH=${WORKSPACE_CONTAINER}/files/simulation/models_pkg \
            -e IGN_FILE_PATH=${WORKSPACE_CONTAINER}/files/simulation/models_pkg \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -v "${WORKSPACE_HOST}:${WORKSPACE_CONTAINER}:rw" \
            -v /dev/dri:/dev/dri \
            "${IMAGE_NAME}" \
            bash
        ok "Container created and started."
    fi
}

# ═════════════════════════════════════════════════════════════════
#  STEP 4 — Build the ROS 2 workspace inside the container
# ═════════════════════════════════════════════════════════════════
build_workspace() {
    info "Installing Python dependencies (ultralytics, opencv) …"
    docker exec -it "${CONTAINER_NAME}" bash -c "
        sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip ros-humble-cv-bridge > /dev/null 2>&1
        # Install ultralytics first (it may pull numpy 2.x)
        python3 -m pip install --quiet --no-cache-dir ultralytics 2>&1 | tail -3
        # Install opencv-python-headless WITHOUT deps so it won't upgrade numpy
        python3 -m pip install --quiet --no-cache-dir --no-deps 'opencv-python-headless<4.11' 2>&1 | tail -3
        # Force numpy<2 LAST so cv_bridge works
        python3 -m pip install --quiet --no-cache-dir 'numpy>=1.24,<2' --force-reinstall 2>&1 | tail -3
        echo \"numpy=\$(python3 -c 'import numpy;print(numpy.__version__)')\"  
    "
    ok "Python dependencies ready."

    info "Building ROS 2 workspace (colcon build) …"
    docker exec -it "${CONTAINER_NAME}" bash -c "
        source /opt/ros/humble/setup.bash && \
        cd ${ROS_WS} && \
        colcon build --symlink-install 2>&1
    "
    ok "ROS 2 workspace built successfully."
}

# ═════════════════════════════════════════════════════════════════
#  STEP 5 — Launch Gazebo simulation + ROS bridge
# ═════════════════════════════════════════════════════════════════
launch_simulation() {
    info "Launching Gazebo simulation via sim.sh …"
    docker exec -dit "${CONTAINER_NAME}" bash -c "
        cd ${WORKSPACE_CONTAINER} && bash sim.sh 2>&1
    "
    ok "Simulation launched in background (sim.sh handles unpause)."
    info "Waiting 15 seconds for Gazebo to initialise + unpause …"
    sleep 15
}

# ═════════════════════════════════════════════════════════════════
#  STEP 6 — Launch the YOLOv8 autonomous driving node
# ═════════════════════════════════════════════════════════════════
launch_autonomy() {
    info "Launching YOLOv8 driving node …"
    docker exec -it "${CONTAINER_NAME}" bash -c "
        source /opt/ros/humble/setup.bash
        source ${ROS_WS}/install/setup.bash

        echo '──────────────────────────────────────────────'
        echo '  YOLOv8 Driving Node — Team Cathı / BFMC'
        echo '  Press Ctrl+C to stop'
        echo '──────────────────────────────────────────────'

        ros2 run car_brain yolov8_driving_node 2>&1
    "
}

# ═════════════════════════════════════════════════════════════════
#  MAIN — parse arguments and run
# ═════════════════════════════════════════════════════════════════
MODE="${1:-full}"   # default = full startup

case "${MODE}" in
    full)
        echo ""
        echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
        echo -e "${CYAN}  Team Cathı — BFMC Full Stack Startup${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
        echo ""
        setup_display
        build_image
        start_container
        build_workspace
        launch_simulation
        launch_autonomy
        ;;
    sim)
        info "Mode: Simulation only"
        setup_display
        build_image
        start_container
        build_workspace
        launch_simulation
        ok "Simulation is running. Attach with: ./start.sh autonomy"
        ;;
    autonomy)
        info "Mode: Autonomy node only (attaching to running container)"
        launch_autonomy
        ;;
    build)
        info "Mode: Build workspace only"
        start_container
        build_workspace
        ;;
    shell)
        info "Mode: Interactive shell"
        setup_display
        build_image
        start_container
        docker exec -it "${CONTAINER_NAME}" bash -c "
            source /opt/ros/humble/setup.bash && \
            source ${ROS_WS}/install/setup.bash 2>/dev/null; \
            bash
        "
        ;;
    stop)
        info "Stopping container '${CONTAINER_NAME}' …"
        docker stop "${CONTAINER_NAME}" 2>/dev/null && ok "Stopped." || warn "Not running."
        ;;
    *)
        echo "Usage: $0 {full|sim|autonomy|build|shell|stop}"
        echo ""
        echo "  full      — Build + Simulation + Autonomy (default)"
        echo "  sim       — Build + Simulation only"
        echo "  autonomy  — Launch driving node (container must be running)"
        echo "  build     — Build ROS workspace only"
        echo "  shell     — Open interactive bash in container"
        echo "  stop      — Stop the container"
        exit 1
        ;;
esac

echo ""
ok "Done."
