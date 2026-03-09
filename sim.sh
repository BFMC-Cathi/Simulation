#!/bin/bash
set -e

# ── 0. Fix NumPy (cv_bridge needs numpy<2) ───────────────────────
python3 -m pip install --quiet --user "numpy>=1.24,<2" --force-reinstall 2>/dev/null || true
echo "[pre-flight] numpy=$(python3 -c 'import numpy;print(numpy.__version__)')"

# ── 1. Move to the ROS workspace ─────────────────────────────────
cd /home/ros_dev/BFMC_workspace/files/ros || exit

# ── 2. Paths for Ignition / Gazebo ───────────────────────────────
MODELS=/home/ros_dev/BFMC_workspace/files/simulation/models_pkg
PLUGIN=/home/ros_dev/BFMC_workspace/files/ros/install/traffic_light_plugin/lib/traffic_light_plugin

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/ros/humble/lib

export IGN_GAZEBO_RESOURCE_PATH=${MODELS}
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=${PLUGIN}

export SDF_PATH=${MODELS}:${SDF_PATH:-}
export IGN_FILE_PATH=${MODELS}:${IGN_FILE_PATH:-}

# ── 3. Render engine — ogre2 (ogre1 causes malloc crash with AMD GPU)
export IGN_GAZEBO_RENDER_ENGINE_GUI=ogre2
export IGN_GAZEBO_RENDER_ENGINE_SERVER=ogre2

# ── 4. GPU / Mesa workarounds ────────────────────────────────────
if [ -d /dev/dri ]; then
    sudo chmod 666 /dev/dri/renderD* 2>/dev/null || true
fi

if ! test -r /dev/dri/renderD128 2>/dev/null; then
    echo "[WARN] /dev/dri not readable — enabling software rendering"
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE=3.3
else
    echo "[OK] GPU access: /dev/dri/renderD128 is readable"
fi
export QT_X11_NO_MITSHM=1

export MESA_GLSL_CACHE_DISABLE=1
export MALLOC_CHECK_=0

# ── 5. Clear Gazebo GUI cache ────────────────────────────────────
rm -rf ~/.ignition/gazebo/

# ── 6. Source ROS + workspace overlays ───────────────────────────
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Environment ready. Launching BFMC Competition World..."

# ── 7. Launch in background, then auto-unpause ──────────────────
ros2 launch car_brain sim_launch.py &
LAUNCH_PID=$!

# Wait for Gazebo world to be ready, then unpause
(
    echo "Waiting for Gazebo to start..."
    for i in $(seq 1 30); do
        sleep 2
        ign service -s /world/bfmc_competition/control \
            --reqtype ignition.msgs.WorldControl \
            --reptype ignition.msgs.Boolean \
            --timeout 2000 \
            --req 'pause: false' 2>/dev/null && {
                echo "[OK] Simulation unpaused after ~$((i*2))s"
                break
            }
    done
) &

# Bring launch back to foreground so Ctrl+C works
wait $LAUNCH_PID