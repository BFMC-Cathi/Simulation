# PID Tuning Guide — BFMC Autonomous Driving

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    perception_node                           │
│  ┌─────────────┐    ┌────────────────────────────────────┐   │
│  │  YOLOv8     │    │  Lane Detection (Classical CV)     │   │
│  │  (threaded) │    │  threshold → BEV → sliding window  │   │
│  │             │    │  → polynomial fit → CTE + heading  │   │
│  └──────┬──────┘    └────────────────┬───────────────────┘   │
│         │                            │                       │
│  /perception/objects          /perception/lane_state         │
│  (JSON: bboxes + TL colour)  (JSON: CTE + heading_error)    │
└─────────┼────────────────────────────┼───────────────────────┘
          │                            │
          ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  control_state_node                          │
│  ┌────────────────┐    ┌─────────────────────────────────┐   │
│  │  FSM (8 states)│    │  PID Controller                 │   │
│  │  INIT → LANE_  │    │  CTE PID + Heading PID → steer  │   │
│  │  FOLLOWING → …  │    │  Speed ramp → linear.x          │   │
│  └───────┬────────┘    └──────────────┬──────────────────┘   │
│          │  target_speed              │  angular.z           │
│          └────────────┬───────────────┘                      │
│                       ▼                                      │
│              /automobile/command (Twist)                      │
└──────────────────────────────────────────────────────────────┘
```

## ROS Topics

| Topic                      | Type                | Direction        | Content                          |
|---------------------------|---------------------|------------------|----------------------------------|
| `/automobile/camera/image_raw` | `sensor_msgs/Image` | Gazebo → perception_node | 640×480 RGB @ 30 Hz |
| `/perception/objects`     | `std_msgs/String`   | perception → control | JSON: YOLO detections + TL colour |
| `/perception/lane_state`  | `std_msgs/String`   | perception → control | JSON: CTE, heading_error, flags  |
| `/perception/debug_image` | `sensor_msgs/Image` | perception → rqt    | Annotated composite image        |
| `/automobile/command`     | `geometry_msgs/Twist` | control → Gazebo   | linear.x (speed), angular.z (steer) |

## FSM States

| State                  | Speed          | Description                                    |
|------------------------|----------------|------------------------------------------------|
| `INIT`                 | 0.0            | Waiting for first frame                        |
| `LANE_FOLLOWING`       | `cruise_speed` | Default: PID lane centring                     |
| `INTERSECTION_APPROACH`| `slow_speed`   | Stop sign or red light detected, decelerating  |
| `STOPPED`              | 0.0            | Full stop for `stop_hold_sec`, then resume     |
| `ROUNDABOUT_NAV`       | `slow_speed`   | Navigating the roundabout at reduced speed     |
| `PARKING`              | `slow_speed/2` | Parking manoeuvre                               |
| `HIGHWAY`              | `highway_speed`| Higher speed on the highway section            |
| `EMERGENCY_STOP`       | 0.0            | No camera frames → safe halt                   |

## PID Tuning Steps

### 1. Prepare the Environment

```bash
# Terminal 1: Launch the simulation
docker exec -it bfmc_sim bash
cd /home/ros_dev/BFMC_workspace/files/ros
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch car_brain sim_launch.py

# Terminal 2: Monitor topics
docker exec -it bfmc_sim bash
source /opt/ros/humble/setup.bash
source /home/ros_dev/BFMC_workspace/files/ros/install/setup.bash
ros2 topic echo /perception/lane_state

# Terminal 3: Watch debug image
# Use rqt_image_view on topic /perception/debug_image
```

### 2. Tune the CTE PID (steering_kp, steering_ki, steering_kd)

The CTE (Cross-Track Error) PID converts the pixel offset between the
car and the lane centre into a steering command.

#### Step 1: Start with P only
```bash
ros2 param set /control_state_node steering_kp 0.005
ros2 param set /control_state_node steering_ki 0.0
ros2 param set /control_state_node steering_kd 0.0
```

- **Too low Kp**: Car drifts to the side, slow correction.
- **Too high Kp**: Car oscillates left-right aggressively.
- **Just right**: Car gently corrects towards centre with slight overshoot.

#### Step 2: Add D to dampen oscillation
```bash
ros2 param set /control_state_node steering_kd 0.002
```

- Increase Kd until overshoot disappears.
- Too much Kd makes the car sluggish to respond.

#### Step 3: Add I to eliminate steady-state error
```bash
ros2 param set /control_state_node steering_ki 0.0001
```

- The integral term corrects for persistent small offsets (e.g., camera
  misalignment).
- Keep Ki very small — large values cause windup and oscillation.

### 3. Tune Speed Parameters

```bash
# Straight sections
ros2 param set /control_state_node cruise_speed 0.5

# Reduce speed in curves (the node does this via FSM for signs,
# but you can lower cruise_speed globally for testing)
ros2 param set /control_state_node cruise_speed 0.3

# Highway section
ros2 param set /control_state_node highway_speed 0.8
```

### 4. Tune Lane Detection (perception_node)

```bash
# If lane lines aren't being detected:
ros2 param set /perception_node white_threshold 180  # lower = more sensitive

# If noise is detected as lane:
ros2 param set /perception_node white_threshold 220  # higher = stricter

# Adjust BEV region (if car sees too far/near):
# Edit bev_src_points in config.py and rebuild
```

### 5. Tune the EMA Filter (steering_alpha)

```bash
# More responsive (but noisier):
ros2 param set /control_state_node steering_alpha 0.5

# Smoother (but slower to react):
ros2 param set /control_state_node steering_alpha 0.2
```

### 6. Tune FSM Debounce

```bash
# More false-positive rejections (slower reaction):
ros2 param set /control_state_node debounce_frames 5

# Faster reaction (more false positives):
ros2 param set /control_state_node debounce_frames 2
```

## Quick Reference: All Parameters

### perception_node
| Parameter               | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `model_path`           | `<pkg>/yolov8s.pt` | Path to YOLO .pt model          |
| `camera_topic`         | `/automobile/camera/image_raw` | Camera subscription   |
| `confidence_threshold` | `0.45`  | YOLO confidence threshold                |
| `white_threshold`      | `200`   | Binary threshold for lane lines          |
| `sliding_window_count` | `9`     | Number of sliding windows                |
| `sliding_window_margin`| `80`    | Half-width of each window (px)           |
| `sliding_window_min_pix`| `50`   | Min pixels to re-centre window           |
| `lane_history_frames`  | `5`     | Frames to average polynomial over        |
| `missing_lane_timeout_sec`| `2.0`| Fallback duration for missing lanes      |
| `control_rate_hz`      | `20.0`  | Processing rate                          |
| `publish_visualisation`| `true`  | Publish debug image                      |

### control_state_node
| Parameter           | Default | Description                               |
|--------------------|---------|-------------------------------------------|
| `cruise_speed`     | `0.5`   | Default lane-following speed (m/s)        |
| `slow_speed`       | `0.25`  | Speed for intersections / roundabouts     |
| `highway_speed`    | `0.8`   | Speed on highway                          |
| `steering_kp`      | `0.008` | PID proportional gain                     |
| `steering_ki`      | `0.0001`| PID integral gain                         |
| `steering_kd`      | `0.004` | PID derivative gain                       |
| `max_steering`     | `0.8`   | Maximum angular.z (rad/s)                 |
| `steering_alpha`   | `0.3`   | EMA filter coefficient (0→smooth, 1→raw)  |
| `max_accel`        | `1.0`   | Speed ramp-up rate (m/s²)                 |
| `max_decel`        | `2.0`   | Speed ramp-down rate (m/s²)               |
| `stop_hold_sec`    | `3.0`   | How long to hold at stop sign (s)         |
| `debounce_frames`  | `3`     | Frames before FSM transition fires        |
| `frame_timeout_sec`| `2.0`   | Emergency stop if no data (s)             |
| `image_height`     | `480`   | Image height for proximity calculations   |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Car doesn't move | INIT state stuck / no camera | Check `/automobile/camera/image_raw` |
| Car oscillates wildly | Kp too high | Lower `steering_kp` |
| Car drifts off lane | Kp too low | Increase `steering_kp` |
| Car jerky/stuttering | Kd too high or alpha too high | Lower `steering_kd` or `steering_alpha` |
| Lanes not detected | Threshold too high | Lower `white_threshold` |
| False lane detections | Threshold too low | Raise `white_threshold` |
| Stop sign not detected | YOLO model issue | Check model path & `confidence_threshold` |
| Car doesn't stop at signs | Debounce too high | Lower `debounce_frames` |
