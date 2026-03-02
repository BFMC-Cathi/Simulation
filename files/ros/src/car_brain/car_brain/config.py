#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module — Team Cathı / BFMC
=========================================
Centralised, dataclass-based configuration for every tunable parameter
in the autonomous driving pipeline.  All values carry sensible defaults
for the BFMC Ignition Gazebo simulation.

Usage
-----
  from car_brain.config import DrivingConfig
  cfg = DrivingConfig()             # all defaults
  cfg = DrivingConfig(cruise_speed=0.3)  # override one field
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YOLO / Detection class names — MUST match your training labels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS_STOP_LINE: str = "stop_line"
CLASS_CROSSWALK: str = "crosswalk"
CLASS_STOP_SIGN: str = "stop_sign"
CLASS_LEFT_LANE: str = "left_lane"
CLASS_RIGHT_LANE: str = "right_lane"
CLASS_TRAFFIC_LIGHT: str = "traffic_light"
CLASS_PEDESTRIAN: str = "pedestrian"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model file search helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_MODEL_FILENAME = "yolov8s.pt"
_SEARCH_DIRS = [
    os.path.dirname(os.path.realpath(__file__)),
    os.path.dirname(os.path.abspath(__file__)),
    "/home/ros_dev/BFMC_workspace/files/ros/src/car_brain/car_brain",
]


def find_model_path(filename: str = _MODEL_FILENAME) -> str:
    """Return the first existing model path, or fall back to first candidate."""
    for d in _SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(_SEARCH_DIRS[0], filename)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Master configuration dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class DrivingConfig:
    """Every tunable knob in one place — easy to serialise / log."""

    # ── ROS 2 topics ────────────────────────────────────────────
    camera_topic: str = "/automobile/camera/image_raw"
    cmd_topic: str = "/automobile/command"
    yolo_vis_topic: str = "/camera/yolo_detections"

    # ── Model ───────────────────────────────────────────────────
    model_path: str = field(default_factory=find_model_path)
    confidence_threshold: float = 0.45

    # ── Cruise / motion ─────────────────────────────────────────
    cruise_speed: float = 0.2          # m/s
    slow_speed: float = 0.10           # m/s  (crosswalk / caution zones)
    stop_speed: float = 0.0            # m/s

    # ── PID steering ────────────────────────────────────────────
    steering_kp: float = 0.005         # proportional gain
    steering_ki: float = 0.0001        # integral gain
    steering_kd: float = 0.002         # derivative gain
    max_steering: float = 0.5          # rad/s clamp

    # ── Speed ramping (smooth acceleration) ─────────────────────
    max_accel: float = 0.5             # m/s² positive ramp
    max_decel: float = 1.0             # m/s² braking ramp (can be more aggressive)

    # ── Low-pass filter on steering ─────────────────────────────
    steering_alpha: float = 0.3        # 0→ignore new, 1→no filter

    # ── State machine ──────────────────────────────────────────
    debounce_frames: int = 3           # consecutive frames for transition
    stop_hold_sec: float = 3.0         # seconds to remain stopped at sign
    frame_timeout_sec: float = 1.0     # emergency stop if no frame received

    # ── Stop / crosswalk proximity thresholds (normalised y) ───
    stop_hard_threshold: float = 0.70
    stop_soft_threshold: float = 0.50
    crosswalk_threshold: float = 0.55

    # ── Traffic-light HSV ranges (H, S, V) ─────────────────────
    #    Tuned for the Ignition simulation's traffic-light emission
    tl_red_lower: List[int] = field(default_factory=lambda: [0, 100, 100])
    tl_red_upper: List[int] = field(default_factory=lambda: [10, 255, 255])
    tl_red_lower2: List[int] = field(default_factory=lambda: [160, 100, 100])
    tl_red_upper2: List[int] = field(default_factory=lambda: [180, 255, 255])
    tl_green_lower: List[int] = field(default_factory=lambda: [40, 80, 80])
    tl_green_upper: List[int] = field(default_factory=lambda: [90, 255, 255])
    tl_yellow_lower: List[int] = field(default_factory=lambda: [15, 100, 100])
    tl_yellow_upper: List[int] = field(default_factory=lambda: [35, 255, 255])
    tl_min_pixel_ratio: float = 0.05   # min ratio of coloured pixels in crop

    # ── Lane offset when only one boundary visible ──────────────
    single_lane_offset_ratio: float = 0.3  # fraction of image width

    # ── Control loop rate ───────────────────────────────────────
    control_rate_hz: float = 20.0      # Hz — decoupled from camera FPS

    # ── Visualisation ───────────────────────────────────────────
    publish_visualisation: bool = True
