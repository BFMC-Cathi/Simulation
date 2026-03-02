#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module — Team Cathı / BFMC
==========================================
Central configuration dataclass and YOLO class-name constants.

Every tunable parameter lives here so that nodes can declare them as
ROS 2 parameters and override from launch files or the command line.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YOLO class-name constants (must match your .pt model's training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASS_STOP_SIGN = "stop sign"
CLASS_STOP_LINE = "stop line"
CLASS_CROSSWALK = "crosswalk"
CLASS_PEDESTRIAN = "pedestrian"
CLASS_TRAFFIC_LIGHT = "traffic light"
CLASS_LEFT_LANE = "left lane"
CLASS_RIGHT_LANE = "right lane"
CLASS_ROUNDABOUT_SIGN = "roundabout"
CLASS_PARKING_SIGN = "parking"
CLASS_HIGHWAY_ENTRY = "highway entry"
CLASS_HIGHWAY_EXIT = "highway exit"
CLASS_ONEWAY_SIGN = "oneway"
CLASS_PRIORITY_SIGN = "priority"
CLASS_PROHIBITED_SIGN = "no entry"
CLASS_ROADBLOCK = "roadblock"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main configuration dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class DrivingConfig:
    """
    All tunable parameters for the autonomous driving pipeline.
    Sensible defaults for the BFMC Gazebo simulation (640×480 camera).
    """

    # ── YOLO ────────────────────────────────────────────────────
    model_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yolov8s.pt"
    )
    confidence_threshold: float = 0.45

    # ── ROS topics ──────────────────────────────────────────────
    camera_topic: str = "/automobile/camera/image_raw"
    cmd_topic: str = "/automobile/command"
    yolo_vis_topic: str = "/perception/debug_image"
    objects_topic: str = "/perception/objects"
    lane_state_topic: str = "/perception/lane_state"

    # ── Camera / image ──────────────────────────────────────────
    image_width: int = 640
    image_height: int = 480

    # ── Lane detection — BEV perspective transform ──────────────
    # Source quad (pixel coords on the original 640×480 image)
    # Calibrated from simulation: lane lines at row 288 are at x~270/625,
    # at row 456 they span x~0/638  →  trapezoid below:
    bev_src_points: List[List[int]] = field(default_factory=lambda: [
        [220, 280],   # top-left  (just inside left lane @ ~60% height)
        [420, 280],   # top-right (just inside right lane @ ~60% height)
        [640, 460],   # bottom-right
        [0,   460],   # bottom-left
    ])
    # Destination quad (in the warped bird's-eye view)
    bev_dst_points: List[List[int]] = field(default_factory=lambda: [
        [100, 0],
        [540, 0],
        [540, 480],
        [100, 480],
    ])
    bev_width: int = 640
    bev_height: int = 480

    # ── Lane detection — thresholds ─────────────────────────────
    white_threshold: int = 160           # binary threshold for white lines (lines are ~229)
    sliding_window_count: int = 12       # number of sliding windows
    sliding_window_margin: int = 100     # half-width of each window (px)
    sliding_window_min_pix: int = 30     # min pixels to re-centre window
    lane_history_frames: int = 5         # frames to average polynomial over

    # ── Control — PID ───────────────────────────────────────────
    steering_kp: float = 0.8
    steering_ki: float = 0.02
    steering_kd: float = 0.3
    max_steering: float = 1.0
    steering_alpha: float = 0.25         # EMA low-pass filter coefficient (lower = smoother)

    # ── Control — speed ─────────────────────────────────────────
    cruise_speed: float = 0.35
    slow_speed: float = 0.2
    stop_speed: float = 0.0
    highway_speed: float = 0.6
    max_accel: float = 0.8               # m/s² ramp-up
    max_decel: float = 2.0               # m/s² ramp-down
    control_rate_hz: float = 30.0

    # ── FSM ─────────────────────────────────────────────────────
    stop_hold_sec: float = 3.0
    debounce_frames: int = 3
    frame_timeout_sec: float = 2.0

    # ── FSM — proximity thresholds (normalised y2 / img_height) ─
    stop_soft_threshold: float = 0.40
    stop_hard_threshold: float = 0.55
    crosswalk_threshold: float = 0.45
    roundabout_threshold: float = 0.40
    parking_threshold: float = 0.45
    single_lane_offset_ratio: float = 0.30

    # ── Traffic-light HSV ranges ────────────────────────────────
    tl_min_pixel_ratio: float = 0.02
    tl_red_lower: List[int] = field(default_factory=lambda: [0, 100, 100])
    tl_red_upper: List[int] = field(default_factory=lambda: [10, 255, 255])
    tl_red_lower2: List[int] = field(default_factory=lambda: [160, 100, 100])
    tl_red_upper2: List[int] = field(default_factory=lambda: [180, 255, 255])
    tl_green_lower: List[int] = field(default_factory=lambda: [40, 50, 50])
    tl_green_upper: List[int] = field(default_factory=lambda: [90, 255, 255])
    tl_yellow_lower: List[int] = field(default_factory=lambda: [15, 100, 100])
    tl_yellow_upper: List[int] = field(default_factory=lambda: [35, 255, 255])

    # ── Visualisation ───────────────────────────────────────────
    publish_visualisation: bool = True

    # ── Missing lanes fallback ──────────────────────────────────
    missing_lane_timeout_sec: float = 2.0  # use last-known fit for this long
