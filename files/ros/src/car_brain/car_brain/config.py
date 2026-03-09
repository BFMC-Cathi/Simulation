#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module — Team Cathı / BFMC
==========================================
Central configuration dataclass and YOLO class-name constants.

Every tunable parameter lives here so that nodes can declare them as
ROS 2 parameters and override from launch files or the command line.

Adapted from VROOM-BFMC-Simulator for Ignition Gazebo + new trackgraph.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YOLO class-name constants
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


@dataclass
class DrivingConfig:
    """All tunable parameters for the autonomous driving pipeline."""

    # ── YOLO ────────────────────────────────────────────────────
    model_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yolov8s.pt"
    )
    confidence_threshold: float = 0.45

    # ── Track graph ─────────────────────────────────────────────
    track_graph_path: str = "/home/ros_dev/BFMC_workspace/trackgraph.graphml"
    nav_source_node: str = "86"
    nav_target_node: str = "274"

    # ── ROS topics ──────────────────────────────────────────────
    camera_topic: str = "/automobile/camera/image_raw"
    imu_topic: str = "/automobile/IMU"
    odom_topic: str = "/model/automobile/odometry"
    cmd_topic: str = "/automobile/command"
    yolo_vis_topic: str = "/perception/debug_image"
    objects_topic: str = "/perception/objects"
    lane_state_topic: str = "/perception/lane_state"

    # ── Camera / image ──────────────────────────────────────────
    image_width: int = 640
    image_height: int = 480

    # ── Lane detection — BEV perspective transform ──────────────
    bev_src_points: List[List[int]] = field(default_factory=lambda: [
        [160, 240], [480, 240], [640, 480], [0, 480],
    ])
    bev_dst_points: List[List[int]] = field(default_factory=lambda: [
        [100, 0], [540, 0], [540, 480], [100, 480],
    ])
    bev_width: int = 640
    bev_height: int = 480

    # ── Lane detection — thresholds ─────────────────────────────
    white_threshold: int = 160
    sliding_window_count: int = 12
    sliding_window_margin: int = 100
    sliding_window_min_pix: int = 30
    lane_history_frames: int = 5

    # ── Control — PID (not used in VROOM port, kept for compat) ──
    steering_kp: float = 1.0
    steering_ki: float = 0.0
    steering_kd: float = 0.0
    max_steering: float = 0.5   # rad, Ackermann limit
    steering_alpha: float = 1.0   # not used

    # ── Control — speed (VROOM: speed = 0.5) ────────────────────
    cruise_speed: float = 0.5
    slow_speed: float = 0.3
    stop_speed: float = 0.0
    highway_speed: float = 0.6
    intersection_speed: float = 0.5
    max_accel: float = 2.0
    max_decel: float = 4.0
    control_rate_hz: float = 30.0

    # ── FSM ─────────────────────────────────────────────────────
    stop_hold_sec: float = 3.0
    intersection_stop_sec: float = 1.5
    intersection_turn_sec: float = 2.5
    intersection_turn_steer: float = -0.8
    debounce_frames: int = 3
    frame_timeout_sec: float = 2.0

    # ── FSM — proximity thresholds ──────────────────────────────
    stop_soft_threshold: float = 0.40
    stop_hard_threshold: float = 0.55
    crosswalk_threshold: float = 0.45
    roundabout_threshold: float = 0.40
    parking_threshold: float = 0.45
    single_lane_offset_ratio: float = 0.20

    # ── Control — heading feedforward (not used in VROOM port) ──
    heading_gain: float = 0.0

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

    # ── Intersection navigation ─────────────────────────────────
    intersection_detect_distance_px: float = 60.0
    intersection_yaw_tolerance: float = 15.0

    # ── Visualisation ───────────────────────────────────────────
    publish_visualisation: bool = True

    # ── Missing lanes fallback ──────────────────────────────────
    missing_lane_timeout_sec: float = 2.0
