#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception Node — Team Cathı / BFMC
=====================================
ROS 2 node that subscribes to the camera image, runs YOLOv8 object
detection (threaded) and classical lane detection, then publishes
structured JSON results for the control/state node to consume.

Published topics
----------------
  /perception/objects      (std_msgs/String)   — JSON list of detections
  /perception/lane_state   (std_msgs/String)   — JSON lane-keeping state
  /perception/debug_image  (sensor_msgs/Image)  — annotated debug image

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String

from car_brain.config import DrivingConfig
from car_brain.perception import (
    PerceptionEngine,
    PerceptionResult,
    ros_image_to_cv2,
    cv2_to_ros_image,
)
from car_brain.lane_detection import LaneDetector


class PerceptionNode(Node):
    """Camera subscriber ➜ YOLO + lane detection ➜ JSON publishers."""

    def __init__(self) -> None:
        super().__init__("perception_node")

        # ── Build config from ROS parameters ────────────────────
        self._cfg = self._declare_and_load_config()

        # ── Subsystems ──────────────────────────────────────────
        self._engine = PerceptionEngine(self._cfg, logger=self.get_logger())
        self._lane_det = LaneDetector(self._cfg)

        # ── Subscribers ─────────────────────────────────────────
        self._img_sub = self.create_subscription(
            Image,
            self._cfg.camera_topic,
            self._image_cb,
            10,
        )

        # ── Publishers ──────────────────────────────────────────
        self._obj_pub = self.create_publisher(
            String, self._cfg.objects_topic, 10
        )
        self._lane_pub = self.create_publisher(
            String, self._cfg.lane_state_topic, 10
        )
        self._debug_pub = self.create_publisher(
            Image, self._cfg.yolo_vis_topic, 10
        )

        self._frame_count: int = 0
        self.get_logger().info(
            f"PerceptionNode started — camera: {self._cfg.camera_topic}"
        )

    # ================================================================
    #  ROS parameter helpers
    # ================================================================

    def _declare_and_load_config(self) -> DrivingConfig:
        """Declare every tunable field as a ROS parameter and build
        a DrivingConfig with the (possibly overridden) values."""
        cfg = DrivingConfig()

        def _p(name: str, default):  # noqa: ANN001
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        cfg.confidence_threshold = _p("confidence_threshold", cfg.confidence_threshold)
        cfg.publish_visualisation = _p("publish_visualisation", cfg.publish_visualisation)
        cfg.control_rate_hz = _p("control_rate_hz", cfg.control_rate_hz)
        cfg.white_threshold = _p("white_threshold", cfg.white_threshold)
        cfg.sliding_window_count = _p("sliding_window_count", cfg.sliding_window_count)
        cfg.sliding_window_margin = _p("sliding_window_margin", cfg.sliding_window_margin)
        cfg.sliding_window_min_pix = _p("sliding_window_min_pix", cfg.sliding_window_min_pix)
        cfg.lane_history_frames = _p("lane_history_frames", cfg.lane_history_frames)
        cfg.missing_lane_timeout_sec = _p("missing_lane_timeout_sec", cfg.missing_lane_timeout_sec)
        cfg.image_width = _p("image_width", cfg.image_width)
        cfg.image_height = _p("image_height", cfg.image_height)

        return cfg

    # ================================================================
    #  Camera callback
    # ================================================================

    def _image_cb(self, msg: Image) -> None:
        frame = ros_image_to_cv2(msg)
        if frame is None:
            return

        self._frame_count += 1

        # ── YOLO (push to background thread) ────────────────────
        self._engine.push_frame(frame)
        result: PerceptionResult | None = self._engine.get_result()

        # ── Lane detection (synchronous — lightweight) ──────────
        lane_state, lane_debug = self._lane_det.process(frame)

        # ── Publish detections ──────────────────────────────────
        if result is not None:
            det_list = []
            for d in result.detections:
                det_list.append({
                    "class": d.class_name,
                    "conf": round(d.confidence, 3),
                    "bbox": [round(d.x1, 1), round(d.y1, 1),
                             round(d.x2, 1), round(d.y2, 1)],
                    "area": round(d.area, 1),
                })
            obj_msg = String()
            obj_msg.data = json.dumps({
                "detections": det_list,
                "traffic_light": result.traffic_light_colour,
                "inference_ms": round(result.inference_ms, 1),
                "frame": self._frame_count,
            })
            self._obj_pub.publish(obj_msg)

        # ── Publish lane state ──────────────────────────────────
        lane_msg = String()
        lane_msg.data = json.dumps({
            "cte": round(lane_state.cte, 4),
            "heading_error": round(lane_state.heading_error, 4),
            "left_valid": lane_state.left_fit_valid,
            "right_valid": lane_state.right_fit_valid,
            "both_valid": lane_state.both_lanes_valid,
            "dashed": lane_state.dashed_line,
            "fallback": lane_state.using_fallback,
            "frame": self._frame_count,
        })
        self._lane_pub.publish(lane_msg)

        # ── Aggressive vision debug log ─────────────────────────
        self.get_logger().info(
            f"[VISION] Mask Pixels: {self._lane_det.dbg_white_px} | "
            f"Center Error: {self._lane_det.dbg_center_error:+.1f} | "
            f"CTE: {lane_state.cte:+.2f} | "
            f"L: {lane_state.left_fit_valid} R: {lane_state.right_fit_valid} | "
            f"Fallback: {lane_state.using_fallback}",
            throttle_duration_sec=0.5,
        )

        # ── ALWAYS publish debug image (critical for tuning) ────
        # Prefer the lane-detection 4-panel debug composite;
        # fall back to YOLO annotated frame.
        debug_frame = lane_debug  # always generated now
        if debug_frame is None and result is not None:
            debug_frame = result.annotated_frame
        if debug_frame is not None:
            self._debug_pub.publish(
                cv2_to_ros_image(debug_frame, stamp=msg.header.stamp)
            )

    # ================================================================
    #  Shutdown
    # ================================================================

    def destroy_node(self) -> None:
        self._engine.shutdown()
        super().destroy_node()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().fatal(f"Unhandled exception: {exc}")
    finally:
        node.get_logger().info("Shutting down PerceptionNode …")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
