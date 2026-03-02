#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception Node — ROS 2
========================
Handles **all** image processing in a single node so that no heavy CV
work runs inside ``control_state_node``'s spin loop.

Subscriptions
-------------
  /automobile/camera/image_raw  (sensor_msgs/Image)

Publications
------------
  /perception/objects     (std_msgs/String — JSON array of detections)
  /perception/lane_state  (std_msgs/String — JSON with CTE + heading)
  /perception/debug_image (sensor_msgs/Image — annotated composite)

Internals
---------
  • **YOLOv8** runs in a background thread (``PerceptionEngine``)
    so the ROS spin loop is never blocked by inference.
  • **Lane detection** (BEV + sliding window) runs synchronously at
    the control-rate timer frequency on the latest frame.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import json
import time
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from car_brain.config import DrivingConfig
from car_brain.perception import (
    PerceptionEngine,
    PerceptionResult,
    ros_image_to_cv2,
    cv2_to_ros_image,
)
from car_brain.lane_detection import LaneDetector, LaneState


class PerceptionNode(Node):
    """
    ROS 2 node: camera image → YOLO detections + lane state.

    This node does **not** generate any vehicle commands.  It only
    publishes perception results that ``control_state_node`` consumes.
    """

    def __init__(self) -> None:
        super().__init__("perception_node")

        # ── Load configuration from ROS parameters ──────────────
        self._cfg = self._declare_and_load_params()

        # ── Vision engines ──────────────────────────────────────
        self._perception = PerceptionEngine(self._cfg, self.get_logger())
        self._lane_detector = LaneDetector(self._cfg)

        # ── Camera subscriber ───────────────────────────────────
        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._image_sub = self.create_subscription(
            Image,
            self._cfg.camera_topic,
            self._on_image,
            cam_qos,
        )

        # ── Publishers ──────────────────────────────────────────
        self._objects_pub = self.create_publisher(
            String, self._cfg.objects_topic, 10
        )
        self._lane_pub = self.create_publisher(
            String, self._cfg.lane_state_topic, 10
        )
        self._debug_pub = self.create_publisher(
            Image, self._cfg.yolo_vis_topic, 5
        )

        # ── Timer for processing at control rate ────────────────
        period = 1.0 / self._cfg.control_rate_hz
        self._process_timer = self.create_timer(period, self._process_tick)

        # ── State ───────────────────────────────────────────────
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._last_frame_time: float = time.monotonic()

        self.get_logger().info(
            f"perception_node started  |  camera={self._cfg.camera_topic}  "
            f"|  YOLO={'LOADED' if self._perception.model_loaded else 'NONE'}  "
            f"|  rate={self._cfg.control_rate_hz} Hz"
        )

    # ================================================================
    #  ROS PARAMETER DECLARATION
    # ================================================================

    def _declare_and_load_params(self) -> DrivingConfig:
        """Declare every tunable value as a ROS 2 parameter."""
        cfg = DrivingConfig()

        def _p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        cfg.model_path = _p("model_path", cfg.model_path)
        cfg.camera_topic = _p("camera_topic", cfg.camera_topic)
        cfg.objects_topic = _p("objects_topic", cfg.objects_topic)
        cfg.lane_state_topic = _p("lane_state_topic", cfg.lane_state_topic)
        cfg.yolo_vis_topic = _p("yolo_vis_topic", cfg.yolo_vis_topic)
        cfg.confidence_threshold = _p(
            "confidence_threshold", cfg.confidence_threshold
        )
        cfg.control_rate_hz = _p("control_rate_hz", cfg.control_rate_hz)
        cfg.publish_visualisation = _p(
            "publish_visualisation", cfg.publish_visualisation
        )
        cfg.white_threshold = _p("white_threshold", cfg.white_threshold)
        cfg.sliding_window_count = _p(
            "sliding_window_count", cfg.sliding_window_count
        )
        cfg.sliding_window_margin = _p(
            "sliding_window_margin", cfg.sliding_window_margin
        )
        cfg.sliding_window_min_pix = _p(
            "sliding_window_min_pix", cfg.sliding_window_min_pix
        )
        cfg.lane_history_frames = _p(
            "lane_history_frames", cfg.lane_history_frames
        )
        cfg.missing_lane_timeout_sec = _p(
            "missing_lane_timeout_sec", cfg.missing_lane_timeout_sec
        )

        return cfg

    # ================================================================
    #  CALLBACKS
    # ================================================================

    def _on_image(self, msg: Image) -> None:
        """Camera callback — convert and store the latest frame."""
        cv_frame = ros_image_to_cv2(msg)
        if cv_frame is None:
            return

        # Skip frames that are essentially blank (sim not rendered yet).
        # Only reject truly uniform frames (e.g. all grey=178 before render).
        gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
        gray_std = float(gray.std())
        if gray_std < 3.0:
            if self._frame_count % 50 == 0:
                self.get_logger().warn(
                    f"Skipping blank frame (std={gray_std:.1f})"
                )
            return

        self._latest_frame = cv_frame
        self._frame_count += 1
        self._last_frame_time = time.monotonic()

        # Push to the YOLO background thread (non-blocking)
        self._perception.push_frame(cv_frame)

    def _process_tick(self) -> None:
        """Timer callback — run lane detection + publish results."""
        now = time.monotonic()

        # ── Safety: no frame for too long ───────────────────────
        if self._latest_frame is None:
            # Warn periodically instead of silently returning
            if int(now) % 3 == 0:
                self.get_logger().warn(
                    "Waiting for first camera frame…"
                )
            return
        if now - self._last_frame_time > self._cfg.frame_timeout_sec:
            self.get_logger().warn("No camera frame received — stale data")
            return

        frame = self._latest_frame

        # ── 1. Lane detection (synchronous, fast) ───────────────
        lane_state, lane_debug = self._lane_detector.process(frame)
        self._publish_lane_state(lane_state)

        # ── 2. YOLO detections (from background thread) ─────────
        yolo_result = self._perception.get_result()
        self._publish_objects(yolo_result)

        # ── 3. Debug visualisation ──────────────────────────────
        if self._cfg.publish_visualisation:
            self._publish_debug(frame, lane_debug, yolo_result)

        # ── Periodic log ────────────────────────────────────────
        if self._frame_count > 0 and self._frame_count % 200 == 0:
            self.get_logger().info(
                f"[frame {self._frame_count}] "
                f"CTE={lane_state.cte:+.1f}  "
                f"heading={np.degrees(lane_state.heading_error):+.1f}°  "
                f"YOLO dets="
                f"{len(yolo_result.detections) if yolo_result else 0}"
            )

    # ================================================================
    #  PUBLISHERS
    # ================================================================

    def _publish_lane_state(self, state: LaneState) -> None:
        """Publish CTE + heading angle as JSON string."""
        msg = String()
        msg.data = json.dumps({
            "cte": round(state.cte, 2),
            "heading_error": round(state.heading_error, 5),
            "left_valid": state.left_fit_valid,
            "right_valid": state.right_fit_valid,
            "both_valid": state.both_lanes_valid,
            "using_fallback": state.using_fallback,
            "stamp": state.timestamp,
        })
        self._lane_pub.publish(msg)

    def _publish_objects(self, result: Optional[PerceptionResult]) -> None:
        """Publish YOLO detections as JSON string."""
        msg = String()
        if result is None or not result.detections:
            msg.data = json.dumps({
                "detections": [],
                "traffic_light_colour": "unknown",
            })
        else:
            dets = []
            for d in result.detections:
                dets.append({
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "x1": round(d.x1, 1),
                    "y1": round(d.y1, 1),
                    "x2": round(d.x2, 1),
                    "y2": round(d.y2, 1),
                })
            msg.data = json.dumps({
                "detections": dets,
                "traffic_light_colour": result.traffic_light_colour,
            })
        self._objects_pub.publish(msg)

    def _publish_debug(
        self,
        frame: np.ndarray,
        lane_debug: Optional[np.ndarray],
        yolo_result: Optional[PerceptionResult],
    ) -> None:
        """Publish a composite debug image with YOLO + lane overlay."""
        # Start with the lane debug image (or original)
        if lane_debug is not None:
            base = lane_debug
        else:
            base = frame.copy()

        # Overlay YOLO bounding boxes on the left half
        if yolo_result is not None and yolo_result.detections:
            # The lane debug is a side-by-side composite (2*w);
            # we draw YOLO boxes only on the left half (original perspective)
            h, total_w = base.shape[:2]
            half_w = total_w // 2 if lane_debug is not None else total_w

            for d in yolo_result.detections:
                x1, y1 = int(d.x1), int(d.y1)
                x2, y2 = int(d.x2), int(d.y2)
                colour = (0, 255, 0)
                if "stop" in d.class_name.lower():
                    colour = (0, 0, 255)
                elif "traffic" in d.class_name.lower():
                    colour = (255, 0, 255)
                elif "roundabout" in d.class_name.lower():
                    colour = (255, 255, 0)

                # Clamp to left half
                x1c = min(x1, half_w - 1)
                x2c = min(x2, half_w - 1)
                cv2.rectangle(base, (x1c, y1), (x2c, y2), colour, 2)

                label = f"{d.class_name} {d.confidence:.2f}"
                cv2.putText(
                    base, label, (x1c, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                )

            # Traffic light colour badge
            if yolo_result.traffic_light_colour != "unknown":
                tl_col = yolo_result.traffic_light_colour
                badge_colour = {
                    "red": (0, 0, 255),
                    "yellow": (0, 255, 255),
                    "green": (0, 255, 0),
                }.get(tl_col, (200, 200, 200))
                cv2.putText(
                    base, f"TL: {tl_col.upper()}",
                    (half_w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_colour, 2,
                )

        ros_img = cv2_to_ros_image(base, self.get_clock().now().to_msg())
        self._debug_pub.publish(ros_img)

    # ================================================================
    #  LIFECYCLE
    # ================================================================

    def shutdown(self) -> None:
        self.get_logger().info("perception_node shutting down…")
        self._perception.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt.")
    except Exception as exc:
        node.get_logger().fatal(f"Unhandled: {exc}")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
