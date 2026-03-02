#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Driving Node — Team Cathı / BFMC  (v2 — Refactored)
=============================================================
A thin ROS 2 orchestrator that wires together the modular pipeline:

    Camera → Perception (background thread)
           → State Machine (FSM)
           → Control (PID + ramping)
           → /automobile/command (Twist)

Architecture
------------
  • **Camera callback** — converts the ROS Image and pushes the frame
    to ``PerceptionEngine`` (non-blocking; latest-frame-only semantics).
  • **Control timer** (``control_rate_hz``) — ticks at a fixed rate
    *independent of the camera FPS*, fetches the latest perception
    result, updates the state machine, computes smooth commands, and
    publishes.
  • **Visualisation** — annotated images are published on a separate
    timer / opportunistically to avoid slowing the control loop.

Author : Team Cathı – Bosch Future Mobility Challenge
Node   : yolov8_driving_node
License: Apache-2.0

Usage
-----
  ros2 run car_brain yolov8_driving_node
  ros2 run car_brain yolov8_driving_node --ros-args \\
      -p model_path:=/path/to/best.pt \\
      -p cruise_speed:=0.25 \\
      -p confidence_threshold:=0.45
"""

from __future__ import annotations

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
from geometry_msgs.msg import Twist

# ── Project modules ─────────────────────────────────────────────────
from car_brain.config import DrivingConfig
from car_brain.perception import (
    PerceptionEngine,
    PerceptionResult,
    ros_image_to_cv2,
    cv2_to_ros_image,
)
from car_brain.state_machine import DrivingStateMachine, DrivingState
from car_brain.control import VehicleController


class YOLOv8DrivingNode(Node):
    """
    Lightweight ROS 2 node — all heavy logic lives in the submodules.
    """

    def __init__(self) -> None:
        super().__init__("yolov8_driving_node")

        # ── Build config from ROS params ────────────────────────
        self._cfg = self._load_config()

        # ── Subsystems ──────────────────────────────────────────
        self._perception = PerceptionEngine(self._cfg, self.get_logger())
        self._fsm = DrivingStateMachine(self._cfg, self.get_logger())
        self._controller = VehicleController(self._cfg)

        # ── QoS ─────────────────────────────────────────────────
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,  # only latest frame matters
        )

        # ── Subscriber ──────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            self._cfg.camera_topic,
            self._image_callback,
            camera_qos,
        )

        # ── Publishers ──────────────────────────────────────────
        self._cmd_pub = self.create_publisher(
            Twist, self._cfg.cmd_topic, 10,
        )
        self._vis_pub = self.create_publisher(
            Image, self._cfg.yolo_vis_topic, 5,
        )

        # ── Fixed-rate control timer ────────────────────────────
        period = 1.0 / self._cfg.control_rate_hz
        self._control_timer = self.create_timer(period, self._control_tick)

        # ── Internal bookkeeping ────────────────────────────────
        self._image_width: int = 640
        self._image_height: int = 480
        self._frame_count: int = 0
        self._last_result: Optional[PerceptionResult] = None

        self.get_logger().info(
            f"yolov8_driving_node v2 initialised — "
            f"control @ {self._cfg.control_rate_hz} Hz, "
            f"model={'LOADED' if self._perception.model_loaded else 'NONE'}"
        )

    # ════════════════════════════════════════════════════════════════
    #  ROS PARAMETER → CONFIG
    # ════════════════════════════════════════════════════════════════
    def _load_config(self) -> DrivingConfig:
        """Declare ROS params and build a ``DrivingConfig``."""
        cfg = DrivingConfig()

        self.declare_parameter("model_path", cfg.model_path)
        self.declare_parameter("camera_topic", cfg.camera_topic)
        self.declare_parameter("cmd_topic", cfg.cmd_topic)
        self.declare_parameter("yolo_vis_topic", cfg.yolo_vis_topic)
        self.declare_parameter("confidence_threshold", cfg.confidence_threshold)
        self.declare_parameter("cruise_speed", cfg.cruise_speed)
        self.declare_parameter("steering_kp", cfg.steering_kp)
        self.declare_parameter("steering_ki", cfg.steering_ki)
        self.declare_parameter("steering_kd", cfg.steering_kd)
        self.declare_parameter("max_steering", cfg.max_steering)
        self.declare_parameter("control_rate_hz", cfg.control_rate_hz)
        self.declare_parameter("publish_visualisation", cfg.publish_visualisation)
        self.declare_parameter("stop_hold_sec", cfg.stop_hold_sec)
        self.declare_parameter("debounce_frames", cfg.debounce_frames)

        cfg.model_path = self.get_parameter("model_path").value
        cfg.camera_topic = self.get_parameter("camera_topic").value
        cfg.cmd_topic = self.get_parameter("cmd_topic").value
        cfg.yolo_vis_topic = self.get_parameter("yolo_vis_topic").value
        cfg.confidence_threshold = self.get_parameter("confidence_threshold").value
        cfg.cruise_speed = self.get_parameter("cruise_speed").value
        cfg.steering_kp = self.get_parameter("steering_kp").value
        cfg.steering_ki = self.get_parameter("steering_ki").value
        cfg.steering_kd = self.get_parameter("steering_kd").value
        cfg.max_steering = self.get_parameter("max_steering").value
        cfg.control_rate_hz = self.get_parameter("control_rate_hz").value
        cfg.publish_visualisation = self.get_parameter("publish_visualisation").value
        cfg.stop_hold_sec = self.get_parameter("stop_hold_sec").value
        cfg.debounce_frames = self.get_parameter("debounce_frames").value

        return cfg

    # ════════════════════════════════════════════════════════════════
    #  CAMERA CALLBACK — lightweight, non-blocking
    # ════════════════════════════════════════════════════════════════
    def _image_callback(self, msg: Image) -> None:
        """Convert ROS Image → numpy and push to perception engine."""
        cv_frame = ros_image_to_cv2(msg)
        if cv_frame is None:
            self.get_logger().warn(
                "Image conversion failed", throttle_duration_sec=5.0,
            )
            return

        self._image_height, self._image_width = cv_frame.shape[:2]
        self._frame_count += 1
        self._fsm.notify_frame_received()
        self._perception.push_frame(cv_frame)

    # ════════════════════════════════════════════════════════════════
    #  CONTROL TICK — fixed-rate, decoupled from camera
    # ════════════════════════════════════════════════════════════════
    def _control_tick(self) -> None:
        """
        Runs at ``control_rate_hz`` regardless of camera FPS.

        1. Fetch latest perception result
        2. Update state machine → StateOutput
        3. Compute smooth controls → TwistCommand
        4. Publish Twist
        5. (Optional) publish annotated visualisation
        """
        # 1. Latest perception
        result = self._perception.get_result()

        # Track if we got a new result (for vis publishing)
        new_result = result is not self._last_result
        self._last_result = result

        # 2. State machine
        state_output = self._fsm.update(
            result, self._image_width, self._image_height,
        )

        # 3. Control
        cmd = self._controller.compute(
            target_speed=state_output.target_speed,
            lateral_error=state_output.lateral_error,
        )

        # 4. Publish Twist
        twist = Twist()
        twist.linear.x = cmd.linear_x
        twist.angular.z = cmd.angular_z
        self._cmd_pub.publish(twist)

        # 5. Visualisation (only when we have a genuinely new annotated frame)
        if (
            self._cfg.publish_visualisation
            and new_result
            and result is not None
            and result.annotated_frame is not None
        ):
            # Overlay state & control info
            annotated = result.annotated_frame.copy()
            info = (
                f"State={state_output.state.name}  "
                f"v={cmd.linear_x:+.2f}  "
                f"steer={cmd.angular_z:+.3f}  "
                f"TL={result.traffic_light_colour}  "
                f"det={len(result.detections)}  "
                f"inf={result.inference_ms:.0f}ms"
            )
            cv2.putText(
                annotated, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            ros_img = cv2_to_ros_image(
                annotated, self.get_clock().now().to_msg(),
            )
            self._vis_pub.publish(ros_img)

        # ── Periodic diagnostic log ─────────────────────────────
        if self._frame_count > 0 and self._frame_count % 100 == 0:
            det_count = len(result.detections) if result else 0
            self.get_logger().info(
                f"[frame {self._frame_count}] "
                f"state={state_output.state.name}  "
                f"det={det_count}  "
                f"v={cmd.linear_x:.2f}  "
                f"steer={cmd.angular_z:.3f}"
            )

    # ════════════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ════════════════════════════════════════════════════════════════
    def shutdown(self) -> None:
        """Graceful cleanup — stop car and perception thread."""
        self.get_logger().info("Shutting down — sending zero velocity …")
        twist = Twist()
        self._cmd_pub.publish(twist)
        self._perception.shutdown()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(args=None) -> None:
    rclpy.init(args=args)
    node = YOLOv8DrivingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt.")
    except Exception as exc:
        node.get_logger().fatal(f"Unhandled exception: {exc}")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
