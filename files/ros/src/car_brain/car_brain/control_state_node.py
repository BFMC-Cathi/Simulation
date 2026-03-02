#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control & State Node — ROS 2
==============================
Houses the **Finite State Machine** and the **PID controller**.
This node does **no** image processing; it only subscribes to the
lightweight topics published by ``perception_node`` and outputs
velocity commands.

Subscriptions
-------------
  /perception/objects     (std_msgs/String — JSON)
  /perception/lane_state  (std_msgs/String — JSON)

Publications
------------
  /automobile/command     (geometry_msgs/Twist)

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from car_brain.config import DrivingConfig
from car_brain.fsm_logic import DrivingFSM, ObjectDetection, FSMOutput
from car_brain.control import PIDController


class ControlStateNode(Node):
    """
    ROS 2 node: perception topics → FSM → PID → /cmd_vel.

    The node runs at a fixed ``control_rate_hz`` timer and never blocks
    on any heavy computation — perception is done in a separate node.
    """

    def __init__(self) -> None:
        super().__init__("control_state_node")

        # ── Configuration ───────────────────────────────────────
        self._cfg = self._declare_and_load_params()

        # ── FSM ─────────────────────────────────────────────────
        self._fsm = DrivingFSM(self._cfg, self.get_logger())

        # ── PID for steering (CTE → angular_z) ─────────────────
        self._steer_pid = PIDController(
            kp=self._cfg.steering_kp,
            ki=self._cfg.steering_ki,
            kd=self._cfg.steering_kd,
            output_min=-self._cfg.max_steering,
            output_max=self._cfg.max_steering,
        )

        # ── PID for heading correction (heading_error → angular_z) ──
        self._heading_pid = PIDController(
            kp=self._cfg.steering_kp * 2.0,  # gentle heading correction
            ki=0.0,
            kd=self._cfg.steering_kd * 0.5,
            output_min=-self._cfg.max_steering * 0.3,
            output_max=self._cfg.max_steering * 0.3,
        )

        # ── Speed ramp state ───────────────────────────────────
        self._current_speed: float = 0.0
        self._filtered_steer: float = 0.0
        self._last_time: float = time.monotonic()

        # ── Latest perception data ─────────────────────────────
        self._latest_objects: Optional[dict] = None
        self._latest_lane: Optional[dict] = None
        self._objects_stamp: float = 0.0
        self._lane_stamp: float = 0.0

        # ── Subscribers ─────────────────────────────────────────
        self._obj_sub = self.create_subscription(
            String,
            self._cfg.objects_topic,
            self._on_objects,
            10,
        )
        self._lane_sub = self.create_subscription(
            String,
            self._cfg.lane_state_topic,
            self._on_lane,
            10,
        )

        # ── Publisher ───────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, self._cfg.cmd_topic, 10)

        # ── Control loop timer ──────────────────────────────────
        period = 1.0 / self._cfg.control_rate_hz
        self._timer = self.create_timer(period, self._control_tick)

        self.get_logger().info(
            f"control_state_node started  |  "
            f"PID Kp={self._cfg.steering_kp} Ki={self._cfg.steering_ki} "
            f"Kd={self._cfg.steering_kd}  |  "
            f"cruise={self._cfg.cruise_speed} m/s  |  "
            f"rate={self._cfg.control_rate_hz} Hz"
        )

    # ================================================================
    #  PARAMETER DECLARATION
    # ================================================================

    def _declare_and_load_params(self) -> DrivingConfig:
        cfg = DrivingConfig()

        def _p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        cfg.cmd_topic = _p("cmd_topic", cfg.cmd_topic)
        cfg.objects_topic = _p("objects_topic", cfg.objects_topic)
        cfg.lane_state_topic = _p("lane_state_topic", cfg.lane_state_topic)
        cfg.cruise_speed = _p("cruise_speed", cfg.cruise_speed)
        cfg.slow_speed = _p("slow_speed", cfg.slow_speed)
        cfg.highway_speed = _p("highway_speed", cfg.highway_speed)
        cfg.steering_kp = _p("steering_kp", cfg.steering_kp)
        cfg.steering_ki = _p("steering_ki", cfg.steering_ki)
        cfg.steering_kd = _p("steering_kd", cfg.steering_kd)
        cfg.max_steering = _p("max_steering", cfg.max_steering)
        cfg.steering_alpha = _p("steering_alpha", cfg.steering_alpha)
        cfg.max_accel = _p("max_accel", cfg.max_accel)
        cfg.max_decel = _p("max_decel", cfg.max_decel)
        cfg.control_rate_hz = _p("control_rate_hz", cfg.control_rate_hz)
        cfg.stop_hold_sec = _p("stop_hold_sec", cfg.stop_hold_sec)
        cfg.debounce_frames = _p("debounce_frames", cfg.debounce_frames)
        cfg.frame_timeout_sec = _p("frame_timeout_sec", cfg.frame_timeout_sec)
        cfg.image_height = _p("image_height", cfg.image_height)

        return cfg

    # ================================================================
    #  SUBSCRIPTION CALLBACKS
    # ================================================================

    def _on_objects(self, msg: String) -> None:
        """Parse JSON detections from perception_node."""
        try:
            self._latest_objects = json.loads(msg.data)
            self._objects_stamp = time.monotonic()
            self._fsm.notify_frame()
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad JSON on objects topic: {e}")

    def _on_lane(self, msg: String) -> None:
        """Parse JSON lane state from perception_node."""
        try:
            self._latest_lane = json.loads(msg.data)
            self._lane_stamp = time.monotonic()
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad JSON on lane topic: {e}")

    # ================================================================
    #  MAIN CONTROL LOOP
    # ================================================================

    def _control_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_time
        if dt <= 0.0:
            dt = 1e-4
        self._last_time = now

        # ── Parse YOLO detections for the FSM ───────────────────
        detections = []
        tl_colour = "unknown"
        if self._latest_objects is not None:
            tl_colour = self._latest_objects.get(
                "traffic_light_colour", "unknown"
            )
            for d in self._latest_objects.get("detections", []):
                detections.append(ObjectDetection(
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    x1=d["x1"], y1=d["y1"],
                    x2=d["x2"], y2=d["y2"],
                ))

        # ── Run FSM ─────────────────────────────────────────────
        fsm_out: FSMOutput = self._fsm.update(
            detections, tl_colour, self._cfg.image_height
        )

        # ── Parse lane state for PID ────────────────────────────
        cte = 0.0
        heading_error = 0.0
        using_fallback = False
        if self._latest_lane is not None:
            cte = self._latest_lane.get("cte", 0.0)
            heading_error = self._latest_lane.get("heading_error", 0.0)
            using_fallback = self._latest_lane.get("using_fallback", False)

        # ── Safety clamp: CTE must be normalised to [-1, +1] ───
        #    Reject any stale / corrupt value that is out of range.
        if abs(cte) > 1.0:
            self.get_logger().warn(
                f"CTE out of range ({cte:.1f}), clamping to 0"
            )
            cte = 0.0
            using_fallback = True
        if abs(heading_error) > 0.5:     # ~28 degrees — larger is noise
            heading_error = max(-0.5, min(0.5, heading_error))

        # ── Speed ramp ──────────────────────────────────────────
        target_speed = fsm_out.target_speed
        speed_diff = target_speed - self._current_speed
        if speed_diff > 0:
            max_step = self._cfg.max_accel * dt
            self._current_speed += min(speed_diff, max_step)
        else:
            max_step = self._cfg.max_decel * dt
            self._current_speed += max(speed_diff, -max_step)
        self._current_speed = max(0.0, self._current_speed)

        # ── Stale lane data check ───────────────────────────────
        lane_age = now - self._lane_stamp
        if lane_age > 0.5:  # lane_state older than 500ms → stale
            cte = 0.0
            heading_error = 0.0
            using_fallback = True

        # ── PID steering ────────────────────────────────────────
        # Dead-zone: ignore very small errors (CTE now normalised -1..+1)
        if abs(cte) < 0.01:
            cte = 0.0
        if abs(heading_error) < 0.01:
            heading_error = 0.0

        # Negate CTE: positive CTE means car is left of center,
        # so we need negative angular_z (turn right in Gazebo).
        steer_cte = self._steer_pid.compute(-cte)
        steer_heading = self._heading_pid.compute(-heading_error)
        raw_steer = steer_cte + steer_heading

        # Clamp
        raw_steer = max(-self._cfg.max_steering,
                        min(self._cfg.max_steering, raw_steer))

        # EMA low-pass filter
        alpha = self._cfg.steering_alpha
        self._filtered_steer = (
            alpha * raw_steer + (1.0 - alpha) * self._filtered_steer
        )

        # If using fallback (missing lanes) → reduce steering slightly
        if using_fallback:
            self._filtered_steer *= 0.7

        # ── Publish Twist ───────────────────────────────────────
        twist = Twist()
        twist.linear.x = self._current_speed
        twist.angular.z = self._filtered_steer
        self._cmd_pub.publish(twist)

        # ── Periodic logging ────────────────────────────────────
        if int(now * self._cfg.control_rate_hz) % 50 == 0:
            self.get_logger().info(
                f"[{fsm_out.state.name}] "
                f"v={self._current_speed:.2f}/{target_speed:.2f}  "
                f"steer={self._filtered_steer:+.3f}  "
                f"CTE={cte:+.1f}  head={heading_error:+.3f}  "
                f"tl={tl_colour}"
            )

    # ================================================================
    #  LIFECYCLE
    # ================================================================

    def shutdown(self) -> None:
        self.get_logger().info(
            "control_state_node shutting down — sending zero velocity"
        )
        try:
            twist = Twist()
            self._cmd_pub.publish(twist)
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlStateNode()
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
