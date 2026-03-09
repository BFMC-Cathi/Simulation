#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control & State Node — Team Cathı / BFMC
==========================================
ROS 2 node that:
  1. Subscribes to ``/perception/objects``, ``/perception/lane_state``,
     ``/automobile/IMU``, and ``/automobile/odometry``
  2. Feeds detections and lane state into the ``DrivingFSM``
  3. Translates the FSM's ``StateOutput`` into PID-controlled
     ``geometry_msgs/Twist`` messages on ``/automobile/command``

This is the "brain" node — it makes all driving decisions.

Fixes applied (v2)
------------------
  • **MultiThreadedExecutor** with separate callback groups so that
    heavy perception subscribers never starve the 30 Hz control timer.
  • **Auto-start**: after ``INIT_TIMEOUT_SEC`` seconds the FSM is
    forced into LANE_FOLLOWING even if no lane data has arrived,
    so the car always starts driving.
  • **Heartbeat log** every control cycle for easy debugging.
  • **Clean shutdown** handling (KeyboardInterrupt + ExternalShutdownException).

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from car_brain.config import DrivingConfig
from car_brain.control import VehicleController, TwistCommand
from car_brain.fsm_logic import DrivingFSM, State, StateOutput
from car_brain.track_graph import TrackGraph

# How long to wait in INIT before forcing LANE_FOLLOWING
INIT_TIMEOUT_SEC = 3.0


def _quaternion_to_yaw(q) -> float:
    """Extract yaw (degrees) from a quaternion (geometry_msgs/Quaternion)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad)


class ControlStateNode(Node):
    """FSM + PID controller — subscribes to perception, publishes Twist."""

    def __init__(self) -> None:
        super().__init__("control_state_node")

        # ── Callback groups ─────────────────────────────────────
        # Perception callbacks (may be slow — JSON parse, etc.)
        self._perception_cb_group = MutuallyExclusiveCallbackGroup()
        # Sensor callbacks (lightweight)
        self._sensor_cb_group = ReentrantCallbackGroup()
        # Control timer — must never be blocked by perception
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ── Build config ────────────────────────────────────────
        self._cfg = self._declare_and_load_config()

        # ── Track graph ─────────────────────────────────────────
        tg: Optional[TrackGraph] = None
        graph_path = self._cfg.track_graph_path
        if os.path.isfile(graph_path):
            try:
                tg = TrackGraph(graph_path)
                self.get_logger().info(
                    f"TrackGraph loaded: {graph_path} "
                    f"({tg.G.number_of_nodes()} nodes, "
                    f"{tg.G.number_of_edges()} edges)"
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to load TrackGraph: {exc}")
        else:
            self.get_logger().warn(
                f"Track graph not found: {graph_path} — "
                "intersection navigation will use timed fallback"
            )

        # ── FSM + controller ────────────────────────────────────
        self._fsm = DrivingFSM(self._cfg, tg, logger=self.get_logger())
        self._ctrl = VehicleController(self._cfg)

        # ── Cached perception data ──────────────────────────────
        self._last_detections: List[Dict] = []
        self._last_lane: Dict = {}
        self._last_traffic_light: str = "unknown"
        self._last_perception_time: float = 0.0

        # ── Pose from odometry ──────────────────────────────────
        self._pos_x: float = 0.0
        self._pos_y: float = 0.0
        self._yaw_deg: float = 0.0
        self._odom_recv_count: int = 0
        self._last_odom_time: float = 0.0  # monotonic time of last odom msg

        # ── IMU yaw (backup) ────────────────────────────────────
        self._imu_yaw_deg: float = 0.0

        # ── Stale perception safeguard ──────────────────────────
        # If perception is stale for longer than this, force stop.
        self._STALE_STOP_SEC: float = 1.0

        # ── Node creation timestamp (for INIT timeout) ──────────
        self._node_start_time: float = time.monotonic()
        self._init_forced: bool = False

        # ── Subscribers ─────────────────────────────────────────
        self._obj_sub = self.create_subscription(
            String, self._cfg.objects_topic, self._objects_cb, 10,
            callback_group=self._perception_cb_group,
        )
        self._lane_sub = self.create_subscription(
            String, self._cfg.lane_state_topic, self._lane_cb, 10,
            callback_group=self._perception_cb_group,
        )
        self._odom_sub = self.create_subscription(
            Odometry, self._cfg.odom_topic, self._odom_cb, 10,
            callback_group=self._sensor_cb_group,
        )
        self._imu_sub = self.create_subscription(
            Imu, self._cfg.imu_topic, self._imu_cb, 10,
            callback_group=self._sensor_cb_group,
        )

        # ── Publisher ───────────────────────────────────────────
        self._cmd_pub = self.create_publisher(
            Twist, self._cfg.cmd_topic, 10
        )

        # ── Control timer (in its own callback group) ───────────
        period = 1.0 / self._cfg.control_rate_hz
        self._timer = self.create_timer(
            period, self._control_loop,
            callback_group=self._timer_cb_group,
        )

        self.get_logger().info(
            f"ControlStateNode started — rate: {self._cfg.control_rate_hz} Hz, "
            f"cmd topic: {self._cfg.cmd_topic}, "
            f"INIT timeout: {INIT_TIMEOUT_SEC}s, "
            f"executor: MultiThreadedExecutor"
        )

    # ================================================================
    #  ROS parameter helpers
    # ================================================================

    def _declare_and_load_config(self) -> DrivingConfig:
        cfg = DrivingConfig()

        def _p(name: str, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # Track graph
        cfg.track_graph_path = _p("track_graph_path", cfg.track_graph_path)
        cfg.nav_source_node = _p("nav_source_node", cfg.nav_source_node)
        cfg.nav_target_node = _p("nav_target_node", cfg.nav_target_node)

        # Speed
        cfg.cruise_speed = _p("cruise_speed", cfg.cruise_speed)
        cfg.slow_speed = _p("slow_speed", cfg.slow_speed)
        cfg.highway_speed = _p("highway_speed", cfg.highway_speed)
        cfg.intersection_speed = _p("intersection_speed", cfg.intersection_speed)

        # PID
        cfg.steering_kp = _p("steering_kp", cfg.steering_kp)
        cfg.steering_ki = _p("steering_ki", cfg.steering_ki)
        cfg.steering_kd = _p("steering_kd", cfg.steering_kd)
        cfg.max_steering = _p("max_steering", cfg.max_steering)
        cfg.steering_alpha = _p("steering_alpha", cfg.steering_alpha)

        # Ramping
        cfg.max_accel = _p("max_accel", cfg.max_accel)
        cfg.max_decel = _p("max_decel", cfg.max_decel)
        cfg.control_rate_hz = _p("control_rate_hz", cfg.control_rate_hz)

        # FSM
        cfg.stop_hold_sec = _p("stop_hold_sec", cfg.stop_hold_sec)
        cfg.intersection_stop_sec = _p("intersection_stop_sec", cfg.intersection_stop_sec)
        cfg.intersection_turn_sec = _p("intersection_turn_sec", cfg.intersection_turn_sec)
        cfg.intersection_turn_steer = _p("intersection_turn_steer", cfg.intersection_turn_steer)
        cfg.debounce_frames = _p("debounce_frames", cfg.debounce_frames)
        cfg.frame_timeout_sec = _p("frame_timeout_sec", cfg.frame_timeout_sec)
        cfg.image_height = _p("image_height", cfg.image_height)
        cfg.image_width = _p("image_width", cfg.image_width)

        # Thresholds
        cfg.stop_soft_threshold = _p("stop_soft_threshold", cfg.stop_soft_threshold)
        cfg.stop_hard_threshold = _p("stop_hard_threshold", cfg.stop_hard_threshold)
        cfg.crosswalk_threshold = _p("crosswalk_threshold", cfg.crosswalk_threshold)
        cfg.roundabout_threshold = _p("roundabout_threshold", cfg.roundabout_threshold)
        cfg.parking_threshold = _p("parking_threshold", cfg.parking_threshold)

        return cfg

    # ================================================================
    #  Subscriber callbacks
    # ================================================================

    def _objects_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._last_detections = data.get("detections", [])
            self._last_traffic_light = data.get("traffic_light", "unknown")
            self._last_perception_time = time.monotonic()
        except (json.JSONDecodeError, TypeError):
            pass

    def _lane_cb(self, msg: String) -> None:
        try:
            self._last_lane = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            pass

    def _odom_cb(self, msg: Odometry) -> None:
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._yaw_deg = _quaternion_to_yaw(msg.pose.pose.orientation)
        self._last_odom_time = time.monotonic()
        self._odom_recv_count += 1
        if self._odom_recv_count == 1:
            self.get_logger().info(
                f"✓ First odom received on '{self._cfg.odom_topic}': "
                f"pos=({self._pos_x:.3f}, {self._pos_y:.3f}) "
                f"yaw={self._yaw_deg:.1f}°"
            )

    def _imu_cb(self, msg: Imu) -> None:
        self._imu_yaw_deg = _quaternion_to_yaw(msg.orientation)

    # ================================================================
    #  Main control loop (timer callback)
    # ================================================================

    def _control_loop(self) -> None:
        now = time.monotonic()

        # ── Auto-start: force out of INIT after timeout ─────────
        if (
            not self._init_forced
            and self._fsm.state == State.INIT
            and (now - self._node_start_time) >= INIT_TIMEOUT_SEC
        ):
            self._init_forced = True
            self.get_logger().warn(
                f"INIT timeout ({INIT_TIMEOUT_SEC}s) — forcing LANE_FOLLOWING"
            )
            # Directly poke the FSM state (acceptable for boot-strap)
            self._fsm._state = State.LANE_FOLLOWING
            self._fsm._state_enter_time = now

        # ── Check perception freshness ──────────────────────────
        perception_stale = (
            now - self._last_perception_time > self._cfg.frame_timeout_sec
            if self._last_perception_time > 0 else True
        )
        perception_stale_sec = (
            now - self._last_perception_time
            if self._last_perception_time > 0 else now - self._node_start_time
        )

        # ── Check odometry freshness ────────────────────────────
        odom_alive = self._odom_recv_count > 0
        odom_age = (
            now - self._last_odom_time
            if self._last_odom_time > 0 else now - self._node_start_time
        )
        if not odom_alive and (now - self._node_start_time) > 5.0:
            self.get_logger().error(
                f"⚠ ODOM DEAD — 0 messages received on "
                f"'{self._cfg.odom_topic}' after "
                f"{now - self._node_start_time:.0f}s! "
                f"Check ros_gz_bridge topic name.",
                throttle_duration_sec=3.0,
            )
        elif odom_alive and odom_age > 1.0:
            self.get_logger().warn(
                f"Odom stale ({odom_age:.1f}s since last msg, "
                f"{self._odom_recv_count} total)",
                throttle_duration_sec=2.0,
            )

        # Use odometry yaw; fall back to IMU if odom not available
        yaw = self._yaw_deg if self._yaw_deg != 0.0 else self._imu_yaw_deg

        # ── Update FSM pose ─────────────────────────────────────
        self._fsm.update_pose(self._pos_x, self._pos_y, yaw)

        # ── Tick the FSM ────────────────────────────────────────
        if perception_stale:
            # No fresh perception — pass empty but let FSM keep running
            detections: List[Dict] = []
            lane: Dict = self._last_lane  # reuse last lane (may be stale but better than {})
            tl = "unknown"
        else:
            detections = self._last_detections
            lane = self._last_lane
            tl = self._last_traffic_light

        output: StateOutput = self._fsm.tick(detections, lane, tl)

        # ── Convert FSM output to Twist ─────────────────────────
        twist = Twist()

        # ── Stale perception safeguard ──────────────────────────
        # If perception is stale beyond threshold, force speed to 0
        # to prevent blind driving off the track.
        force_stop = (
            perception_stale
            and perception_stale_sec > self._STALE_STOP_SEC
        )

        if force_stop:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif output.use_lane_keeping:
            # Use PID lane-centring + heading feedforward
            cte = lane.get("cte", 0.0)
            heading_err = lane.get("heading_error", 0.0)
            cmd: TwistCommand = self._ctrl.compute(
                target_speed=output.target_speed,
                lateral_error=cte,
                heading_error=heading_err,
            )
            twist.linear.x = cmd.linear_x
            twist.angular.z = cmd.angular_z
        else:
            # Direct speed/steer from FSM
            twist.linear.x = output.target_speed
            twist.angular.z = (
                output.steering_override
                if output.steering_override is not None
                else 0.0
            )

        self._cmd_pub.publish(twist)

        # ── Heartbeat log (with PID debug + route info) ─────────
        cte_val = lane.get("cte", 0.0) if lane else 0.0
        gx, gy = self._fsm.dbg_graph_xy
        d_next = self._fsm.dbg_dist_next_node
        next_id = self._fsm.dbg_next_node_id
        turn_dir = self._fsm.dbg_turn_dir
        path_idx = self._fsm._path_idx
        path_len = len(self._fsm._full_path)

        route_info = (
            f"Node:{next_id}[{path_idx}/{path_len}] "
            f"d={d_next:.2f}m"
        )
        if turn_dir:
            route_info += f" T:{turn_dir.upper()}"

        # VROOM steering internals (from VehicleController debug attrs)
        pid_info = (
            f"VROOM[ang={self._ctrl.dbg_raw_cte:+.1f}deg "
            f"mapped={self._ctrl.dbg_pid_out:+.1f}deg "
            f"rad={self._ctrl.dbg_filtered_steer:+.3f}]"
        )

        odom_check = "\u2713" if odom_alive else "\u2717"
        odom_diag = f"Odom:{self._odom_recv_count}{odom_check}"
        stop_diag = " FORCE_STOP" if force_stop else ""

        self.get_logger().info(
            f"[CTRL] {output.state_name:20s} | "
            f"Spd:{twist.linear.x:+.3f} Str:{twist.angular.z:+.3f} | "
            f"{pid_info} | "
            f"Pos:({gx:.2f},{gy:.2f}) | "
            f"{route_info} | "
            f"{odom_diag} | "
            f"{'OK' if not perception_stale else 'STALE'}"
            f"{stop_diag}",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlStateNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().fatal(f"Unhandled exception: {exc}")
    finally:
        node.get_logger().info("Shutting down ControlStateNode …")
        # Send a zero-velocity stop command before exiting
        stop_twist = Twist()
        node._cmd_pub.publish(stop_twist)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
