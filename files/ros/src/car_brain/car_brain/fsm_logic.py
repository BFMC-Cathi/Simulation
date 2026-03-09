#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-State Machine — Team Cathı / BFMC  (v3 — Route-Aware)
===============================================================
Port of VROOM-BFMC-Simulator ``main_FSM.py`` adapted for:
  • ROS 2 (rclpy)
  • New ``trackgraph.graphml`` parsed dynamically via TrackGraph
  • Standard ``geometry_msgs/Twist`` on ``/automobile/command``
  • Classical lane detection + YOLO perception pipeline

**v3 changes — Route-Following at Intersections**
  • Continuous proximity-based waypoint tracking via odometry
  • Auto-calibrating odom→graph coordinate transform
  • Intersection approach triggered by distance to entry node
  • Heading-based turn execution (pure-pursuit style)
  • Debug logging: distance to next node, turn direction

States
------
  INIT                 → waiting for first camera frame
  LANE_FOLLOWING       → normal cruise with PID lane-centering
  INTERSECTION_APPROACH→ slowing down, approaching an intersection
  INTERSECTION_STOP    → brief stop before intersection (if stop sign/line)
  INTERSECTION_NAV     → heading-guided turn through intersection
  STOPPED              → temporary stop (pedestrian / red light)
  ROUNDABOUT_NAV       → roundabout traversal
  PARKING              → parallel-parking manoeuvre
  HIGHWAY              → high-speed lane following
  EMERGENCY_STOP       → obstacle / no-entry / roadblock

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from car_brain.config import (
    DrivingConfig,
    CLASS_STOP_SIGN,
    CLASS_STOP_LINE,
    CLASS_CROSSWALK,
    CLASS_PEDESTRIAN,
    CLASS_TRAFFIC_LIGHT,
    CLASS_ROUNDABOUT_SIGN,
    CLASS_PARKING_SIGN,
    CLASS_HIGHWAY_ENTRY,
    CLASS_HIGHWAY_EXIT,
    CLASS_PROHIBITED_SIGN,
    CLASS_ROADBLOCK,
)
from car_brain.track_graph import TrackGraph


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State enumeration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class State(Enum):
    INIT = auto()
    LANE_FOLLOWING = auto()
    INTERSECTION_APPROACH = auto()
    INTERSECTION_STOP = auto()
    INTERSECTION_NAV = auto()
    STOPPED = auto()
    ROUNDABOUT_NAV = auto()
    PARKING = auto()
    HIGHWAY = auto()
    EMERGENCY_STOP = auto()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FSM output (what the control node consumes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StateOutput:
    """Compact struct describing what the controller should do."""
    __slots__ = (
        "target_speed", "steering_override", "use_lane_keeping",
        "state_name",
    )

    def __init__(
        self,
        target_speed: float = 0.0,
        steering_override: Optional[float] = None,
        use_lane_keeping: bool = True,
        state_name: str = "INIT",
    ) -> None:
        self.target_speed = target_speed
        self.steering_override = steering_override
        self.use_lane_keeping = use_lane_keeping
        self.state_name = state_name


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Debounce helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _Debounce:
    """Require a condition to hold for N consecutive frames."""

    def __init__(self, required: int = 3) -> None:
        self._required = required
        self._count: int = 0

    def update(self, active: bool) -> bool:
        if active:
            self._count = min(self._count + 1, self._required + 1)
        else:
            self._count = 0
        return self._count >= self._required

    def reset(self) -> None:
        self._count = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper: normalise angle to [-180, +180]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _norm_angle(a: float) -> float:
    """Normalise angle to (-180, 180] degrees."""
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Driving FSM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingFSM:
    """
    Graph-aware finite-state machine for autonomous driving.

    Call ``tick(detections, lane_state, traffic_light)`` every control
    cycle.  It returns a ``StateOutput`` that the control node
    translates into a ``Twist`` message.

    **v3 key changes:**
      * Odom→graph auto-calibration on first odometry reading
      * Continuous distance monitoring to upcoming waypoints
      * Proximity triggers for INTERSECTION_APPROACH / INTERSECTION_NAV
      * Heading-based turn execution (not time-based)
    """

    # ── Tunable constants ───────────────────────────────────────
    INTERSECTION_APPROACH_DIST = 2.00  # metres — start slowing
    INTERSECTION_ENTER_DIST    = 1.50  # metres — start turning
    INTERSECTION_EXIT_DIST     = 0.60  # metres — close to exit ⇒ done
    NAV_TIMEOUT_SEC            = 8.0   # safety exit timeout
    NODE_PASSED_DIST           = 0.50  # node considered "passed"

    # ── Heading-proportional intersection steering ──────────────
    # Inspired by MACH-GO/BFMC-MachCore lane_follower_pid.cpp:
    #   steer = kp * heading_error, clamped to ±steer_max_rad,
    #   rate-limited to steer_rate_limit_rad_s for smoothness,
    #   speed reduced proportional to |steer|.
    #
    # Our plugin: positive angular.z = LEFT in ROS 2.
    # heading_error > 0 means target is to the LEFT → steer LEFT
    # → positive angular.z.  Signs match directly.
    INT_KP           = 0.020     # rad per degree of heading error
    INT_STEER_MAX    = 0.42      # rad — don't saturate the 0.5 limit
    INT_RATE_LIMIT   = 1.5       # rad/s — max steering change rate
    INT_SPEED_TURN   = 0.30      # m/s during active turning
    INT_SPEED_STRAIGHT = 0.45    # m/s for straight-through
    INT_MIN_TIME     = 0.3       # seconds before allowing exit
    INT_YAW_DONE_TOL = 20.0      # degrees — yaw within this of target = done
    DEG2RAD = math.pi / 180.0

    def __init__(
        self,
        cfg: DrivingConfig,
        track_graph: Optional[TrackGraph] = None,
        logger: Any = None,
    ) -> None:
        self._cfg = cfg
        self._tg = track_graph
        self._log = logger

        # ── Core state ──────────────────────────────────────────
        self._state = State.INIT
        self._state_enter_time: float = time.monotonic()
        self._prev_state = State.INIT

        # ── Graph navigation ────────────────────────────────────
        self._full_path: List[str] = []
        self._path_idx: int = 0

        # ── Intersection manoeuvre state ────────────────────────
        self._int_entry_node: Optional[str] = None
        self._int_central_node: Optional[str] = None
        self._int_exit_node: Optional[str] = None
        self._int_turn_dir: str = "straight"   # "left"/"right"/"straight"
        self._int_turn_type: int = 0             # VROOM: -1=left, 0=straight, 1=right
        self._int_target_heading: float = 0.0
        self._int_start_yaw: float = 0.0         # yaw when intersection was detected
        self._int_inter_node: Optional[str] = None  # entry node (2 before target in VROOM)
        self._int_enter_time: float = 0.0

        # ── Odom → graph coordinate transform ──────────────────
        self._odom_calibrated: bool = False
        self._odom_offset_x: float = 0.0
        self._odom_offset_y: float = 0.0

        # ── Pose tracking (graph coordinates) ───────────────────
        self._gx: float = 0.0
        self._gy: float = 0.0
        self._yaw: float = 0.0

        # ── Debounce counters ───────────────────────────────────
        req = cfg.debounce_frames
        self._deb_stop = _Debounce(req)
        self._deb_traffic = _Debounce(req)
        self._deb_crosswalk = _Debounce(req)
        self._deb_roundabout = _Debounce(req)
        self._deb_parking = _Debounce(req)
        self._deb_highway_entry = _Debounce(req)
        self._deb_highway_exit = _Debounce(req)
        self._deb_pedestrian = _Debounce(req)
        self._deb_no_entry = _Debounce(req)

        # ── Misc timers ─────────────────────────────────────────
        self._stop_until: float = 0.0
        self._last_frame_time: float = time.monotonic()

        # ── Debug stats (exposed to control_state_node) ─────────
        self.dbg_dist_next_node: float = -1.0
        self.dbg_next_node_id: str = ""
        self.dbg_turn_dir: str = ""
        self.dbg_graph_xy: Tuple[float, float] = (0.0, 0.0)

        # ── Initialise graph path ───────────────────────────────
        if self._tg is not None:
            self._full_path = self._tg.shortest_path(
                cfg.nav_source_node, cfg.nav_target_node
            )
            self._path_idx = 0
            self._info(
                f"Graph path: {len(self._full_path)} nodes  "
                f"({cfg.nav_source_node} → {cfg.nav_target_node})"
            )
            if self._full_path:
                first_few = [
                    f"{n}({self._tg.node_xy(n)[0]:.1f},{self._tg.node_xy(n)[1]:.1f})"
                    for n in self._full_path[:5]
                ]
                self._info(f"  First: {' → '.join(first_few)} …")

    # ================================================================
    #  PUBLIC API
    # ================================================================

    @property
    def state(self) -> State:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    # ────────────────────────────────────────────────────────────
    #  Pose update  (called every cycle from control_state_node)
    # ────────────────────────────────────────────────────────────

    def update_pose(self, odom_x: float, odom_y: float, yaw_deg: float) -> None:
        """
        Convert raw odometry into graph coordinates and advance the
        path pointer.  Auto-calibrates on the first call.
        """
        self._yaw = yaw_deg

        # ── Auto-calibrate on first reading ─────────────────────
        if not self._odom_calibrated and self._tg and self._full_path:
            src = self._full_path[0]
            gx_src, gy_src = self._tg.node_xy(src)
            self._odom_offset_x = gx_src - odom_x
            self._odom_offset_y = gy_src - odom_y
            self._odom_calibrated = True
            self._info(
                f"Odom→Graph calibrated: offset=({self._odom_offset_x:.3f}, "
                f"{self._odom_offset_y:.3f})  "
                f"odom=({odom_x:.3f},{odom_y:.3f}) → "
                f"graph=({gx_src:.3f},{gy_src:.3f})"
            )

        # ── Apply transform ─────────────────────────────────────
        self._gx = odom_x + self._odom_offset_x
        self._gy = odom_y + self._odom_offset_y
        self.dbg_graph_xy = (self._gx, self._gy)

        # ── Advance path pointer past visited nodes ─────────────
        self._advance_path_pointer()

    def _advance_path_pointer(self) -> None:
        """Skip nodes we have already passed.

        Two criteria (either triggers advance):
          1. We are within NODE_PASSED_DIST of the current node AND
             the next node is closer than (or nearly as far as) current.
          2. We are within 2× NODE_PASSED_DIST of the current node AND
             the next node is strictly closer.  The proximity guard
             prevents the pointer from racing ahead when all nodes
             are far away.
        """
        if not self._tg or not self._full_path:
            return

        while self._path_idx < len(self._full_path) - 1:
            nid = self._full_path[self._path_idx]
            nx_, ny = self._tg.node_xy(nid)
            d = math.hypot(nx_ - self._gx, ny - self._gy)

            nid2 = self._full_path[self._path_idx + 1]
            nx2, ny2 = self._tg.node_xy(nid2)
            d2 = math.hypot(nx2 - self._gx, ny2 - self._gy)

            # Criterion 1: close enough to current node, next is reachable
            close_enough = d < self.NODE_PASSED_DIST and d2 < d + 0.30
            # Criterion 2: next node is strictly closer AND we are
            #              reasonably near the current node (< 2× threshold)
            overshot = d < self.NODE_PASSED_DIST * 2.0 and d2 < d - 0.10

            if close_enough or overshot:
                self._path_idx += 1
            else:
                break

        # ── Update debug info ───────────────────────────────────
        if self._path_idx < len(self._full_path):
            nid = self._full_path[self._path_idx]
            nx_, ny = self._tg.node_xy(nid)
            self.dbg_dist_next_node = math.hypot(nx_ - self._gx, ny - self._gy)
            self.dbg_next_node_id = nid

            # ── Heading to target node ──────────────────────────
            dx = nx_ - self._gx
            dy = ny - self._gy
            deg_to_target = math.degrees(math.atan2(dy, dx))

            # ── High-frequency distance log (every tick) ────────
            # Shows distance to the next waypoint + upcoming intersection
            upcoming = self._find_upcoming_intersection()
            int_info = ""
            if upcoming is not None:
                _, entry_id, central_id, exit_id = upcoming
                d_entry = self._dist_to_node(entry_id)
                int_info = (
                    f" | INT: {entry_id}→{central_id}→{exit_id} "
                    f"d_entry={d_entry:.2f}m"
                )
            self._info(
                f"[NAV] Heading to Node {nid} | "
                f"Deg to Target: {deg_to_target:+.1f} | "
                f"Dist: {self.dbg_dist_next_node:.2f}m | "
                f"Car: ({self._gx:.2f}, {self._gy:.2f}) | "
                f"Yaw: {self._yaw:.1f}°"
                f"{int_info}"
            )
        else:
            self.dbg_dist_next_node = -1.0
            self.dbg_next_node_id = "DONE"

    # ────────────────────────────────────────────────────────────
    #  Main tick
    # ────────────────────────────────────────────────────────────

    def tick(
        self,
        detections: List[Dict],
        lane: Dict,
        traffic_light: str = "unknown",
    ) -> StateOutput:
        now = time.monotonic()
        self._last_frame_time = now

        flags = self._compute_detection_flags(detections, traffic_light)

        new_state = self._decide_transition(flags, lane, now)
        if new_state != self._state:
            self._prev_state = self._state
            self._info(f"FSM: {self._state.name} → {new_state.name}")
            self._state = new_state
            self._state_enter_time = now

        output = self._compute_output(flags, lane, now)
        return output

    # ================================================================
    #  Waypoint & intersection look-ahead
    # ================================================================

    def _find_upcoming_intersection(self) -> Optional[Tuple[int, str, str, str]]:
        """
        Scan ahead in ``_full_path`` from ``_path_idx`` for the next
        intersection entry → central → exit sequence.

        Detects:
          1. Classic intersections (entry node flagged by graph topology)
          2. T-junctions / merges (node before a central/fan-out node
             even if the entry node itself isn't in the entry set)

        Returns ``(entry_path_idx, entry_id, central_id, exit_id)``
        or ``None``.
        """
        if not self._tg or not self._full_path:
            return None

        limit = min(self._path_idx + 20, len(self._full_path))
        for i in range(self._path_idx, limit):
            nid = self._full_path[i]

            # ── Classic intersection entry node ─────────────────
            if self._tg.is_intersection_node(nid):
                if i + 2 < len(self._full_path):
                    central = self._full_path[i + 1]
                    exit_n = self._full_path[i + 2]
                    return (i, nid, central, exit_n)

            # ── T-junction / merge detection ────────────────────
            # If the NEXT node on the path is a central (fan-out)
            # node, treat the current node as the entry even if
            # it wasn't auto-detected as an intersection entry.
            # This catches merges where a side road feeds into a
            # central node with out-degree >= 2.
            if i + 2 < len(self._full_path):
                next_nid = self._full_path[i + 1]
                if self._tg.is_central_node(next_nid):
                    exit_n = self._full_path[i + 2]
                    return (i, nid, next_nid, exit_n)

        return None

    def _dist_to_node(self, node_id: str) -> float:
        if not self._tg:
            return 999.0
        nx_, ny = self._tg.node_xy(node_id)
        return math.hypot(nx_ - self._gx, ny - self._gy)

    def _heading_to_node(self, node_id: str) -> float:
        """Heading (deg) from current pos to node.  0°=+X, 90°=+Y."""
        if not self._tg:
            return 0.0
        nx_, ny = self._tg.node_xy(node_id)
        dx = nx_ - self._gx
        dy = ny - self._gy
        return math.degrees(math.atan2(dy, dx))

    # ================================================================
    #  Detection flag computation
    # ================================================================

    def _compute_detection_flags(
        self, detections: List[Dict], traffic_light: str
    ) -> Dict[str, Any]:
        seen: Dict[str, float] = {}
        for d in detections:
            cls = d.get("class", "")
            area = d.get("area", 0.0)
            if cls not in seen or area > seen[cls]:
                seen[cls] = area

        img_area = self._cfg.image_width * self._cfg.image_height

        def _ratio(cls: str) -> float:
            return seen.get(cls, 0.0) / max(img_area, 1)

        stop_ratio = max(_ratio(CLASS_STOP_SIGN), _ratio(CLASS_STOP_LINE))
        crosswalk_ratio = _ratio(CLASS_CROSSWALK)
        pedestrian_ratio = _ratio(CLASS_PEDESTRIAN)
        roundabout_ratio = _ratio(CLASS_ROUNDABOUT_SIGN)
        parking_ratio = _ratio(CLASS_PARKING_SIGN)
        no_entry_ratio = _ratio(CLASS_PROHIBITED_SIGN) + _ratio(CLASS_ROADBLOCK)

        # ── Graph proximity ─────────────────────────────────────
        intersection_nearby = False
        dist_to_intersection = 999.0
        upcoming = self._find_upcoming_intersection()
        if upcoming is not None:
            _, entry_id, _, _ = upcoming
            dist_to_intersection = self._dist_to_node(entry_id)
            intersection_nearby = (
                dist_to_intersection < self.INTERSECTION_APPROACH_DIST
            )

        return {
            "stop_close": self._deb_stop.update(
                stop_ratio > self._cfg.stop_hard_threshold
            ),
            "stop_approaching": stop_ratio > self._cfg.stop_soft_threshold * 0.5,
            "crosswalk_close": self._deb_crosswalk.update(
                crosswalk_ratio > self._cfg.crosswalk_threshold
            ),
            "pedestrian": self._deb_pedestrian.update(
                pedestrian_ratio > 0.02
            ),
            "roundabout": self._deb_roundabout.update(
                roundabout_ratio > self._cfg.roundabout_threshold
            ),
            "parking": self._deb_parking.update(
                parking_ratio > self._cfg.parking_threshold
            ),
            "highway_entry": self._deb_highway_entry.update(
                CLASS_HIGHWAY_ENTRY in seen
            ),
            "highway_exit": self._deb_highway_exit.update(
                CLASS_HIGHWAY_EXIT in seen
            ),
            "no_entry": self._deb_no_entry.update(
                no_entry_ratio > 0.02
            ),
            "traffic_light": traffic_light,
            "traffic_red": self._deb_traffic.update(
                traffic_light == "red"
            ),
            # Graph-based
            "intersection_nearby": intersection_nearby,
            "dist_to_intersection": dist_to_intersection,
            "upcoming_intersection": upcoming,
        }

    # ================================================================
    #  State transition logic
    # ================================================================

    def _decide_transition(
        self, flags: Dict[str, Any], lane: Dict, now: float
    ) -> State:
        elapsed = now - self._state_enter_time

        # ── INIT ─────────────────────────────────────────────────
        if self._state == State.INIT:
            if lane.get("left_valid") or lane.get("right_valid"):
                return State.LANE_FOLLOWING
            return State.INIT

        # ── Global emergency overrides ──────────────────────────
        if flags["no_entry"]:
            return State.EMERGENCY_STOP

        if flags["pedestrian"] and self._state not in (
            State.STOPPED, State.EMERGENCY_STOP
        ):
            self._stop_until = now + self._cfg.stop_hold_sec
            return State.STOPPED

        if flags["traffic_red"] and self._state not in (
            State.INTERSECTION_NAV, State.ROUNDABOUT_NAV,
            State.STOPPED, State.EMERGENCY_STOP,
        ):
            self._stop_until = now + self._cfg.stop_hold_sec
            return State.STOPPED

        # ── State-specific ──────────────────────────────────────

        if self._state == State.LANE_FOLLOWING:
            # 1) YOLO stop sign very close
            if flags["stop_close"]:
                return State.INTERSECTION_STOP

            # 2) Graph-based intersection approach (THE KEY LINK)
            if flags["intersection_nearby"]:
                upcoming = flags["upcoming_intersection"]
                if upcoming is not None:
                    self._prepare_intersection(upcoming)
                    d = flags["dist_to_intersection"]
                    if d < self.INTERSECTION_ENTER_DIST:
                        self._int_enter_time = now
                        self._int_start_yaw = self._yaw
                        self._info(
                            f"Direct → NAV (dist={d:.2f}m, "
                            f"turn={self._int_turn_dir})"
                        )
                        return State.INTERSECTION_NAV
                    self._info(
                        f"Intersection ahead: entry={self._int_entry_node} "
                        f"turn={self._int_turn_dir} dist={d:.2f}m"
                    )
                    return State.INTERSECTION_APPROACH

            # 3) YOLO-based approaching
            if flags["stop_approaching"]:
                return State.INTERSECTION_APPROACH

            if flags["crosswalk_close"]:
                self._stop_until = now + self._cfg.stop_hold_sec
                return State.STOPPED
            if flags["roundabout"]:
                return State.ROUNDABOUT_NAV
            if flags["parking"]:
                return State.PARKING
            if flags["highway_entry"]:
                return State.HIGHWAY
            return State.LANE_FOLLOWING

        elif self._state == State.INTERSECTION_APPROACH:
            # Close enough to enter?
            if self._int_entry_node is not None:
                d = self._dist_to_node(self._int_entry_node)
                if d < self.INTERSECTION_ENTER_DIST:
                    self._int_enter_time = now
                    self._int_start_yaw = self._yaw
                    self._info(
                        f"Entering intersection: turn={self._int_turn_dir} "
                        f"dist={d:.2f}m"
                    )
                    return State.INTERSECTION_NAV

            if flags["stop_close"]:
                return State.INTERSECTION_STOP

            # Timeout
            if elapsed > 4.0:
                self._int_enter_time = now
                self._int_start_yaw = self._yaw
                self._warn("APPROACH timeout — forcing INTERSECTION_NAV")
                return State.INTERSECTION_NAV

            # Passed the intersection without entering
            if not flags["intersection_nearby"] and elapsed > 1.5:
                self._clear_intersection()
                return State.LANE_FOLLOWING

            return State.INTERSECTION_APPROACH

        elif self._state == State.INTERSECTION_STOP:
            if elapsed >= self._cfg.intersection_stop_sec:
                upcoming = self._find_upcoming_intersection()
                if upcoming and self._int_entry_node is None:
                    self._prepare_intersection(upcoming)
                self._int_enter_time = now
                self._int_start_yaw = self._yaw
                return State.INTERSECTION_NAV
            return State.INTERSECTION_STOP

        elif self._state == State.INTERSECTION_NAV:
            nav_elapsed = now - self._int_enter_time

            # Safety timeout
            if nav_elapsed > self.NAV_TIMEOUT_SEC:
                self._warn(
                    f"NAV timeout ({self.NAV_TIMEOUT_SEC}s) — "
                    f"→ LANE_FOLLOWING"
                )
                self._clear_intersection()
                return State.LANE_FOLLOWING

            # ── VROOM exit check: has the car passed the next_node? ──
            # Source: PathPlanning.intersection_navigation() lines 180-194
            # "If closest_node_index >= next_node_index → reached_target"
            reached = self._vroom_check_reached_target()
            if reached:
                self._info(
                    f"{self._int_turn_dir.upper()} done "
                    f"(passed exit node, t={nav_elapsed:.1f}s)"
                )
                self._clear_intersection()
                return State.LANE_FOLLOWING

            return State.INTERSECTION_NAV

        elif self._state == State.STOPPED:
            if now >= self._stop_until:
                if not flags["pedestrian"] and not flags["traffic_red"]:
                    return State.LANE_FOLLOWING
            return State.STOPPED

        elif self._state == State.ROUNDABOUT_NAV:
            if elapsed > 4.0 and not flags["roundabout"]:
                return State.LANE_FOLLOWING
            return State.ROUNDABOUT_NAV

        elif self._state == State.PARKING:
            if elapsed > 8.0:
                return State.LANE_FOLLOWING
            return State.PARKING

        elif self._state == State.HIGHWAY:
            if flags["highway_exit"]:
                return State.LANE_FOLLOWING
            return State.HIGHWAY

        elif self._state == State.EMERGENCY_STOP:
            if elapsed > 5.0 and not flags["no_entry"]:
                return State.LANE_FOLLOWING
            return State.EMERGENCY_STOP

        return self._state

    # ================================================================
    #  Output computation per state
    # ================================================================

    def _compute_output(
        self, flags: Dict[str, Any], lane: Dict, now: float
    ) -> StateOutput:
        cfg = self._cfg

        if self._state == State.INIT:
            return StateOutput(target_speed=0.0, use_lane_keeping=False,
                               state_name="INIT")

        elif self._state == State.LANE_FOLLOWING:
            return StateOutput(target_speed=cfg.cruise_speed,
                               use_lane_keeping=True,
                               state_name="LANE_FOLLOWING")

        elif self._state == State.INTERSECTION_APPROACH:
            return StateOutput(target_speed=cfg.slow_speed,
                               use_lane_keeping=True,
                               state_name="INTERSECTION_APPROACH")

        elif self._state == State.INTERSECTION_STOP:
            return StateOutput(target_speed=0.0, steering_override=0.0,
                               use_lane_keeping=False,
                               state_name="INTERSECTION_STOP")

        elif self._state == State.INTERSECTION_NAV:
            steer = self._compute_intersection_steer(now)
            # MACH-GO pattern: reduce speed when steering is large
            # speed = base * (1 - scale * |steer|/max_steer)
            if self._int_turn_type == 0:  # straight
                speed = self.INT_SPEED_STRAIGHT
            else:
                turn_frac = abs(steer) / max(self.INT_STEER_MAX, 0.01)
                speed = self.INT_SPEED_TURN * (1.0 - 0.3 * turn_frac)
                speed = max(speed, 0.18)  # don't stall
            return StateOutput(target_speed=speed,
                               steering_override=steer,
                               use_lane_keeping=False,
                               state_name="INTERSECTION_NAV")

        elif self._state == State.STOPPED:
            return StateOutput(target_speed=0.0, steering_override=0.0,
                               use_lane_keeping=False,
                               state_name="STOPPED")

        elif self._state == State.ROUNDABOUT_NAV:
            return StateOutput(target_speed=cfg.slow_speed,
                               use_lane_keeping=True,
                               state_name="ROUNDABOUT_NAV")

        elif self._state == State.PARKING:
            return self._run_parking(now)

        elif self._state == State.HIGHWAY:
            return StateOutput(target_speed=cfg.highway_speed,
                               use_lane_keeping=True,
                               state_name="HIGHWAY")

        elif self._state == State.EMERGENCY_STOP:
            return StateOutput(target_speed=0.0, steering_override=0.0,
                               use_lane_keeping=False,
                               state_name="EMERGENCY_STOP")

        return StateOutput(target_speed=0.0, use_lane_keeping=False,
                           state_name=self._state.name)

    # ================================================================
    #  Intersection navigation — heading-based controller
    # ================================================================

    def _prepare_intersection(
        self, upcoming: Tuple[int, str, str, str]
    ) -> None:
        """
        Latch the intersection geometry for steering.

        VROOM mapping (PathPlanning.intersection_navigation lines 180-210):
          inter_node  = complete_path[target_index - 2]  → our entry_node
          target_node = complete_path[target_index - 1]  → our central_node
          next_node   = G[target_node][0]                → our exit_node
        """
        _, entry_id, central_id, exit_id = upcoming
        self._int_entry_node = entry_id
        self._int_central_node = central_id
        self._int_exit_node = exit_id
        self._int_inter_node = entry_id  # "inter_node" in VROOM = our entry
        self._int_start_yaw = self._yaw

        if self._tg:
            self._int_turn_dir = self._tg.classify_turn(
                entry_id, central_id, exit_id
            )
            # VROOM turn_type: -1 = left, 0 = straight, 1 = right
            if self._int_turn_dir == "left":
                self._int_turn_type = -1
            elif self._int_turn_dir == "right":
                self._int_turn_type = 1
            else:
                self._int_turn_type = 0

            # Target heading = heading from central → exit
            cx, cy = self._tg.node_xy(central_id)
            ex, ey = self._tg.node_xy(exit_id)
            self._int_target_heading = math.degrees(
                math.atan2(ey - cy, ex - cx)
            )
        else:
            self._int_turn_dir = "straight"
            self._int_turn_type = 0
            self._int_target_heading = self._yaw

        self.dbg_turn_dir = self._int_turn_dir

        self._info(
            f"Intersection: {entry_id}→{central_id}→{exit_id}  "
            f"Turn: {self._int_turn_dir.upper()} (type={self._int_turn_type})  "
            f"Target hdg: {self._int_target_heading:.1f}°"
        )

    def _compute_intersection_steer(self, now: float) -> float:
        """
        Compute steering during intersection navigation.

        Replaces VROOM's open-loop cardinal-direction time ramps with
        a closed-loop heading-error proportional controller.

        Inspired by MACH-GO/BFMC-MachCore lane_follower_pid.cpp:
          steer = Kp * heading_error
          rate-limited for smoothness
          clamped to ±INT_STEER_MAX

        The target heading is the bearing from the car's current
        position to the EXIT node of the intersection.
        heading_error > 0 → target is to the LEFT of the car
        → positive angular.z (ROS: turn LEFT).  Signs match directly.

        For STRAIGHT intersections, the target is the exit node.
        For LEFT/RIGHT turns, the target starts as the central node
        (to pull the car into the intersection) then switches to
        the exit node once the car is closer to it.
        """
        if self._int_exit_node is None or not self._tg:
            return 0.0

        elapsed = now - self._int_enter_time
        yaw = self._yaw  # current heading in degrees

        # ── Choose target node ──────────────────────────────────
        # For turns: aim at central node first to pull into the
        # intersection, then switch to exit node when closer.
        # For straight: always aim at exit node.
        d_central = self._dist_to_node(self._int_central_node) if self._int_central_node else 999.0
        d_exit = self._dist_to_node(self._int_exit_node)

        if self._int_turn_type == 0:  # STRAIGHT
            target_node = self._int_exit_node
        elif d_central < 0.35 or d_exit < d_central:
            # Close to or past central → aim at exit
            target_node = self._int_exit_node
        else:
            target_node = self._int_central_node

        # ── Compute heading error ───────────────────────────────
        # bearing_to_target: angle from car pos to target node
        target_bearing = self._heading_to_node(target_node)
        heading_error = _norm_angle(target_bearing - yaw)  # degrees

        # ── Proportional control (MACH-GO Kp approach) ──────────
        # steer_rad = Kp * heading_error_deg
        # Kp = 0.020 rad/deg → 20° error → 0.40 rad steering
        steer_raw = self.INT_KP * heading_error

        # ── Clamp to safe range ─────────────────────────────────
        steer_raw = max(-self.INT_STEER_MAX,
                        min(self.INT_STEER_MAX, steer_raw))

        # ── Rate limiting (MACH-GO: steer_rate_limit_rad_s) ─────
        # Prevents sudden jerks when switching from lane-keeping
        # to intersection steering.  dt ≈ 1/30 at 30 Hz.
        dt = 1.0 / max(self._cfg.control_rate_hz, 1.0)
        max_delta = self.INT_RATE_LIMIT * dt
        if not hasattr(self, '_prev_int_steer'):
            self._prev_int_steer = 0.0
        steer_rad = max(self._prev_int_steer - max_delta,
                        min(self._prev_int_steer + max_delta, steer_raw))
        self._prev_int_steer = steer_rad

        self._info(
            f"INT Steer: {self._int_turn_dir.upper()} | "
            f"target={target_node} bearing={target_bearing:+.1f}° | "
            f"yaw={yaw:+.1f}° h_err={heading_error:+.1f}° | "
            f"steer={steer_rad:+.3f}rad | "
            f"d_cen={d_central:.2f}m d_exit={d_exit:.2f}m | "
            f"t={elapsed:.1f}s"
        )

        return steer_rad

    def _vroom_check_reached_target(self) -> bool:
        """
        Exit condition for intersection navigation.

        Three independent checks (any one triggers exit):
          1. Path pointer advanced past exit node (odometry-based)
          2. Close to exit node AND heading within tolerance of
             target heading (heading-based — works for any approach)
          3. Straight-through: distance to exit node < threshold

        Inspired by MACH-GO's clean architecture — no hardcoded
        cardinal direction buckets.  Works for any yaw.
        """
        if not self._tg or self._int_exit_node is None:
            return False

        elapsed = time.monotonic() - self._int_enter_time

        # Don't exit too early — let the car actually start turning
        if elapsed < self.INT_MIN_TIME:
            return False

        # ── Check 1: Path-index based (odometry pointer past exit) ──
        try:
            exit_idx = self._full_path.index(self._int_exit_node)
            if self._path_idx >= exit_idx:
                return True
        except ValueError:
            pass

        d_exit = self._dist_to_node(self._int_exit_node)

        # ── Check 2: Heading aligned with target + close enough ─────
        # target_heading is entry→exit bearing, computed in _prepare.
        # If the car's yaw is within tolerance of this AND the car
        # is within 1.0m of the exit node, consider the turn done.
        yaw = self._yaw
        heading_err = abs(_norm_angle(self._int_target_heading - yaw))
        if heading_err < self.INT_YAW_DONE_TOL and d_exit < 1.0:
            return True

        # ── Check 3: Very close to exit node (any heading) ──────────
        if d_exit < self.INTERSECTION_EXIT_DIST:
            return True

        # ── Check 4: Straight-through timeout ───────────────────────
        if self._int_turn_type == 0 and elapsed > 2.0:
            return True

        return False

    def _clear_intersection(self) -> None:
        """Reset intersection state after completing a manoeuvre."""
        self._int_entry_node = None
        self._int_central_node = None
        self._int_exit_node = None
        self._int_inter_node = None
        self._int_turn_dir = "straight"
        self._int_turn_type = 0
        self._int_target_heading = 0.0
        self._int_start_yaw = 0.0
        self._prev_int_steer = 0.0   # reset rate-limit state
        self.dbg_turn_dir = ""

    # ================================================================
    #  Parking (simplified)
    # ================================================================

    def _run_parking(self, now: float) -> StateOutput:
        cfg = self._cfg
        elapsed = now - self._state_enter_time
        if elapsed < 2.0:
            return StateOutput(target_speed=-cfg.slow_speed,
                               steering_override=0.6,
                               use_lane_keeping=False, state_name="PARKING")
        elif elapsed < 4.0:
            return StateOutput(target_speed=-cfg.slow_speed,
                               steering_override=-0.6,
                               use_lane_keeping=False, state_name="PARKING")
        elif elapsed < 6.0:
            return StateOutput(target_speed=cfg.slow_speed * 0.5,
                               steering_override=0.0,
                               use_lane_keeping=False, state_name="PARKING")
        else:
            return StateOutput(target_speed=0.0, steering_override=0.0,
                               use_lane_keeping=False, state_name="PARKING")

    # ================================================================
    #  Logging
    # ================================================================

    def _info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)

    def _warn(self, msg: str) -> None:
        if self._log:
            self._log.warn(msg)
