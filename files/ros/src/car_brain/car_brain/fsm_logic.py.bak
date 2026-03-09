#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-State Machine — Team Cathı / BFMC
==========================================
Port of VROOM-BFMC-Simulator ``main_FSM.py`` adapted for:
  • ROS 2 (rclpy)
  • New ``trackgraph.graphml`` parsed dynamically via TrackGraph
  • Standard ``geometry_msgs/Twist`` on ``/automobile/command``
  • Classical lane detection + YOLO perception pipeline

States
------
  INIT                 → waiting for first camera frame
  LANE_FOLLOWING       → normal cruise with PID lane-centering
  INTERSECTION_APPROACH→ slowing down, approaching an intersection
  INTERSECTION_STOP    → full stop before intersection (stop line/sign)
  INTERSECTION_NAV     → graph-guided turn through intersection
  STOPPED              → temporary stop (pedestrian / red light)
  ROUNDABOUT_NAV       → roundabout traversal
  PARKING              → parallel-parking manoeuvre
  HIGHWAY              → high-speed lane following
  EMERGENCY_STOP       → obstacle / no-entry / roadblock

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

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
#  Driving FSM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingFSM:
    """
    Graph-aware finite-state machine for autonomous driving.

    Call ``tick(detections, lane_state, pose, dt)`` every control cycle.
    It returns a ``StateOutput`` that the control node translates into
    a ``Twist`` message.

    Parameters
    ----------
    cfg : DrivingConfig
    track_graph : TrackGraph | None
        If ``None``, graph-based intersection navigation is disabled
        and the FSM falls back to timed-turn heuristics.
    logger
        Optional ROS logger for debug prints.
    """

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
        self._nav_path: List[str] = []
        self._full_path: List[str] = []
        self._intersection_target: Optional[str] = None
        self._intersection_start_yaw: float = 0.0
        self._intersection_enter_time: float = 0.0

        # ── Pose tracking (from odometry) ───────────────────────
        self._pos_x: float = 0.0
        self._pos_y: float = 0.0
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

        # ── Initialize graph path ───────────────────────────────
        if self._tg is not None:
            self._full_path = self._tg.shortest_path(
                cfg.nav_source_node, cfg.nav_target_node
            )
            self._nav_path = self._tg.navigation_path(self._full_path)
            self._info(
                f"Graph path: {len(self._full_path)} nodes, "
                f"nav path: {len(self._nav_path)} nodes"
            )

    # ================================================================
    #  PUBLIC API
    # ================================================================

    @property
    def state(self) -> State:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    def update_pose(self, x: float, y: float, yaw_deg: float) -> None:
        """Call every cycle with the latest odometry-derived pose."""
        self._pos_x = x
        self._pos_y = y
        self._yaw = yaw_deg

        # Also trim the navigation path
        if self._tg is not None and self._nav_path:
            self._nav_path, finished = self._tg.update_path(
                self._nav_path, x, y, self._cfg.nav_target_node
            )
            if finished:
                self._info("Navigation target reached!")

    def tick(
        self,
        detections: List[Dict],
        lane: Dict,
        traffic_light: str = "unknown",
    ) -> StateOutput:
        """
        Advance the FSM by one control cycle.

        Parameters
        ----------
        detections : list of dict
            Each dict has keys: class, conf, bbox, area.
        lane : dict
            Keys: cte, heading_error, left_valid, right_valid,
            both_valid, dashed, fallback.
        traffic_light : str
            "red" | "yellow" | "green" | "unknown".
        """
        now = time.monotonic()
        self._last_frame_time = now

        # ── Pre-compute detection flags ─────────────────────────
        flags = self._compute_detection_flags(detections, traffic_light)

        # ── State transitions ───────────────────────────────────
        new_state = self._decide_transition(flags, lane, now)
        if new_state != self._state:
            self._prev_state = self._state
            self._info(f"FSM: {self._state.name} → {new_state.name}")
            self._state = new_state
            self._state_enter_time = now

        # ── Compute output for current state ────────────────────
        output = self._compute_output(flags, lane, now)
        return output

    # ================================================================
    #  Detection flag computation
    # ================================================================

    def _compute_detection_flags(
        self, detections: List[Dict], traffic_light: str
    ) -> Dict[str, Any]:
        """Debounce raw YOLO detections into stable boolean flags."""
        # Identify interesting detections and their largest bounding-box area
        seen: Dict[str, float] = {}
        for d in detections:
            cls = d.get("class", "")
            area = d.get("area", 0.0)
            if cls not in seen or area > seen[cls]:
                seen[cls] = area

        img_area = self._cfg.image_width * self._cfg.image_height

        def _ratio(cls: str) -> float:
            return seen.get(cls, 0.0) / max(img_area, 1)

        # Compute per-class relative sizes
        stop_ratio = max(_ratio(CLASS_STOP_SIGN), _ratio(CLASS_STOP_LINE))
        crosswalk_ratio = _ratio(CLASS_CROSSWALK)
        pedestrian_ratio = _ratio(CLASS_PEDESTRIAN)
        roundabout_ratio = _ratio(CLASS_ROUNDABOUT_SIGN)
        parking_ratio = _ratio(CLASS_PARKING_SIGN)
        no_entry_ratio = _ratio(CLASS_PROHIBITED_SIGN) + _ratio(CLASS_ROADBLOCK)

        # Debounced flags
        flags: Dict[str, Any] = {
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
            # Graph awareness
            "intersection_ahead": (
                self._tg.check_intersection_ahead(self._nav_path)
                if self._tg is not None and self._nav_path
                else False
            ),
        }
        return flags

    # ================================================================
    #  State transition logic
    # ================================================================

    def _decide_transition(
        self, flags: Dict[str, Any], lane: Dict, now: float
    ) -> State:
        """Determine the next state based on current state + flags."""
        elapsed = now - self._state_enter_time

        # ── INIT ─────────────────────────────────────────────────
        if self._state == State.INIT:
            # Wait until we get valid lane data
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

        # ── State-specific transitions ──────────────────────────

        if self._state == State.LANE_FOLLOWING:
            if flags["stop_close"]:
                return State.INTERSECTION_STOP
            if flags["stop_approaching"] or flags["intersection_ahead"]:
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
            if flags["stop_close"]:
                return State.INTERSECTION_STOP
            # If stop sign/line disappeared, we may have passed it
            if not flags["stop_approaching"] and not flags["intersection_ahead"]:
                return State.LANE_FOLLOWING
            return State.INTERSECTION_APPROACH

        elif self._state == State.INTERSECTION_STOP:
            if elapsed >= self._cfg.intersection_stop_sec:
                # Start navigating the intersection
                self._prepare_intersection_nav()
                return State.INTERSECTION_NAV
            return State.INTERSECTION_STOP

        elif self._state == State.INTERSECTION_NAV:
            if elapsed >= self._cfg.intersection_turn_sec + 2.0:
                # Safety timeout — force exit
                return State.LANE_FOLLOWING
            # Graph-based completion check
            if self._intersection_target is None:
                return State.LANE_FOLLOWING
            return State.INTERSECTION_NAV

        elif self._state == State.STOPPED:
            if now >= self._stop_until:
                if not flags["pedestrian"] and not flags["traffic_red"]:
                    return State.LANE_FOLLOWING
            return State.STOPPED

        elif self._state == State.ROUNDABOUT_NAV:
            # Exit when no longer seeing roundabout sign and time has passed
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
        """Produce the StateOutput for the current state."""
        cfg = self._cfg

        if self._state == State.INIT:
            return StateOutput(
                target_speed=0.0,
                use_lane_keeping=False,
                state_name="INIT",
            )

        elif self._state == State.LANE_FOLLOWING:
            return StateOutput(
                target_speed=cfg.cruise_speed,
                use_lane_keeping=True,
                state_name="LANE_FOLLOWING",
            )

        elif self._state == State.INTERSECTION_APPROACH:
            return StateOutput(
                target_speed=cfg.slow_speed,
                use_lane_keeping=True,
                state_name="INTERSECTION_APPROACH",
            )

        elif self._state == State.INTERSECTION_STOP:
            return StateOutput(
                target_speed=0.0,
                steering_override=0.0,
                use_lane_keeping=False,
                state_name="INTERSECTION_STOP",
            )

        elif self._state == State.INTERSECTION_NAV:
            steer = self._run_intersection_nav(now)
            return StateOutput(
                target_speed=cfg.intersection_speed,
                steering_override=steer,
                use_lane_keeping=False,
                state_name="INTERSECTION_NAV",
            )

        elif self._state == State.STOPPED:
            return StateOutput(
                target_speed=0.0,
                steering_override=0.0,
                use_lane_keeping=False,
                state_name="STOPPED",
            )

        elif self._state == State.ROUNDABOUT_NAV:
            return StateOutput(
                target_speed=cfg.slow_speed,
                use_lane_keeping=True,
                state_name="ROUNDABOUT_NAV",
            )

        elif self._state == State.PARKING:
            return self._run_parking(now)

        elif self._state == State.HIGHWAY:
            return StateOutput(
                target_speed=cfg.highway_speed,
                use_lane_keeping=True,
                state_name="HIGHWAY",
            )

        elif self._state == State.EMERGENCY_STOP:
            return StateOutput(
                target_speed=0.0,
                steering_override=0.0,
                use_lane_keeping=False,
                state_name="EMERGENCY_STOP",
            )

        # Fallback
        return StateOutput(
            target_speed=0.0,
            use_lane_keeping=False,
            state_name=self._state.name,
        )

    # ================================================================
    #  Intersection navigation helpers
    # ================================================================

    def _prepare_intersection_nav(self) -> None:
        """Set up intersection traversal using the track graph."""
        self._intersection_enter_time = time.monotonic()
        self._intersection_start_yaw = self._yaw

        if self._tg is not None and self._nav_path:
            target, finished = self._tg.find_target_after_intersection(
                self._nav_path
            )
            self._intersection_target = target
            if target:
                self._info(f"Intersection target: node {target}")
            else:
                self._info("No intersection target found — will use timed turn")
        else:
            self._intersection_target = None

    def _run_intersection_nav(self, now: float) -> float:
        """
        Compute steering during intersection traversal.

        If the track graph is available, delegate to
        ``TrackGraph.intersection_steer``; otherwise fall back to a
        fixed timed turn (like VROOM's default).

        Returns normalised steering in [-1, 1].
        """
        elapsed = now - self._intersection_enter_time

        if self._tg is not None and self._intersection_target is not None:
            raw_steer, reached = self._tg.intersection_steer(
                path=self._nav_path,
                complete_path=self._full_path,
                target_node=self._intersection_target,
                x=self._pos_x,
                y=self._pos_y,
                yaw=self._yaw,
                start_yaw=self._intersection_start_yaw,
                elapsed_time=elapsed,
                speed=self._cfg.intersection_speed,
            )
            if reached:
                self._intersection_target = None
                return 0.0
            # Normalise raw steer (±20 range from TrackGraph) to [-1, 1]
            return max(-1.0, min(1.0, raw_steer / 20.0))

        # Fallback: timed turn (from VROOM heuristic)
        if elapsed < self._cfg.intersection_turn_sec:
            return self._cfg.intersection_turn_steer
        self._intersection_target = None
        return 0.0

    # ================================================================
    #  Parking manoeuvre (simplified)
    # ================================================================

    def _run_parking(self, now: float) -> StateOutput:
        """
        Simple parallel-parking sequence:
          Phase 1 (0-2s): steer right + reverse
          Phase 2 (2-4s): steer left + reverse
          Phase 3 (4-6s): straighten
          Phase 4 (6-8s): stop
        """
        cfg = self._cfg
        elapsed = now - self._state_enter_time

        if elapsed < 2.0:
            return StateOutput(
                target_speed=-cfg.slow_speed,
                steering_override=0.6,
                use_lane_keeping=False,
                state_name="PARKING",
            )
        elif elapsed < 4.0:
            return StateOutput(
                target_speed=-cfg.slow_speed,
                steering_override=-0.6,
                use_lane_keeping=False,
                state_name="PARKING",
            )
        elif elapsed < 6.0:
            return StateOutput(
                target_speed=cfg.slow_speed * 0.5,
                steering_override=0.0,
                use_lane_keeping=False,
                state_name="PARKING",
            )
        else:
            return StateOutput(
                target_speed=0.0,
                steering_override=0.0,
                use_lane_keeping=False,
                state_name="PARKING",
            )

    # ================================================================
    #  Logging helpers
    # ================================================================

    def _info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)

    def _warn(self, msg: str) -> None:
        if self._log:
            self._log.warn(msg)
