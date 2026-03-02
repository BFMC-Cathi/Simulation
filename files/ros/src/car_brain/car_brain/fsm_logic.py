#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSM Logic — Finite State Machine for Autonomous Driving
=========================================================
A clean, self-contained FSM using the ``transitions`` library pattern
(implemented without external dependencies for maximum portability).

States
------
  INIT                 — Waiting for first valid camera frame
  LANE_FOLLOWING       — Default: cruise speed + PID lane centring
  INTERSECTION_APPROACH— Stop sign / traffic light detected, decelerating
  STOPPED              — Fully stopped, timed hold before resuming
  ROUNDABOUT_NAV       — Navigating through the roundabout
  PARKING              — Executing parking manoeuvre

Transition triggers come from the ``/perception/objects`` topic
(YOLO detections) and are debounced to prevent flickering.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from car_brain.config import (
    DrivingConfig,
    CLASS_STOP_SIGN,
    CLASS_STOP_LINE,
    CLASS_TRAFFIC_LIGHT,
    CLASS_CROSSWALK,
    CLASS_PEDESTRIAN,
    CLASS_ROUNDABOUT_SIGN,
    CLASS_PARKING_SIGN,
    CLASS_HIGHWAY_ENTRY,
    CLASS_HIGHWAY_EXIT,
    CLASS_ROADBLOCK,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingState(Enum):
    """All possible driving states."""
    INIT = auto()
    LANE_FOLLOWING = auto()
    INTERSECTION_APPROACH = auto()
    STOPPED = auto()
    ROUNDABOUT_NAV = auto()
    PARKING = auto()
    HIGHWAY = auto()
    EMERGENCY_STOP = auto()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection container (lightweight — decoupled from perception.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class ObjectDetection:
    """Minimal detection data consumed by the FSM."""
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def y2_ratio(self) -> float:
        """Must be set externally by dividing y2 by image height."""
        return 0.0  # placeholder — use helper function

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def proximity_ratio(det: ObjectDetection, img_height: int) -> float:
    """Normalised proximity: how close the detection's bottom edge is
    to the bottom of the image (0.0 = top, 1.0 = bottom)."""
    if img_height <= 0:
        return 0.0
    return det.y2 / float(img_height)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FSM Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class FSMOutput:
    """What the FSM tells the controller to do each tick."""
    state: DrivingState = DrivingState.INIT
    target_speed: float = 0.0
    reason: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Debounce counter helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _DebounceCounter:
    """Increment while condition is True, reset to 0 when False."""

    def __init__(self) -> None:
        self._count: int = 0

    def tick(self, condition: bool) -> int:
        if condition:
            self._count += 1
        else:
            self._count = 0
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Finite State Machine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingFSM:
    """
    Deterministic FSM that maps object detections + traffic light
    colour to an ``FSMOutput`` (target speed + state).

    Usage from ``control_state_node``::

        fsm = DrivingFSM(cfg, logger)
        output = fsm.update(detections, tl_colour, img_h)
    """

    def __init__(self, cfg: DrivingConfig, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger

        self._state = DrivingState.INIT
        self._prev_state = DrivingState.INIT

        # ── Debounce counters ───────────────────────────────────
        self._stop_ctr = _DebounceCounter()
        self._tl_red_ctr = _DebounceCounter()
        self._roundabout_ctr = _DebounceCounter()
        self._parking_ctr = _DebounceCounter()
        self._highway_ctr = _DebounceCounter()
        self._highway_exit_ctr = _DebounceCounter()
        self._clear_ctr = _DebounceCounter()

        # ── Timers ──────────────────────────────────────────────
        self._stop_entry_time: float = 0.0
        self._last_frame_time: float = time.monotonic()

        # ── Stop-sign cooldown (avoid re-stopping immediately) ──
        self._stop_cooldown_until: float = 0.0

    # ── Public ──────────────────────────────────────────────────

    @property
    def state(self) -> DrivingState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    def notify_frame(self) -> None:
        """Call every time a valid camera frame arrives."""
        self._last_frame_time = time.monotonic()

    def update(
        self,
        detections: List[ObjectDetection],
        traffic_light_colour: str,
        image_height: int,
    ) -> FSMOutput:
        """
        Core FSM tick.

        Parameters
        ----------
        detections
            List of YOLO object detections for the current frame.
        traffic_light_colour
            ``"red"`` | ``"yellow"`` | ``"green"`` | ``"unknown"``
        image_height
            Frame height in pixels (for normalised proximity).
        """
        now = time.monotonic()
        output = FSMOutput(state=self._state)
        db = self._cfg.debounce_frames

        # ════════════════════════════════════════════════════════
        #  Frame-drop watchdog
        # ════════════════════════════════════════════════════════
        if now - self._last_frame_time > self._cfg.frame_timeout_sec:
            self._transition(DrivingState.EMERGENCY_STOP)
            output.state = self._state
            output.target_speed = 0.0
            output.reason = "frame timeout"
            return output

        # Recover from emergency stop
        if self._state == DrivingState.EMERGENCY_STOP:
            if now - self._last_frame_time < self._cfg.frame_timeout_sec * 0.5:
                self._transition(DrivingState.LANE_FOLLOWING)

        # ════════════════════════════════════════════════════════
        #  Compute proximity triggers
        # ════════════════════════════════════════════════════════
        stop_classes = {CLASS_STOP_SIGN, CLASS_STOP_LINE}
        stop_near = self._any_near(
            detections, stop_classes, image_height,
            self._cfg.stop_hard_threshold
        )
        tl_near = self._any_near(
            detections, {CLASS_TRAFFIC_LIGHT}, image_height,
            self._cfg.stop_soft_threshold
        )
        roundabout_near = self._any_near(
            detections, {CLASS_ROUNDABOUT_SIGN}, image_height,
            self._cfg.roundabout_threshold
        )
        parking_near = self._any_near(
            detections, {CLASS_PARKING_SIGN}, image_height,
            self._cfg.parking_threshold
        )
        highway_entry = self._any_near(
            detections, {CLASS_HIGHWAY_ENTRY}, image_height,
            self._cfg.stop_soft_threshold
        )
        highway_exit = self._any_near(
            detections, {CLASS_HIGHWAY_EXIT}, image_height,
            self._cfg.stop_soft_threshold
        )

        tl_is_red = traffic_light_colour in ("red", "yellow") and tl_near
        tl_is_green = traffic_light_colour == "green"

        in_cooldown = now < self._stop_cooldown_until

        # ── Debounce ticks ──────────────────────────────────────
        self._stop_ctr.tick(stop_near and not in_cooldown)
        self._tl_red_ctr.tick(tl_is_red)
        self._roundabout_ctr.tick(roundabout_near)
        self._parking_ctr.tick(parking_near)
        self._highway_ctr.tick(highway_entry)
        self._highway_exit_ctr.tick(highway_exit)

        nothing_close = (
            not stop_near and not tl_is_red and not roundabout_near
            and not parking_near
        )
        self._clear_ctr.tick(nothing_close)

        # ════════════════════════════════════════════════════════
        #  STATE TRANSITIONS
        # ════════════════════════════════════════════════════════

        if self._state == DrivingState.INIT:
            # Transition out of INIT once we get a frame
            self._transition(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.LANE_FOLLOWING:
            if self._stop_ctr.count >= db:
                self._transition(DrivingState.INTERSECTION_APPROACH)
            elif self._tl_red_ctr.count >= db:
                self._transition(DrivingState.INTERSECTION_APPROACH)
            elif self._roundabout_ctr.count >= db:
                self._transition(DrivingState.ROUNDABOUT_NAV)
            elif self._parking_ctr.count >= db:
                self._transition(DrivingState.PARKING)
            elif self._highway_ctr.count >= db:
                self._transition(DrivingState.HIGHWAY)

        elif self._state == DrivingState.INTERSECTION_APPROACH:
            # If it was a traffic light and it turned green → go
            if tl_is_green and self._tl_red_ctr.count == 0:
                self._transition(DrivingState.LANE_FOLLOWING)
                self._reset_counters()
            # If the object is very close → full stop
            elif stop_near and self._stop_ctr.count >= db:
                self._stop_entry_time = now
                self._transition(DrivingState.STOPPED)
            elif tl_is_red:
                # Stay approaching / decelerate
                pass
            elif self._clear_ctr.count >= db * 3:
                # False alarm — object left FOV
                self._transition(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.STOPPED:
            # If red light — wait until green
            if tl_is_red:
                self._stop_entry_time = now  # keep resetting
            elif tl_is_green:
                self._transition(DrivingState.LANE_FOLLOWING)
                self._stop_cooldown_until = now + 5.0
                self._reset_counters()
            else:
                elapsed = now - self._stop_entry_time
                if elapsed >= self._cfg.stop_hold_sec:
                    self._transition(DrivingState.LANE_FOLLOWING)
                    self._stop_cooldown_until = now + 5.0
                    self._reset_counters()

        elif self._state == DrivingState.ROUNDABOUT_NAV:
            # Exit roundabout when no more roundabout signs in view
            if self._clear_ctr.count >= db * 4:
                self._transition(DrivingState.LANE_FOLLOWING)
                self._reset_counters()

        elif self._state == DrivingState.PARKING:
            # Parking logic placeholder — stays until clear
            if self._clear_ctr.count >= db * 5:
                self._transition(DrivingState.LANE_FOLLOWING)
                self._reset_counters()

        elif self._state == DrivingState.HIGHWAY:
            if self._highway_exit_ctr.count >= db:
                self._transition(DrivingState.LANE_FOLLOWING)
                self._reset_counters()

        elif self._state == DrivingState.EMERGENCY_STOP:
            pass  # recovery handled above

        # ════════════════════════════════════════════════════════
        #  STATE → SPEED MAPPING
        # ════════════════════════════════════════════════════════
        output.state = self._state
        speed_map = {
            DrivingState.INIT: 0.0,
            DrivingState.LANE_FOLLOWING: self._cfg.cruise_speed,
            DrivingState.INTERSECTION_APPROACH: self._cfg.slow_speed,
            DrivingState.STOPPED: 0.0,
            DrivingState.ROUNDABOUT_NAV: self._cfg.slow_speed,
            DrivingState.PARKING: self._cfg.slow_speed * 0.5,
            DrivingState.HIGHWAY: self._cfg.highway_speed,
            DrivingState.EMERGENCY_STOP: 0.0,
        }
        output.target_speed = speed_map.get(self._state, 0.0)
        output.reason = self._state.name
        return output

    # ── Internal helpers ────────────────────────────────────────

    def _transition(self, new_state: DrivingState) -> None:
        if new_state != self._state:
            old_name = self._state.name
            self._prev_state = self._state
            self._state = new_state
            if self._logger:
                self._logger.info(f"FSM: {old_name} → {new_state.name}")

    def _reset_counters(self) -> None:
        self._stop_ctr.reset()
        self._tl_red_ctr.reset()
        self._roundabout_ctr.reset()
        self._parking_ctr.reset()
        self._highway_ctr.reset()
        self._highway_exit_ctr.reset()
        self._clear_ctr.reset()

    @staticmethod
    def _any_near(
        detections: List[ObjectDetection],
        class_names: Set[str],
        image_height: int,
        threshold: float,
    ) -> bool:
        """Return True if any detection of the given classes exceeds
        the proximity threshold."""
        for d in detections:
            if d.class_name in class_names:
                if proximity_ratio(d, image_height) > threshold:
                    return True
        return False
