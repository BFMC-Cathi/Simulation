#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Machine Module — Team Cathı / BFMC
=========================================
A robust, enum-based Finite State Machine (FSM) for autonomous driving.

States
------
  LANE_FOLLOWING       – Default cruising & lane-centring
  APPROACHING_STOP     – Decelerating towards a stop line / sign
  STOPPED              – Fully stopped; waiting ``stop_hold_sec``
  APPROACHING_CROSSWALK– Slowing down for a detected crosswalk
  CROSSWALK_WAIT       – Stopped because a pedestrian is on the crosswalk
  TRAFFIC_LIGHT_WAIT   – Stopped at a red / yellow traffic light
  EMERGENCY_STOP       – No camera frames received → safe halt

Design principles:
  • **Debounced transitions** — a trigger must persist for ``N``
    consecutive control ticks before a transition fires, avoiding
    flickering caused by false-positive detections.
  • **Hysteresis** — entry thresholds are stricter than exit thresholds
    to prevent oscillation near boundaries.
  • **Timed holds** — the STOPPED state auto-transitions back to
    LANE_FOLLOWING after ``stop_hold_sec``.
  • **Edge-case safety** — frame-drop watchdog triggers EMERGENCY_STOP;
    re-entering LANE_FOLLOWING once frames resume.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional

from car_brain.config import (
    DrivingConfig,
    CLASS_STOP_LINE,
    CLASS_STOP_SIGN,
    CLASS_CROSSWALK,
    CLASS_PEDESTRIAN,
    CLASS_TRAFFIC_LIGHT,
)
from car_brain.perception import Detection, PerceptionResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingState(Enum):
    LANE_FOLLOWING = auto()
    APPROACHING_STOP = auto()
    STOPPED = auto()
    APPROACHING_CROSSWALK = auto()
    CROSSWALK_WAIT = auto()
    TRAFFIC_LIGHT_WAIT = auto()
    EMERGENCY_STOP = auto()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FSM output — consumed by the control module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class StateOutput:
    """What the state machine tells the controller to do."""
    target_speed: float = 0.0
    lateral_error: float = 0.0   # pixel error for lane-centring PID
    state: DrivingState = DrivingState.LANE_FOLLOWING


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State Machine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivingStateMachine:
    """
    Deterministic FSM that maps perception results to a ``StateOutput``.

    Usage::

        sm = DrivingStateMachine(cfg)
        output = sm.update(perception_result, image_width, image_height)
    """

    def __init__(self, cfg: DrivingConfig, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger

        self._state = DrivingState.LANE_FOLLOWING
        self._prev_state = self._state

        # ── Debounce counters ───────────────────────────────────
        self._stop_trigger_count: int = 0
        self._crosswalk_trigger_count: int = 0
        self._tl_red_trigger_count: int = 0
        self._clear_count: int = 0  # frames with no obstacle

        # ── Timers ──────────────────────────────────────────────
        self._stop_entry_time: float = 0.0
        self._last_frame_time: float = time.monotonic()

    # ── Public API ──────────────────────────────────────────────

    @property
    def state(self) -> DrivingState:
        return self._state

    def notify_frame_received(self) -> None:
        """Call every time a camera frame is successfully received."""
        self._last_frame_time = time.monotonic()

    def update(
        self,
        result: Optional[PerceptionResult],
        image_width: int,
        image_height: int,
    ) -> StateOutput:
        """
        Core tick — evaluate perception, transition states, produce output.

        Parameters
        ----------
        result : PerceptionResult | None
            Latest perception output (``None`` when no result is available yet).
        image_width, image_height : int
            Frame dimensions for normalisation.
        """
        output = StateOutput(state=self._state)
        now = time.monotonic()

        # ── Frame-drop watchdog ─────────────────────────────────
        if now - self._last_frame_time > self._cfg.frame_timeout_sec:
            self._transition_to(DrivingState.EMERGENCY_STOP)
            output.state = self._state
            output.target_speed = self._cfg.stop_speed
            return output

        # Recover from emergency stop once frames resume
        if self._state == DrivingState.EMERGENCY_STOP:
            if now - self._last_frame_time < self._cfg.frame_timeout_sec * 0.5:
                self._transition_to(DrivingState.LANE_FOLLOWING)

        if result is None:
            # No inference yet — hold previous output
            output.target_speed = self._cfg.cruise_speed if self._state == DrivingState.LANE_FOLLOWING else self._cfg.stop_speed
            return output

        detections = result.detections
        tl_colour = result.traffic_light_colour

        # ── Compute proximity triggers ──────────────────────────
        stop_near = self._check_proximity(
            detections, (CLASS_STOP_LINE, CLASS_STOP_SIGN),
            image_height, self._cfg.stop_hard_threshold,
        )
        stop_approaching = self._check_proximity(
            detections, (CLASS_STOP_LINE, CLASS_STOP_SIGN),
            image_height, self._cfg.stop_soft_threshold,
        )
        crosswalk_near = self._check_proximity(
            detections, (CLASS_CROSSWALK,),
            image_height, self._cfg.crosswalk_threshold,
        )
        pedestrian_present = any(
            d.class_name == CLASS_PEDESTRIAN for d in detections
        )
        tl_is_red = tl_colour in ("red", "yellow")
        tl_is_green = tl_colour == "green"

        # ── Debounce logic ──────────────────────────────────────
        self._stop_trigger_count = (
            self._stop_trigger_count + 1 if stop_near else 0
        )
        self._crosswalk_trigger_count = (
            self._crosswalk_trigger_count + 1 if crosswalk_near else 0
        )
        self._tl_red_trigger_count = (
            self._tl_red_trigger_count + 1 if tl_is_red else 0
        )
        # Clear counter: no stop / crosswalk / red light
        if not stop_approaching and not crosswalk_near and not tl_is_red:
            self._clear_count += 1
        else:
            self._clear_count = 0

        debounce = self._cfg.debounce_frames

        # ════════════════════════════════════════════════════════
        #  STATE TRANSITIONS
        # ════════════════════════════════════════════════════════

        if self._state == DrivingState.LANE_FOLLOWING:
            if self._stop_trigger_count >= debounce:
                self._transition_to(DrivingState.APPROACHING_STOP)
            elif self._tl_red_trigger_count >= debounce:
                self._transition_to(DrivingState.TRAFFIC_LIGHT_WAIT)
            elif self._crosswalk_trigger_count >= debounce:
                if pedestrian_present:
                    self._transition_to(DrivingState.CROSSWALK_WAIT)
                else:
                    self._transition_to(DrivingState.APPROACHING_CROSSWALK)

        elif self._state == DrivingState.APPROACHING_STOP:
            if stop_near and self._stop_trigger_count >= debounce:
                self._transition_to(DrivingState.STOPPED)
                self._stop_entry_time = now
            elif self._clear_count >= debounce * 2:
                # False alarm — obstacle left view
                self._transition_to(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.STOPPED:
            elapsed = now - self._stop_entry_time
            if elapsed >= self._cfg.stop_hold_sec:
                self._transition_to(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.APPROACHING_CROSSWALK:
            if pedestrian_present:
                self._transition_to(DrivingState.CROSSWALK_WAIT)
            elif self._clear_count >= debounce * 2:
                self._transition_to(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.CROSSWALK_WAIT:
            if not pedestrian_present and self._clear_count >= debounce:
                self._transition_to(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.TRAFFIC_LIGHT_WAIT:
            if tl_is_green and self._tl_red_trigger_count == 0:
                self._transition_to(DrivingState.LANE_FOLLOWING)
            elif self._clear_count >= debounce * 3:
                # Light left view (possibly after passing)
                self._transition_to(DrivingState.LANE_FOLLOWING)

        elif self._state == DrivingState.EMERGENCY_STOP:
            pass  # recovery handled above

        # ════════════════════════════════════════════════════════
        #  STATE → OUTPUT MAPPING
        # ════════════════════════════════════════════════════════
        output.state = self._state
        output.lateral_error = self._compute_lateral_error(
            detections, image_width,
        )

        if self._state == DrivingState.LANE_FOLLOWING:
            output.target_speed = self._cfg.cruise_speed

        elif self._state == DrivingState.APPROACHING_STOP:
            # Proportional slow-down based on proximity
            prox = self._max_proximity(
                detections, (CLASS_STOP_LINE, CLASS_STOP_SIGN), image_height,
            )
            factor = max(0.0, 1.0 - prox)
            output.target_speed = self._cfg.cruise_speed * factor

        elif self._state in (
            DrivingState.STOPPED,
            DrivingState.CROSSWALK_WAIT,
            DrivingState.TRAFFIC_LIGHT_WAIT,
            DrivingState.EMERGENCY_STOP,
        ):
            output.target_speed = self._cfg.stop_speed

        elif self._state == DrivingState.APPROACHING_CROSSWALK:
            output.target_speed = self._cfg.slow_speed

        return output

    # ── Internal helpers ────────────────────────────────────────

    def _transition_to(self, new_state: DrivingState) -> None:
        if new_state != self._state:
            self._prev_state = self._state
            if self._logger:
                self._logger.info(
                    f"FSM: {self._state.name} → {new_state.name}"
                )
            self._state = new_state

    @staticmethod
    def _check_proximity(
        detections: List[Detection],
        class_names: tuple,
        image_height: int,
        threshold: float,
    ) -> bool:
        """Return True if any detection of *class_names* exceeds threshold."""
        if image_height <= 0:
            return False
        for d in detections:
            if d.class_name in class_names:
                if d.y2 / image_height > threshold:
                    return True
        return False

    @staticmethod
    def _max_proximity(
        detections: List[Detection],
        class_names: tuple,
        image_height: int,
    ) -> float:
        """Return the highest normalised y2 among matching detections."""
        if image_height <= 0:
            return 0.0
        vals = [
            d.y2 / image_height
            for d in detections
            if d.class_name in class_names
        ]
        return max(vals) if vals else 0.0

    def _compute_lateral_error(
        self,
        detections: List[Detection],
        image_width: int,
    ) -> float:
        """
        Proportional lateral error for lane-centring.

        Returns the pixel offset between the desired driving centre
        and the image centre.  Positive → car is too far left.
        """
        from car_brain.config import CLASS_LEFT_LANE, CLASS_RIGHT_LANE

        img_cx = image_width / 2.0
        left_xs = [d.cx for d in detections if d.class_name == CLASS_LEFT_LANE]
        right_xs = [d.cx for d in detections if d.class_name == CLASS_RIGHT_LANE]

        offset = self._cfg.single_lane_offset_ratio * image_width

        if left_xs and right_xs:
            avg_l = sum(left_xs) / len(left_xs)
            avg_r = sum(right_xs) / len(right_xs)
            desired = (avg_l + avg_r) / 2.0
        elif left_xs:
            desired = sum(left_xs) / len(left_xs) + offset
        elif right_xs:
            desired = sum(right_xs) / len(right_xs) - offset
        else:
            return 0.0  # no lane info — no correction

        return desired - img_cx
