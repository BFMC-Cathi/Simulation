#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control Module — VROOM-Gazebo Hybrid
======================================
Takes the geometric steering angle (degrees) from lane detection and
converts it smoothly to the Ackermann plugin's angular.z (radians).

Pipeline:
  1. Receive angle_deg from lane detection (±25° range)
  2. NaN/inf guard
  3. Lightweight EMA (α = 0.35) to damp oscillation
  4. Linear map: degrees → radians (direct proportional, no VROOM
     normalizeSteer ±25→±35 re-mapping which was tuned for their
     physical car — we go straight deg→rad)
  5. Hard clamp to ±0.5 rad (Ackermann steering_limit)

No PID.  No complex filters.  Just smooth geometric steering.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from car_brain.config import DrivingConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Command output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class TwistCommand:
    """Final velocity command sent to the vehicle."""
    linear_x: float = 0.0   # m/s
    angular_z: float = 0.0   # rad/s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PID Controller (kept for interface compatibility — unused)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PIDController:
    """Placeholder — not used in hybrid port."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0,
                 output_min=-1.0, output_max=1.0,
                 integrator_max=50.0):
        pass

    def reset(self):
        pass

    def compute(self, error: float) -> float:
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vehicle controller — smooth geometric steering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VehicleController:
    """
    Converts a steering angle in degrees to a smooth Ackermann command.

    Key design:
      • EMA (exponential moving average) with α = 0.35 to damp
        frame-to-frame jitter without adding lag.
      • Direct deg → rad conversion (π/180), no re-mapping.
      • Hard clamp to ±0.5 rad (the plugin's steering_limit).

    The ``lateral_error`` parameter carries the **VROOM-style angle
    in degrees** (positive = steer right), **not** a pixel offset.
    """

    # EMA smoothing factor.  Higher = more responsive, lower = smoother.
    _EMA_ALPHA = 0.35

    # Maximum steering output (radians).  Matches AckermannSteering
    # plugin ``steering_limit`` in the SDF.
    _MAX_STEER_RAD = 0.5

    def __init__(self, cfg: DrivingConfig) -> None:
        self._cfg = cfg
        self._ema_steer: float = 0.0   # EMA state (radians)

        # ── Debug attributes (control_state_node reads these) ───
        self.dbg_raw_cte: float = 0.0
        self.dbg_norm_cte: float = 0.0
        self.dbg_pid_out: float = 0.0
        self.dbg_heading_ff: float = 0.0
        self.dbg_integral: float = 0.0
        self.dbg_filtered_steer: float = 0.0

    def reset(self) -> None:
        self._ema_steer = 0.0

    def compute(
        self,
        target_speed: float,
        lateral_error: float,
        heading_error: float = 0.0,
    ) -> TwistCommand:
        """
        Produce a twist command.

        Parameters
        ----------
        target_speed : Desired forward speed (m/s).
        lateral_error : Steering angle in **degrees** from lane det.
        heading_error : Unused (kept for interface compat).
        """
        angle_deg = lateral_error

        # ── Guard NaN / inf ─────────────────────────────────────
        if not math.isfinite(angle_deg):
            angle_deg = 0.0

        # ── Debug (raw) ─────────────────────────────────────────
        self.dbg_raw_cte = angle_deg
        self.dbg_heading_ff = 0.0
        self.dbg_integral = 0.0

        # ── Degrees → radians (direct, no re-mapping) ──────────
        # CRITICAL: VROOM convention: positive angle = steer RIGHT.
        # ROS 2 AckermannSteering: positive angular.z = turn LEFT.
        # We INVERT so that a positive VROOM angle becomes a
        # negative angular.z (= turn right).
        steer_raw_rad = -1.0 * angle_deg * (math.pi / 180.0)

        # ── Lightweight EMA to smooth oscillation ───────────────
        alpha = self._EMA_ALPHA
        self._ema_steer = alpha * steer_raw_rad + (1.0 - alpha) * self._ema_steer
        steer_rad = self._ema_steer

        # ── Hard clamp ──────────────────────────────────────────
        steer_rad = max(-self._MAX_STEER_RAD,
                        min(self._MAX_STEER_RAD, steer_rad))

        # ── Debug ───────────────────────────────────────────────
        self.dbg_norm_cte = angle_deg / 25.0 if angle_deg != 0 else 0.0
        self.dbg_pid_out = angle_deg      # "mapped deg" (now just raw deg)
        self.dbg_filtered_steer = steer_rad

        # ── Speed ───────────────────────────────────────────────
        speed = max(-1.0, min(1.0, target_speed))

        return TwistCommand(
            linear_x=speed,
            angular_z=steer_rad,
        )
