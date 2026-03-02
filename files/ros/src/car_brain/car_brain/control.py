#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control Module — Team Cathı / BFMC
====================================
Translates the high-level ``StateOutput`` from the state machine into
smooth, simulation-friendly Twist commands.

Features
--------
  • **PID steering controller** — proportional + integral + derivative
    gains for accurate lane-centring with zero steady-state error.
  • **Speed ramping** — acceleration and deceleration are clamped to
    ``max_accel`` / ``max_decel`` so the car accelerates and brakes
    smoothly (no instantaneous jumps).
  • **Low-pass filter on steering** — exponential moving average
    prevents high-frequency oscillation / jitter on the steering axis.
  • **Dead-zone** — very small errors are zeroed to avoid micro-jitter.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import time
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
#  PID Controller (single-axis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PIDController:
    """Discrete PID with anti-windup clamp."""

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integrator_max: float = 50.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integrator_max = integrator_max

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_time: float = time.monotonic()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.monotonic()

    def compute(self, error: float) -> float:
        now = time.monotonic()
        dt = now - self._prev_time
        if dt <= 0.0:
            dt = 1e-4

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup clamp)
        self._integral += error * dt
        self._integral = max(
            -self.integrator_max, min(self.integrator_max, self._integral)
        )
        i_term = self.ki * self._integral

        # Derivative
        d_term = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = now

        output = p_term + i_term + d_term
        return max(self.output_min, min(self.output_max, output))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vehicle controller
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VehicleController:
    """
    Converts ``(target_speed, lateral_error)`` into a smooth ``TwistCommand``.

    Usage::

        ctrl = VehicleController(cfg)
        cmd = ctrl.compute(target_speed=0.2, lateral_error=15.0)
    """

    def __init__(self, cfg: DrivingConfig) -> None:
        self._cfg = cfg

        # PID for steering (lateral error → angular.z)
        self._steer_pid = PIDController(
            kp=cfg.steering_kp,
            ki=cfg.steering_ki,
            kd=cfg.steering_kd,
            output_min=-cfg.max_steering,
            output_max=cfg.max_steering,
        )

        # ── Smoothing state ─────────────────────────────────────
        self._current_speed: float = 0.0     # ramped speed
        self._filtered_steer: float = 0.0    # EMA-filtered steering
        self._last_time: float = time.monotonic()

    def reset(self) -> None:
        self._steer_pid.reset()
        self._current_speed = 0.0
        self._filtered_steer = 0.0
        self._last_time = time.monotonic()

    def compute(
        self, target_speed: float, lateral_error: float,
    ) -> TwistCommand:
        """
        Produce a smooth twist command.

        Parameters
        ----------
        target_speed : float
            Desired forward speed (m/s).  May be 0 for a stop.
        lateral_error : float
            Pixel offset between desired lane-centre and image centre.
            Positive → car should steer right.

        Returns
        -------
        TwistCommand
        """
        now = time.monotonic()
        dt = now - self._last_time
        if dt <= 0.0:
            dt = 1e-4
        self._last_time = now

        # ── Speed ramping ───────────────────────────────────────
        speed_diff = target_speed - self._current_speed
        if speed_diff > 0:
            # Accelerating
            max_step = self._cfg.max_accel * dt
            self._current_speed += min(speed_diff, max_step)
        else:
            # Decelerating / braking
            max_step = self._cfg.max_decel * dt
            self._current_speed += max(speed_diff, -max_step)

        # Clamp
        self._current_speed = max(
            0.0, min(self._cfg.cruise_speed, self._current_speed)
        )

        # ── Steering PID ────────────────────────────────────────
        # Dead-zone: ignore tiny errors (< 2 px) to avoid micro-jitter
        if abs(lateral_error) < 2.0:
            lateral_error = 0.0

        raw_steer = self._steer_pid.compute(lateral_error)

        # Low-pass (EMA) filter
        alpha = self._cfg.steering_alpha
        self._filtered_steer = (
            alpha * raw_steer + (1.0 - alpha) * self._filtered_steer
        )

        return TwistCommand(
            linear_x=self._current_speed,
            angular_z=self._filtered_steer,
        )
