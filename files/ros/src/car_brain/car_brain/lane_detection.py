#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Detection — VROOM-Gazebo Hybrid
======================================
VROOM's proven sliding-window + polynomial pipeline, **re-calibrated**
for the Gazebo simulator's camera physics.

Gazebo camera:
  • 640×480, HFOV ≈ 62°, mounted 20 cm high, pitched 15° down
  • Bright-white lane markings on dark asphalt (grayscale threshold)

Key differences from the strict VROOM port:
  1. BEV source trapezoid computed from Gazebo camera geometry,
     NOT VROOM's hardcoded (70, 350, 0, 443).
  2. Single-lane mode: offset by half the expected BEV lane width
     to estimate the lane centre — prevents edge-locking.
  3. 15-frame steering holdover when both lanes are lost.
  4. VROOM's core math is preserved: sliding window, polynomial fit,
     weighted-average error, ``90 - atan2(...)`` angle formula,
     stabilise_steering_angle.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from car_brain.config import DrivingConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class LaneState:
    """Published on ``/perception/lane_state``."""
    cte: float = 0.0              # steering angle in degrees
    heading_error: float = 0.0    # heading component (radians)
    left_fit_valid: bool = False
    right_fit_valid: bool = False
    both_lanes_valid: bool = False
    dashed_line: bool = False
    using_fallback: bool = False
    timestamp: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lane Detector — Gazebo-calibrated VROOM hybrid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LaneDetector:
    """
    Stateful lane detector.

    BEV calibration for Gazebo:
      Camera is at 20 cm height, pitched 15° down, HFOV ≈ 62°.
      The road horizon sits around y ≈ 260 in a 640×480 frame.
      We pick a trapezoid whose top edge starts just below the
      horizon and whose bottom edge spans the full width at the
      frame bottom.

    Source trapezoid (in 640×480 camera image):
      Top-left:     (~179, ~312)   ← narrow road strip below horizon
      Top-right:    (~461, ~312)
      Bottom-left:  ( ~32, ~456)   ← wide view above bonnet
      Bottom-right: (~608, ~456)

    Destination: full 640×480 BEV rectangle.
    """

    # ── Expected BEV lane width (pixels) ────────────────────────
    # On a correctly warped BEV the lane is approximately this wide.
    # Used to synthesise the missing lane when only one line is seen.
    EXPECTED_LANE_WIDTH_PX = 280

    def __init__(self, cfg: DrivingConfig) -> None:
        self._cfg = cfg
        self._width = cfg.bev_width    # 640
        self._height = cfg.bev_height  # 480

        # ── Gazebo-calibrated BEV source trapezoid ──────────────
        W, H = self._width, self._height
        # Top edge: 65% down the frame — safely below horizon,
        # only sees actual road surface (was 58% — too aggressive).
        top_y = int(H * 0.65)          # ≈ 312
        top_inset = int(W * 0.28)      # ≈ 179 from each side
        # Bottom edge: 95% down — avoids any bonnet pixels
        bot_y = int(H * 0.95)          # ≈ 456
        bot_inset = int(W * 0.05)      # ≈ 32 from each side

        self._src_points = np.float32([
            [top_inset,         top_y],         # top-left
            [W - top_inset,     top_y],         # top-right
            [bot_inset,         bot_y],         # bottom-left
            [W - bot_inset,     bot_y],         # bottom-right
        ])
        self._dst_points = np.float32([
            [0, 0],
            [W, 0],
            [0, H],
            [W, H],
        ])
        self._warp_matrix = cv2.getPerspectiveTransform(
            self._src_points, self._dst_points
        )
        self._inv_warp_matrix = cv2.getPerspectiveTransform(
            self._dst_points, self._src_points
        )

        # ── Sliding-window parameters (VROOM-style, tuned) ──────
        self._n_windows = 12
        self._margin = 80             # slightly tighter than VROOM's 100
        self._minpix = 40
        self._min_lane_pts = 800      # lowered: Gazebo lines are thinner

        # ── VROOM error parameters ──────────────────────────────
        self._num_lines = 20          # horizontal scan lines
        self._nose2wheel = 320        # VROOM constant

        # ── Steering stabilisation (VROOM LaneKeeping.steer) ────
        self._curr_steering_angle: float = 0.0
        self._max_angle_dev_both: float = 5.0   # degrees per frame
        self._max_angle_dev_one: float = 7.0     # slightly more responsive

        # ── 15-frame holdover ───────────────────────────────────
        self._no_lane_frames: int = 0
        self._HOLD_FRAMES: int = 15
        self._last_good_time: float = 0.0

        # ── Debug stats ─────────────────────────────────────────
        self.dbg_white_px: int = 0
        self.dbg_bev_nonzero: int = 0
        self.dbg_lane_width_px: float = 0.0
        self.dbg_center_error: float = 0.0

    # ================================================================
    #  Threshold — grayscale for Gazebo
    # ================================================================

    def _threshold(self, frame: np.ndarray) -> np.ndarray:
        """Simple grayscale threshold: Gazebo lanes are bright white."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return mask

    # ================================================================
    #  BEV warp
    # ================================================================

    def _warp_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            frame, self._warp_matrix, (self._width, self._height)
        )

    # ================================================================
    #  Sliding-window polyfit (VROOM logic, tuned thresholds)
    # ================================================================

    def _polyfit_sliding_window(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Sliding-window lane search → 2nd-order polynomial fits.
        Returns (left_fit, right_fit, debug_image).
        """
        h, w = frame.shape[:2]

        if frame.max() <= 0:
            out = np.dstack((frame, frame, frame))
            return None, None, out

        # ── Histogram to find lane bases ────────────────────────
        histogram = None
        for cutoff in [int(h * 0.66), int(h * 0.50), 0]:
            histogram = np.sum(frame[cutoff:, :], axis=0).astype(float)
            if histogram.max() > 0:
                break

        if histogram is None or histogram.max() == 0:
            out = np.dstack((frame, frame, frame))
            return None, None, out

        midpoint = w // 2
        # Ignore a narrow centre strip to avoid picking the same line
        gap = 20
        left_hist = histogram[: max(midpoint - gap, 1)]
        right_hist = histogram[midpoint + gap:]

        leftx_base = int(np.argmax(left_hist)) if left_hist.max() > 0 else 0
        rightx_base = (
            int(np.argmax(right_hist)) + midpoint + gap
            if right_hist.max() > 0 else w - 1
        )

        out = np.dstack((frame, frame, frame))
        window_height = max(h // self._n_windows, 1)

        nonzero = frame.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds: List[np.ndarray] = []
        right_lane_inds: List[np.ndarray] = []

        for win in range(self._n_windows):
            win_y_low = h - (1 + win) * window_height
            win_y_high = h - win * window_height

            # LEFT window
            xl_lo = max(leftx_current - self._margin, 0)
            xl_hi = min(leftx_current + self._margin, w)
            cv2.rectangle(out, (xl_lo, win_y_low), (xl_hi, win_y_high),
                          (0, 0, 255), 2)
            good_left = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                & (nonzerox >= xl_lo) & (nonzerox < xl_hi)
            ).nonzero()[0]
            left_lane_inds.append(good_left)

            # RIGHT window
            xr_lo = max(rightx_current - self._margin, 0)
            xr_hi = min(rightx_current + self._margin, w)
            cv2.rectangle(out, (xr_lo, win_y_low), (xr_hi, win_y_high),
                          (0, 255, 0), 2)
            good_right = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                & (nonzerox >= xr_lo) & (nonzerox < xr_hi)
            ).nonzero()[0]
            right_lane_inds.append(good_right)

            # Recenter
            if len(good_left) > self._minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self._minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = right_fit = None

        if len(leftx) >= self._min_lane_pts:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) >= self._min_lane_pts:
            right_fit = np.polyfit(righty, rightx, 2)

        # ── Same-line guard ─────────────────────────────────────
        if left_fit is not None and right_fit is not None:
            lx_bot = np.polyval(left_fit, h - 1)
            rx_bot = np.polyval(right_fit, h - 1)
            if abs(rx_bot - lx_bot) < 60:
                if len(leftx) >= len(rightx):
                    right_fit = None
                else:
                    left_fit = None

        # Colour detected pixels
        if len(left_lane_inds) > 0:
            out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        if len(right_lane_inds) > 0:
            out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 10, 255]

        return left_fit, right_fit, out

    # ================================================================
    #  Polynomial evaluation helpers
    # ================================================================

    def _get_poly_points(
        self, left_fit: np.ndarray, right_fit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        plot_y = np.linspace(0, self._height - 1, self._height)
        plot_xleft = np.polyval(left_fit, plot_y)
        plot_xright = np.polyval(right_fit, plot_y)
        return (plot_xleft.astype(int), plot_y.astype(int),
                plot_xright.astype(int), plot_y.astype(int))

    # ================================================================
    #  VROOM weighted-average error (exact logic)
    # ================================================================

    def _get_error(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Horizontal scan-line error computation (VROOM's get_error).
        Scans the midpoint-dot image at N horizontal lines and
        computes a weighted-average lateral offset from centre.
        """
        h, w = self._height, self._width
        num_lines = self._num_lines
        line_height = max(h // num_lines, 1)

        # Create scan-line mask
        line_im = np.zeros_like(frame)
        for i in range(num_lines):
            y_pos = h - line_height * i
            if 0 <= y_pos < h:
                cv2.line(line_im, (0, y_pos), (w, y_pos),
                         (255, 255, 255), 2)

        cut_im = np.bitwise_and(frame, line_im)
        nonzero = cut_im.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        mean_list = []
        for i in range(num_lines):
            y_val = i * line_height
            mask = (nonzeroy == y_val)
            x_coors = nonzerox[mask]
            if len(x_coors) > 0:
                mean_list.append(float(np.mean(x_coors)))

        if len(mean_list) > 0:
            weighted_mean = self._weighted_average(np.array(mean_list))
        else:
            weighted_mean = w / 2.0

        error = weighted_mean - w / 2.0
        return float(error), float(weighted_mean)

    @staticmethod
    def _weighted_average(arr: np.ndarray) -> float:
        """Linearly increasing weights (VROOM: weighted_average)."""
        n = len(arr)
        weights = np.arange(1, n + 1, dtype=float)
        return float(np.sum(arr * weights) / np.sum(weights))

    # ================================================================
    #  Steering stabilisation (VROOM LaneKeeping.steer)
    # ================================================================

    def _stabilize_steering_angle(
        self, curr: float, new: float, num_lanes: int,
    ) -> float:
        max_dev = (
            self._max_angle_dev_both if num_lanes >= 2
            else self._max_angle_dev_one
        )
        diff = new - curr
        if abs(diff) > max_dev:
            step = max_dev if diff > 0 else -max_dev
            return curr + step
        return new

    # ================================================================
    #  Plot midpoint dots (for error computation)
    # ================================================================

    @staticmethod
    def _plot_midpoints(
        left_x: np.ndarray, left_y: np.ndarray,
        right_x: np.ndarray, right_y: np.ndarray,
        h: int, w: int,
    ) -> np.ndarray:
        """Plot white dots at midpoints of left & right lane curves."""
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(left_x)):
            mx = (int(left_x[i]) + int(right_x[i])) // 2
            my = int(left_y[i])
            if 0 <= mx < w and 0 <= my < h:
                cv2.circle(out, (mx, my), 8, (255, 255, 255), -1)
        return out

    # ================================================================
    #  PUBLIC API — process()
    # ================================================================

    def process(
        self, frame: np.ndarray
    ) -> Tuple[LaneState, Optional[np.ndarray]]:
        """
        Full lane-detection pipeline.
        Returns (LaneState, debug_image).
        """
        state = LaneState(timestamp=time.monotonic())

        # 1. BEV warp
        warped = self._warp_image(frame)

        # 2. Grayscale threshold
        thresh = self._threshold(warped)
        self.dbg_white_px = int(np.count_nonzero(thresh))
        self.dbg_bev_nonzero = self.dbg_white_px

        # 3. Sliding-window polyfit
        left_fit, right_fit, window_img = self._polyfit_sliding_window(thresh)

        angle = 0.0
        center_error = 0.0
        h, w = self._height, self._width
        half_lane = self.EXPECTED_LANE_WIDTH_PX / 2.0

        if left_fit is not None and right_fit is not None:
            # ── Both lanes: VROOM midpoint-error pipeline ───────
            state.left_fit_valid = True
            state.right_fit_valid = True
            state.both_lanes_valid = True

            left_x, left_y, right_x, right_y = self._get_poly_points(
                left_fit, right_fit
            )

            midpoint_img = self._plot_midpoints(
                left_x, left_y, right_x, right_y, h, w
            )
            error, _ = self._get_error(midpoint_img)
            center_error = error

            # VROOM angle: 90 - atan2(nose2wheel, error)
            if abs(error) < 1.0:
                angle = 0.0
            else:
                angle = 90.0 - math.degrees(
                    math.atan2(self._nose2wheel, error)
                )

            lx_bot = np.polyval(left_fit, h - 1)
            rx_bot = np.polyval(right_fit, h - 1)
            self.dbg_lane_width_px = abs(rx_bot - lx_bot)

        elif left_fit is not None and right_fit is None:
            # ── Left lane only → synthesise right lane ──────────
            state.left_fit_valid = True

            synth_right = left_fit.copy()
            synth_right[2] += half_lane

            left_x, left_y, right_x, right_y = self._get_poly_points(
                left_fit, synth_right
            )
            midpoint_img = self._plot_midpoints(
                left_x, left_y, right_x, right_y, h, w
            )
            error, _ = self._get_error(midpoint_img)
            center_error = error

            if abs(error) < 1.0:
                angle = 0.0
            else:
                angle = 90.0 - math.degrees(
                    math.atan2(self._nose2wheel, error)
                )

        elif right_fit is not None and left_fit is None:
            # ── Right lane only → synthesise left lane ──────────
            state.right_fit_valid = True

            synth_left = right_fit.copy()
            synth_left[2] -= half_lane

            left_x, left_y, right_x, right_y = self._get_poly_points(
                synth_left, right_fit
            )
            midpoint_img = self._plot_midpoints(
                left_x, left_y, right_x, right_y, h, w
            )
            error, _ = self._get_error(midpoint_img)
            center_error = error

            if abs(error) < 1.0:
                angle = 0.0
            else:
                angle = 90.0 - math.degrees(
                    math.atan2(self._nose2wheel, error)
                )

        else:
            # ── No lanes ────────────────────────────────────────
            angle = 0.0

        # ── Stabilise (VROOM LaneKeeping.steer) ─────────────────
        num_lanes = (1 if state.left_fit_valid else 0) + \
                    (1 if state.right_fit_valid else 0)

        if num_lanes > 0:
            angle = self._stabilize_steering_angle(
                self._curr_steering_angle, angle, num_lanes
            )
            self._curr_steering_angle = angle
            self._last_good_time = time.monotonic()
            self._no_lane_frames = 0
        else:
            # 15-frame holdover
            self._no_lane_frames += 1
            if self._no_lane_frames <= self._HOLD_FRAMES:
                angle = self._curr_steering_angle
            else:
                angle = 0.0
                self._curr_steering_angle = 0.0
            state.using_fallback = True

        self.dbg_center_error = center_error

        state.cte = angle
        state.heading_error = 0.0

        debug_img = self._draw_debug(
            frame, thresh, window_img, left_fit, right_fit, state
        )
        return state, debug_img

    # ================================================================
    #  Debug visualisation (4-panel)
    # ================================================================

    def _draw_debug(
        self,
        original: np.ndarray,
        thresh: np.ndarray,
        window_img: np.ndarray,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        state: LaneState,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        plot_y = np.linspace(0, self._height - 1, self._height)
        half_lane = self.EXPECTED_LANE_WIDTH_PX / 2.0

        # ── Helper: effective left/right fits (including synth) ─
        eff_left = left_fit
        eff_right = right_fit
        if left_fit is not None and right_fit is None:
            eff_right = left_fit.copy()
            eff_right[2] += half_lane
        elif right_fit is not None and left_fit is None:
            eff_left = right_fit.copy()
            eff_left[2] -= half_lane

        # ────────────────────────────────────────────────────────
        # Panel 1: Original + BEV trapezoid + un-warped lane fill
        # ────────────────────────────────────────────────────────
        overlay = original.copy()
        src_pts = self._src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [src_pts], True, (0, 255, 255), 2)

        if eff_left is not None and eff_right is not None:
            lx = np.polyval(eff_left, plot_y)
            rx = np.polyval(eff_right, plot_y)
            pts_left = np.column_stack((lx, plot_y)).astype(np.int32)
            pts_right = np.column_stack((rx, plot_y)).astype(np.int32)
            pts_all = np.vstack([pts_left, pts_right[::-1]])
            colour_warp = np.zeros((self._height, self._width, 3), np.uint8)
            cv2.fillPoly(colour_warp, [pts_all], (0, 180, 0))
            try:
                unwarped = cv2.warpPerspective(
                    colour_warp, self._inv_warp_matrix, (w, h)
                )
                overlay = cv2.addWeighted(overlay, 1.0, unwarped, 0.4, 0.0)
            except Exception:
                pass

        lines_txt = [
            f"Angle: {state.cte:+.1f}deg  L:{state.left_fit_valid} R:{state.right_fit_valid}",
            f"White:{self.dbg_white_px}  LaneW:{self.dbg_lane_width_px:.0f}px  Err:{self.dbg_center_error:+.0f}",
        ]
        if state.using_fallback:
            lines_txt.append("** FALLBACK **")
        for i, txt in enumerate(lines_txt):
            c = (0, 255, 0) if not state.using_fallback else (0, 0, 255)
            cv2.putText(overlay, txt, (8, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        panel1 = cv2.resize(overlay, (w, h))

        # ────────────────────────────────────────────────────────
        # Panel 2: BEV threshold mask with lane lines drawn on it
        # ────────────────────────────────────────────────────────
        mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        if eff_left is not None:
            pts = np.column_stack((
                np.polyval(eff_left, plot_y).astype(int),
                plot_y.astype(int)
            ))
            is_synth = (left_fit is None)
            colour_l = (255, 0, 255) if is_synth else (255, 0, 0)
            cv2.polylines(mask_bgr, [pts], False, colour_l, 2)
        if eff_right is not None:
            pts = np.column_stack((
                np.polyval(eff_right, plot_y).astype(int),
                plot_y.astype(int)
            ))
            is_synth = (right_fit is None)
            colour_r = (255, 0, 255) if is_synth else (0, 255, 0)
            cv2.polylines(mask_bgr, [pts], False, colour_r, 2)
        if eff_left is not None and eff_right is not None:
            cx = ((np.polyval(eff_left, plot_y) +
                   np.polyval(eff_right, plot_y)) / 2.0).astype(int)
            pts_c = np.column_stack((cx, plot_y.astype(int)))
            cv2.polylines(mask_bgr, [pts_c], False, (0, 255, 255), 2)
        # Draw image centre reference line
        cv2.line(mask_bgr, (w // 2, 0), (w // 2, h - 1), (0, 0, 255), 1)
        cv2.putText(mask_bgr, "BEV Mask  L=blue R=green C=yellow", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        if eff_left is not None and eff_right is not None:
            bot_cx = int((np.polyval(eff_left, h - 1) +
                          np.polyval(eff_right, h - 1)) / 2.0)
            cv2.circle(mask_bgr, (bot_cx, h - 20), 8, (0, 255, 255), -1)
            cv2.circle(mask_bgr, (w // 2, h - 20), 8, (0, 0, 255), -1)
        panel2 = cv2.resize(mask_bgr, (w, h))

        # ────────────────────────────────────────────────────────
        # Panel 3: BEV colour warp (what the camera sees warped)
        # ────────────────────────────────────────────────────────
        panel3 = cv2.resize(self._warp_image(original), (w, h))
        cv2.putText(panel3, "BEV Warp", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ────────────────────────────────────────────────────────
        # Panel 4: Sliding windows + poly curves
        # ────────────────────────────────────────────────────────
        bev_panel = window_img.copy()
        if left_fit is not None:
            pts = np.column_stack((
                np.polyval(left_fit, plot_y), plot_y
            )).astype(np.int32)
            cv2.polylines(bev_panel, [pts], False, (255, 255, 0), 2)
        if right_fit is not None:
            pts = np.column_stack((
                np.polyval(right_fit, plot_y), plot_y
            )).astype(np.int32)
            cv2.polylines(bev_panel, [pts], False, (0, 255, 255), 2)
        panel4 = cv2.resize(bev_panel, (w, h))
        cv2.putText(panel4, "Sliding Window", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        top_row = np.hstack([panel1, panel2])
        bot_row = np.hstack([panel3, panel4])
        return np.vstack([top_row, bot_row])
