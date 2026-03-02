#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Detection — Bird's-Eye View + Sliding Window
===================================================
Classical computer-vision lane detection for the BFMC simulation track
(pure white lane lines on a solid black ground).

Pipeline
--------
  1. Binary threshold  → isolate white lines from black track
  2. Perspective transform → Bird's-Eye View (BEV)
  3. Histogram-based sliding window → find left/right lane pixels
  4. 2nd-order polynomial fit → smooth lane curves
  5. Cross-Track Error (CTE) + heading angle → published to control node

The module maintains a short history of polynomial fits so that
**missing road markings** can be handled via the last-known-good fit
for up to ``missing_lane_timeout_sec``.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from car_brain.config import DrivingConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class LaneState:
    """Published on ``/perception/lane_state``."""
    cte: float = 0.0              # cross-track error (pixels, +right)
    heading_error: float = 0.0    # heading angle error (radians, +CW)
    left_fit_valid: bool = False
    right_fit_valid: bool = False
    both_lanes_valid: bool = False
    using_fallback: bool = False   # True when using last-known-good fit
    timestamp: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lane Detector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LaneDetector:
    """
    Stateful lane detector that converts a raw BGR camera frame into
    a ``LaneState`` (CTE + heading error) using classical CV only.
    """

    def __init__(self, cfg: DrivingConfig) -> None:
        self._cfg = cfg

        # ── Perspective transform matrices ──────────────────────
        src = np.float32(cfg.bev_src_points)
        dst = np.float32(cfg.bev_dst_points)
        self._M = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)

        # ── Polynomial history for temporal smoothing ───────────
        self._left_fits: Deque[np.ndarray] = deque(
            maxlen=cfg.lane_history_frames
        )
        self._right_fits: Deque[np.ndarray] = deque(
            maxlen=cfg.lane_history_frames
        )

        # ── Last-known-good fit for missing-lane fallback ───────
        self._last_good_left: Optional[np.ndarray] = None
        self._last_good_right: Optional[np.ndarray] = None
        self._last_good_time: float = 0.0

        # ── Warmup: skip first N frames to avoid bad early fits ─
        self._warmup_frames: int = 0
        self._warmup_threshold: int = 5   # need 5 valid frames

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def process(
        self, frame: np.ndarray
    ) -> Tuple[LaneState, Optional[np.ndarray]]:
        """
        Run the full lane-detection pipeline on a single BGR frame.

        Returns
        -------
        state : LaneState
            Cross-track error and heading angle.
        debug_image : np.ndarray | None
            Annotated debug image (or ``None`` if visualisation is off).
        """
        state = LaneState(timestamp=time.monotonic())
        bev_w = self._cfg.bev_width
        bev_h = self._cfg.bev_height

        # 1. Binary threshold on the full image
        binary = self._threshold(frame)

        # 2. Bird's-eye view warp
        bev = cv2.warpPerspective(binary, self._M, (bev_w, bev_h))

        # 3. Sliding window search
        left_x, left_y, right_x, right_y, window_img = (
            self._sliding_window(bev)
        )

        # 4. Polynomial fits
        left_fit = self._fit_poly(left_x, left_y)
        right_fit = self._fit_poly(right_x, right_y)

        # ── Sanity-check the polynomial fits ────────────────────
        #    A valid lane should evaluate to an x within the BEV.
        left_fit = self._validate_fit(left_fit, bev_h, bev_w)
        right_fit = self._validate_fit(right_fit, bev_h, bev_w)

        # If both detected, left must be to the left of right
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit, bev_h - 1)
            rx = np.polyval(right_fit, bev_h - 1)
            if lx >= rx:
                left_fit = None
                right_fit = None

        now = time.monotonic()

        # ── Warmup: don't trust first few frames ───────────────
        if self._warmup_frames < self._warmup_threshold:
            if left_fit is not None or right_fit is not None:
                self._warmup_frames += 1
            # During warmup, return zero error
            state.using_fallback = True
            debug_img = (
                self._draw_debug(
                    frame, bev, window_img, None, None, state
                )
                if self._cfg.publish_visualisation
                else None
            )
            return state, debug_img

        # ── Temporal averaging ──────────────────────────────────
        if left_fit is not None:
            self._left_fits.append(left_fit)
            self._last_good_left = self._average_fit(self._left_fits)
            state.left_fit_valid = True
        if right_fit is not None:
            self._right_fits.append(right_fit)
            self._last_good_right = self._average_fit(self._right_fits)
            state.right_fit_valid = True

        if left_fit is not None or right_fit is not None:
            self._last_good_time = now

        state.both_lanes_valid = (
            state.left_fit_valid and state.right_fit_valid
        )

        # ── Fallback to last-known-good if lanes are lost ───────
        avg_left = (
            self._average_fit(self._left_fits)
            if self._left_fits else None
        )
        avg_right = (
            self._average_fit(self._right_fits)
            if self._right_fits else None
        )

        using_fallback = False
        if avg_left is None and avg_right is None:
            elapsed = now - self._last_good_time
            if elapsed < self._cfg.missing_lane_timeout_sec:
                avg_left = self._last_good_left
                avg_right = self._last_good_right
                using_fallback = True
            else:
                # Totally lost — return zero error (dead reckoning)
                state.using_fallback = True
                debug_img = (
                    self._draw_debug(
                        frame, bev, window_img, None, None, state
                    )
                    if self._cfg.publish_visualisation
                    else None
                )
                return state, debug_img

        state.using_fallback = using_fallback

        # 5. Compute CTE and heading from polynomial fits
        cte, heading = self._compute_errors(
            avg_left, avg_right, bev_h, bev_w
        )
        state.cte = cte
        state.heading_error = heading

        # 6. Debug visualisation
        debug_img = (
            self._draw_debug(
                frame, bev, window_img, avg_left, avg_right, state
            )
            if self._cfg.publish_visualisation
            else None
        )

        return state, debug_img

    # ================================================================
    #  PIPELINE STAGES
    # ================================================================

    def _threshold(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale → Gaussian blur → binary threshold.
        
        Only processes the lower portion of the frame (road area)
        to avoid false positives from sky/background.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Mask out the top portion (sky/background) — road starts ~55% down
        roi_top = int(h * 0.50)
        mask = np.zeros_like(gray)
        mask[roi_top:, :] = gray[roi_top:, :]

        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, self._cfg.white_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Remove small noise blobs
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _sliding_window(
        self, bev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               Optional[np.ndarray]]:
        """
        Histogram-based sliding window algorithm.

        Returns
        -------
        left_x, left_y   : pixel coordinates belonging to the left lane
        right_x, right_y : pixel coordinates belonging to the right lane
        window_img        : colourised BEV with window rectangles drawn
        """
        h, w = bev.shape[:2]
        n_windows = self._cfg.sliding_window_count
        margin = self._cfg.sliding_window_margin
        min_pix = self._cfg.sliding_window_min_pix
        win_h = h // n_windows

        # Histogram of bottom half to find starting x positions
        histogram = np.sum(bev[h // 2:, :], axis=0).astype(float)
        midpoint = w // 2
        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:])) + midpoint

        left_current = left_base
        right_current = right_base

        # Non-zero pixel coordinates
        nonzero = bev.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_lane_inds: List[np.ndarray] = []
        right_lane_inds: List[np.ndarray] = []

        # Debug image
        if self._cfg.publish_visualisation:
            out_img = np.dstack((bev, bev, bev))
        else:
            out_img = None

        for win_idx in range(n_windows):
            y_low = h - (win_idx + 1) * win_h
            y_high = h - win_idx * win_h

            xl_low = max(0, left_current - margin)
            xl_high = min(w, left_current + margin)
            xr_low = max(0, right_current - margin)
            xr_high = min(w, right_current + margin)

            if out_img is not None:
                cv2.rectangle(
                    out_img, (xl_low, y_low), (xl_high, y_high),
                    (0, 255, 0), 2
                )
                cv2.rectangle(
                    out_img, (xr_low, y_low), (xr_high, y_high),
                    (0, 255, 0), 2
                )

            good_left = (
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= xl_low) & (nonzero_x < xl_high)
            ).nonzero()[0]

            good_right = (
                (nonzero_y >= y_low) & (nonzero_y < y_high)
                & (nonzero_x >= xr_low) & (nonzero_x < xr_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > min_pix:
                left_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > min_pix:
                right_current = int(np.mean(nonzero_x[good_right]))

        # Concatenate all found indices
        left_cat = (
            np.concatenate(left_lane_inds)
            if left_lane_inds else np.array([], dtype=int)
        )
        right_cat = (
            np.concatenate(right_lane_inds)
            if right_lane_inds else np.array([], dtype=int)
        )

        lx = nonzero_x[left_cat] if len(left_cat) > 0 else np.array([])
        ly = nonzero_y[left_cat] if len(left_cat) > 0 else np.array([])
        rx = nonzero_x[right_cat] if len(right_cat) > 0 else np.array([])
        ry = nonzero_y[right_cat] if len(right_cat) > 0 else np.array([])

        # Colour detected pixels
        if out_img is not None:
            if len(left_cat) > 0:
                out_img[ly, lx] = [255, 0, 0]    # blue = left
            if len(right_cat) > 0:
                out_img[ry, rx] = [0, 0, 255]     # red = right

        return lx, ly, rx, ry, out_img

    @staticmethod
    def _fit_poly(
        xs: np.ndarray, ys: np.ndarray, min_points: int = 20
    ) -> Optional[np.ndarray]:
        """Fit a 2nd-order polynomial ``x = f(y)``. Returns ``None``
        if too few points."""
        if len(xs) < min_points:
            return None
        try:
            return np.polyfit(ys, xs, 2)
        except (np.RankWarning, np.linalg.LinAlgError):
            return None

    @staticmethod
    def _average_fit(fits: Deque[np.ndarray]) -> Optional[np.ndarray]:
        """Average the recent polynomial coefficients for smoothing."""
        if not fits:
            return None
        return np.mean(np.array(list(fits)), axis=0)

    def _validate_fit(
        self,
        fit: Optional[np.ndarray],
        bev_h: int,
        bev_w: int,
    ) -> Optional[np.ndarray]:
        """Reject a polynomial fit whose lane position falls outside
        the BEV image at key rows."""
        if fit is None:
            return None
        for y in [bev_h - 1, bev_h * 0.5, bev_h * 0.3]:
            x = np.polyval(fit, y)
            if x < -bev_w * 0.5 or x > bev_w * 1.5:
                return None
        return fit

    # ================================================================
    #  ERROR COMPUTATION
    # ================================================================

    def _compute_errors(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        bev_h: int,
        bev_w: int,
    ) -> Tuple[float, float]:
        """
        Compute cross-track error (CTE) and heading angle error from
        polynomial lane fits in BEV space.

        CTE      : offset of lane centre from image centre at the
                   bottom of the BEV (positive → car left of centre)
        Heading  : tangent angle difference at a look-ahead point
        """
        img_cx = bev_w / 2.0
        y_bottom = float(bev_h - 1)
        y_lookahead = float(bev_h * 0.5)  # 50% from top — closer to car, more reliable

        single_offset = bev_w * self._cfg.single_lane_offset_ratio

        if left_fit is not None and right_fit is not None:
            left_x_bot = np.polyval(left_fit, y_bottom)
            right_x_bot = np.polyval(right_fit, y_bottom)
            centre_bot = (left_x_bot + right_x_bot) / 2.0

            left_x_la = np.polyval(left_fit, y_lookahead)
            right_x_la = np.polyval(right_fit, y_lookahead)
            centre_la = (left_x_la + right_x_la) / 2.0

        elif left_fit is not None:
            left_x_bot = np.polyval(left_fit, y_bottom)
            centre_bot = left_x_bot + single_offset
            left_x_la = np.polyval(left_fit, y_lookahead)
            centre_la = left_x_la + single_offset

        elif right_fit is not None:
            right_x_bot = np.polyval(right_fit, y_bottom)
            centre_bot = right_x_bot - single_offset
            right_x_la = np.polyval(right_fit, y_lookahead)
            centre_la = right_x_la - single_offset

        else:
            return 0.0, 0.0

        # CTE: positive means car is left of lane centre
        # Normalise to [-1, +1] by dividing by half the BEV width
        cte_px = centre_bot - img_cx
        cte = cte_px / (bev_w / 2.0)
        cte = max(-1.0, min(1.0, cte))  # clamp

        # Heading error: angle between vertical and the bottom→lookahead
        # vector on the lane centre curve
        dx = centre_la - centre_bot
        dy = y_lookahead - y_bottom  # negative
        heading = float(np.arctan2(dx, -dy)) if abs(dy) > 1e-6 else 0.0

        return float(cte), heading

    # ================================================================
    #  DEBUG VISUALISATION
    # ================================================================

    def _draw_debug(
        self,
        original: np.ndarray,
        bev: np.ndarray,
        window_img: Optional[np.ndarray],
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        state: LaneState,
    ) -> np.ndarray:
        """
        Build a composite debug image:
          Left half  : original image with un-warped lane overlay
          Right half : BEV with sliding windows and polynomial curves
        """
        bev_h = self._cfg.bev_height
        bev_w = self._cfg.bev_width
        h, w = original.shape[:2]

        # ── BEV panel with fitted curves ────────────────────────
        if window_img is not None:
            bev_panel = window_img.copy()
        else:
            bev_panel = np.dstack((bev, bev, bev))

        plot_y = np.linspace(0, bev_h - 1, bev_h)

        if left_fit is not None:
            left_x_pts = np.polyval(left_fit, plot_y)
            pts = np.array(
                [np.column_stack((left_x_pts, plot_y))], dtype=np.int32
            )
            cv2.polylines(bev_panel, pts, False, (255, 255, 0), 2)

        if right_fit is not None:
            right_x_pts = np.polyval(right_fit, plot_y)
            pts = np.array(
                [np.column_stack((right_x_pts, plot_y))], dtype=np.int32
            )
            cv2.polylines(bev_panel, pts, False, (0, 255, 255), 2)

        # ── Lane overlay on original ────────────────────────────
        overlay = original.copy()
        if left_fit is not None and right_fit is not None:
            left_x_pts = np.polyval(left_fit, plot_y)
            right_x_pts = np.polyval(right_fit, plot_y)

            pts_left = np.column_stack(
                (left_x_pts, plot_y)
            ).astype(np.int32)
            pts_right = np.column_stack(
                (right_x_pts, plot_y)
            ).astype(np.int32)
            pts_all = np.vstack([pts_left, pts_right[::-1]])

            colour_warp = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
            cv2.fillPoly(colour_warp, [pts_all], (0, 180, 0))

            unwarped = cv2.warpPerspective(
                colour_warp, self._M_inv, (w, h)
            )
            overlay = cv2.addWeighted(overlay, 1.0, unwarped, 0.4, 0.0)

        # ── Draw the BEV source trapezoid on original ───────────
        src_pts = np.array(self._cfg.bev_src_points, dtype=np.int32)
        cv2.polylines(overlay, [src_pts], True, (0, 255, 255), 1)

        # ── Text annotations ────────────────────────────────────
        info_lines = [
            f"CTE: {state.cte:+.1f}px",
            f"Heading: {np.degrees(state.heading_error):+.1f}deg",
            f"Left:{state.left_fit_valid}  Right:{state.right_fit_valid}",
        ]
        if state.using_fallback:
            info_lines.append("** FALLBACK (missing lanes) **")

        for i, line in enumerate(info_lines):
            colour = (
                (0, 255, 255) if not state.using_fallback
                else (0, 0, 255)
            )
            cv2.putText(
                overlay, line, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )

        # ── Compose side-by-side ────────────────────────────────
        bev_resized = cv2.resize(bev_panel, (w, h))
        composite = np.hstack([overlay, bev_resized])
        return composite
