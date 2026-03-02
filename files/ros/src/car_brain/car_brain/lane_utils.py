from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FitResult:
    fit: Optional[np.ndarray]
    confidence: float
    inlier_ratio: float
    inlier_count: int
    residual_mean: float
    degree: int


def threshold_lane_binary(
    frame_bgr: np.ndarray,
    gray_threshold: int,
    sat_threshold: int,
    canny_low: int,
    canny_high: int,
    morph_kernel: int,
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    white_mask = (gray >= gray_threshold) & (lightness >= gray_threshold)
    low_sat_mask = saturation <= sat_threshold
    color_binary = (white_mask & low_sat_mask).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.bitwise_or(color_binary, edges)

    kernel_size = max(1, morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined


def build_dynamic_roi_mask(
    shape: Tuple[int, int],
    top_y_ratio: float,
    top_width_ratio: float,
    center_shift_px: float,
) -> np.ndarray:
    height, width = shape
    top_y = int(np.clip(top_y_ratio, 0.1, 0.95) * height)
    top_half_width = int(np.clip(top_width_ratio, 0.1, 0.9) * width * 0.5)
    cx = int(width // 2 + center_shift_px)

    top_left_x = int(np.clip(cx - top_half_width, 0, width - 1))
    top_right_x = int(np.clip(cx + top_half_width, 0, width - 1))
    poly = np.array(
        [[0, height - 1], [width - 1, height - 1], [top_right_x, top_y], [top_left_x, top_y]],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def parse_points(flat_points: Sequence[float]) -> np.ndarray:
    arr = np.asarray(flat_points, dtype=np.float32)
    if arr.size != 8:
        raise ValueError("Perspective points must have exactly 8 values")
    return arr.reshape(4, 2)


def find_histogram_bases(binary_bev: np.ndarray, midpoint_bias_px: int = 0) -> Tuple[int, int]:
    height, width = binary_bev.shape
    histogram = np.sum(binary_bev[height // 2 :, :], axis=0).astype(np.float32)
    midpoint = int(width // 2 + midpoint_bias_px)
    midpoint = int(np.clip(midpoint, width * 0.25, width * 0.75))

    left_hist = histogram[:midpoint]
    right_hist = histogram[midpoint:]

    left_base = int(np.argmax(left_hist)) if left_hist.size else width // 4
    right_base = (int(np.argmax(right_hist)) + midpoint) if right_hist.size else (3 * width // 4)
    return left_base, right_base


def adaptive_sliding_window(
    binary_bev: np.ndarray,
    left_base: int,
    right_base: int,
    window_count: int,
    margin_base: int,
    margin_curve_gain: float,
    min_pixels_recenter: int,
    curve_hint_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    height, width = binary_bev.shape
    nwindows = max(4, int(window_count))
    win_h = max(1, height // nwindows)
    margin = int(max(30, margin_base + margin_curve_gain * abs(curve_hint_px)))

    nonzero = binary_bev.nonzero()
    nonzero_y = np.asarray(nonzero[0])
    nonzero_x = np.asarray(nonzero[1])

    left_current = int(left_base)
    right_current = int(right_base)

    left_inds: List[np.ndarray] = []
    right_inds: List[np.ndarray] = []
    windows: List[Tuple[int, int, int, int]] = []

    for win_idx in range(nwindows):
        y_low = height - (win_idx + 1) * win_h
        y_high = height - win_idx * win_h

        xl_low = max(0, left_current - margin)
        xl_high = min(width, left_current + margin)
        xr_low = max(0, right_current - margin)
        xr_high = min(width, right_current + margin)
        windows.append((xl_low, y_low, xl_high, y_high))
        windows.append((xr_low, y_low, xr_high, y_high))

        left_good = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= xl_low)
            & (nonzero_x < xl_high)
        ).nonzero()[0]
        right_good = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= xr_low)
            & (nonzero_x < xr_high)
        ).nonzero()[0]

        left_inds.append(left_good)
        right_inds.append(right_good)

        if left_good.size >= min_pixels_recenter:
            left_current = int(np.mean(nonzero_x[left_good]))
        if right_good.size >= min_pixels_recenter:
            right_current = int(np.mean(nonzero_x[right_good]))

    left_cat = np.concatenate(left_inds) if left_inds else np.array([], dtype=np.int64)
    right_cat = np.concatenate(right_inds) if right_inds else np.array([], dtype=np.int64)

    lx = nonzero_x[left_cat] if left_cat.size else np.array([], dtype=np.float32)
    ly = nonzero_y[left_cat] if left_cat.size else np.array([], dtype=np.float32)
    rx = nonzero_x[right_cat] if right_cat.size else np.array([], dtype=np.float32)
    ry = nonzero_y[right_cat] if right_cat.size else np.array([], dtype=np.float32)
    return lx, ly, rx, ry, windows


def _safe_polyfit(y_vals: np.ndarray, x_vals: np.ndarray, degree: int) -> Optional[np.ndarray]:
    if y_vals.size < degree + 1:
        return None
    try:
        return np.polyfit(y_vals, x_vals, degree)
    except Exception:
        return None


def polyfit_ransac(
    y_vals: np.ndarray,
    x_vals: np.ndarray,
    degree: int,
    iterations: int,
    inlier_threshold_px: float,
    min_inliers: int,
    random_state: int,
) -> FitResult:
    n_points = int(y_vals.size)
    if n_points < max(min_inliers, degree + 2):
        return FitResult(None, 0.0, 0.0, 0, 1e6, degree)

    rng = np.random.default_rng(random_state)
    sample_size = degree + 1
    best_fit = None
    best_mask = None
    best_inliers = -1
    best_residual = 1e9

    for _ in range(max(10, iterations)):
        sample_idx = rng.choice(n_points, size=sample_size, replace=False)
        candidate = _safe_polyfit(y_vals[sample_idx], x_vals[sample_idx], degree)
        if candidate is None:
            continue

        residuals = np.abs(x_vals - np.polyval(candidate, y_vals))
        mask = residuals < inlier_threshold_px
        inliers = int(np.sum(mask))
        if inliers < min_inliers:
            continue

        mean_residual = float(np.mean(residuals[mask])) if inliers > 0 else 1e9
        if inliers > best_inliers or (inliers == best_inliers and mean_residual < best_residual):
            best_inliers = inliers
            best_residual = mean_residual
            best_fit = candidate
            best_mask = mask

    if best_fit is None or best_mask is None:
        return FitResult(None, 0.0, 0.0, 0, 1e6, degree)

    refined = _safe_polyfit(y_vals[best_mask], x_vals[best_mask], degree)
    if refined is None:
        return FitResult(None, 0.0, 0.0, 0, 1e6, degree)

    refined_residuals = np.abs(x_vals[best_mask] - np.polyval(refined, y_vals[best_mask]))
    inlier_ratio = best_inliers / float(n_points)
    residual_mean = float(np.mean(refined_residuals)) if refined_residuals.size else 1e6
    confidence = float(np.clip(0.7 * inlier_ratio + 0.3 * np.exp(-residual_mean / 8.0), 0.0, 1.0))

    return FitResult(
        fit=refined,
        confidence=confidence,
        inlier_ratio=inlier_ratio,
        inlier_count=best_inliers,
        residual_mean=residual_mean,
        degree=degree,
    )


def choose_best_fit(results: Sequence[FitResult]) -> FitResult:
    valid = [result for result in results if result.fit is not None]
    if not valid:
        return FitResult(None, 0.0, 0.0, 0, 1e6, 2)

    best = max(
        valid,
        key=lambda result: (
            result.confidence,
            result.inlier_count,
            -result.residual_mean,
        ),
    )
    return best


def derivative_at_y(fit: np.ndarray, y: float) -> float:
    deriv = np.polyder(fit)
    return float(np.polyval(deriv, y))


def curvature_radius_m(
    fit_px: np.ndarray,
    y_eval_px: float,
    ym_per_px: float,
    xm_per_px: float,
) -> float:
    degree = fit_px.size - 1
    if degree < 2:
        return 1e6

    ys = np.array([0.0, y_eval_px * 0.5, y_eval_px], dtype=np.float64)
    xs = np.polyval(fit_px, ys)
    fit_m = _safe_polyfit(ys * ym_per_px, xs * xm_per_px, min(2, degree))
    if fit_m is None:
        return 1e6

    y_m = y_eval_px * ym_per_px
    a = float(fit_m[0])
    b = float(fit_m[1])
    denom = abs(2.0 * a)
    if denom < 1e-9:
        return 1e6
    radius = ((1.0 + (2.0 * a * y_m + b) ** 2) ** 1.5) / denom
    return float(np.clip(radius, 1.0, 1e6))


def sanity_check_lane_pair(
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    image_height: int,
    image_width: int,
    lane_width_min_px: float,
    lane_width_max_px: float,
    lane_width_var_max_px: float,
    slope_diff_max: float,
    curvature_min_m: float,
    curvature_max_m: float,
    ym_per_px: float,
    xm_per_px: float,
) -> Tuple[bool, Dict[str, float], str]:
    y_bottom = float(image_height - 1)
    y_mid = float(image_height * 0.6)
    y_top = float(image_height * 0.3)

    left_bottom = float(np.polyval(left_fit, y_bottom))
    right_bottom = float(np.polyval(right_fit, y_bottom))
    left_mid = float(np.polyval(left_fit, y_mid))
    right_mid = float(np.polyval(right_fit, y_mid))
    left_top = float(np.polyval(left_fit, y_top))
    right_top = float(np.polyval(right_fit, y_top))

    width_bottom = right_bottom - left_bottom
    width_mid = right_mid - left_mid
    width_top = right_top - left_top

    if left_bottom >= right_bottom or left_mid >= right_mid:
        return False, {}, "left_right_swap"

    if min(width_bottom, width_mid, width_top) < lane_width_min_px:
        return False, {}, "lane_width_too_small"
    if max(width_bottom, width_mid, width_top) > lane_width_max_px:
        return False, {}, "lane_width_too_large"
    if (max(width_bottom, width_mid, width_top) - min(width_bottom, width_mid, width_top)) > lane_width_var_max_px:
        return False, {}, "lane_width_inconsistent"

    slope_left = derivative_at_y(left_fit, y_mid)
    slope_right = derivative_at_y(right_fit, y_mid)
    slope_diff = abs(slope_left - slope_right)
    if slope_diff > slope_diff_max:
        return False, {}, "lane_non_parallel"

    left_radius = curvature_radius_m(left_fit, y_bottom, ym_per_px, xm_per_px)
    right_radius = curvature_radius_m(right_fit, y_bottom, ym_per_px, xm_per_px)
    mean_radius = 0.5 * (left_radius + right_radius)
    if mean_radius < curvature_min_m or mean_radius > curvature_max_m:
        return False, {}, "curvature_out_of_range"

    inside_frame = (
        -0.2 * image_width <= left_bottom <= 1.2 * image_width
        and -0.2 * image_width <= right_bottom <= 1.2 * image_width
    )
    if not inside_frame:
        return False, {}, "fit_outside_image"

    metrics = {
        "lane_width_bottom_px": float(width_bottom),
        "lane_width_mid_px": float(width_mid),
        "lane_width_top_px": float(width_top),
        "slope_diff": float(slope_diff),
        "left_radius_m": float(left_radius),
        "right_radius_m": float(right_radius),
        "radius_m": float(mean_radius),
    }
    return True, metrics, "ok"


def smooth_fit_from_history(history: Sequence[np.ndarray], ema_alpha: float) -> Optional[np.ndarray]:
    if not history:
        return None
    alpha = float(np.clip(ema_alpha, 0.01, 1.0))
    smoothed = np.array(history[0], dtype=np.float64)
    for coeffs in history[1:]:
        smoothed = alpha * np.asarray(coeffs, dtype=np.float64) + (1.0 - alpha) * smoothed
    return smoothed.astype(np.float64)


def lane_direction_hint_px(center_fit: Optional[np.ndarray], y_bottom: int, y_top: int) -> float:
    if center_fit is None:
        return 0.0
    xb = float(np.polyval(center_fit, float(y_bottom)))
    xt = float(np.polyval(center_fit, float(y_top)))
    return xt - xb


def build_center_fit(left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray], y_max: int) -> Optional[np.ndarray]:
    if left_fit is None and right_fit is None:
        return None
    ys = np.linspace(0.0, float(y_max), 40)
    if left_fit is not None and right_fit is not None:
        xs = 0.5 * (np.polyval(left_fit, ys) + np.polyval(right_fit, ys))
    elif left_fit is not None:
        xs = np.polyval(left_fit, ys)
    else:
        xs = np.polyval(right_fit, ys)
    return _safe_polyfit(ys, xs, 2)
