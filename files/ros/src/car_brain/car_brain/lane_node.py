from __future__ import annotations

import json
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, String

from car_brain.lane_utils import (
    adaptive_sliding_window,
    build_center_fit,
    build_dynamic_roi_mask,
    choose_best_fit,
    curvature_radius_m,
    find_histogram_bases,
    lane_direction_hint_px,
    parse_points,
    polyfit_ransac,
    sanity_check_lane_pair,
    smooth_fit_from_history,
    threshold_lane_binary,
)


class LaneDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_node")
        self._declare_params()
        self._load_params()

        self.left_history: Deque[np.ndarray] = deque(maxlen=self.temporal_window)
        self.right_history: Deque[np.ndarray] = deque(maxlen=self.temporal_window)
        self.last_stable_left: Optional[np.ndarray] = None
        self.last_stable_right: Optional[np.ndarray] = None
        self.last_stable_stamp: float = 0.0
        self.last_center_fit: Optional[np.ndarray] = None
        self.last_lane_width_px: Optional[float] = None

        self.frame_counter = 0
        self.last_log_time = time.monotonic()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if self.input_is_compressed:
            self.subscriber = self.create_subscription(
                CompressedImage,
                self.input_topic,
                self._on_compressed,
                qos,
            )
        else:
            self.subscriber = self.create_subscription(
                Image,
                self.input_topic,
                self._on_image,
                qos,
            )

        self.pub_annotated = self.create_publisher(Image, self.output_annotated_topic, 5)
        self.pub_lane_info = self.create_publisher(String, self.output_lane_info_topic, 10)
        self.pub_curvature = self.create_publisher(Float32, self.output_curvature_topic, 10)

        self.get_logger().info(
            f"lane_node started | input={self.input_topic}"
            f" | compressed={self.input_is_compressed}"
            f" | outputs=({self.output_annotated_topic}, {self.output_lane_info_topic}, {self.output_curvature_topic})"
        )

    def _declare_params(self) -> None:
        self.declare_parameter("input_topic", "/automobile/camera/image_raw")
        self.declare_parameter("input_is_compressed", False)
        self.declare_parameter("output_annotated_topic", "/lane_detection/image_annotated")
        self.declare_parameter("output_lane_info_topic", "/lane_detection/lane_info")
        self.declare_parameter("output_curvature_topic", "/lane_detection/curvature")

        self.declare_parameter("target_fps", 30.0)
        self.declare_parameter("publish_annotated", True)

        self.declare_parameter("gray_threshold", 160)
        self.declare_parameter("sat_threshold", 95)
        self.declare_parameter("canny_low", 45)
        self.declare_parameter("canny_high", 140)
        self.declare_parameter("morph_kernel", 3)

        self.declare_parameter("roi_top_y_ratio", 0.50)
        self.declare_parameter("roi_top_width_ratio", 0.50)
        self.declare_parameter("roi_dynamic_shift_px", 80.0)

        self.declare_parameter("bev_width", 640)
        self.declare_parameter("bev_height", 480)
        self.declare_parameter("bev_src_points", [220.0, 280.0, 420.0, 280.0, 640.0, 460.0, 0.0, 460.0])
        self.declare_parameter("bev_dst_points", [100.0, 0.0, 540.0, 0.0, 540.0, 480.0, 100.0, 480.0])

        self.declare_parameter("window_count", 12)
        self.declare_parameter("window_margin_base", 85)
        self.declare_parameter("window_margin_curve_gain", 0.25)
        self.declare_parameter("window_min_pixels", 25)

        self.declare_parameter("ransac_iterations", 50)
        self.declare_parameter("ransac_inlier_threshold_px", 14.0)
        self.declare_parameter("ransac_min_inliers", 50)
        self.declare_parameter("poly_degree_candidates", [2, 3])

        self.declare_parameter("temporal_window", 6)
        self.declare_parameter("temporal_ema_alpha", 0.35)
        self.declare_parameter("fallback_timeout_sec", 1.2)
        self.declare_parameter("min_pair_confidence", 0.40)

        self.declare_parameter("lane_width_min_px", 180.0)
        self.declare_parameter("lane_width_max_px", 460.0)
        self.declare_parameter("lane_width_var_max_px", 95.0)
        self.declare_parameter("slope_diff_max", 1.2)
        self.declare_parameter("curvature_min_m", 3.0)
        self.declare_parameter("curvature_max_m", 5000.0)

        self.declare_parameter("ym_per_px", 0.02)
        self.declare_parameter("xm_per_px", 0.005)
        self.declare_parameter("single_lane_center_offset_px", 190.0)

    def _load_params(self) -> None:
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.input_is_compressed = bool(self.get_parameter("input_is_compressed").value)
        self.output_annotated_topic = str(self.get_parameter("output_annotated_topic").value)
        self.output_lane_info_topic = str(self.get_parameter("output_lane_info_topic").value)
        self.output_curvature_topic = str(self.get_parameter("output_curvature_topic").value)
        self.target_fps = float(self.get_parameter("target_fps").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self.gray_threshold = int(self.get_parameter("gray_threshold").value)
        self.sat_threshold = int(self.get_parameter("sat_threshold").value)
        self.canny_low = int(self.get_parameter("canny_low").value)
        self.canny_high = int(self.get_parameter("canny_high").value)
        self.morph_kernel = int(self.get_parameter("morph_kernel").value)

        self.roi_top_y_ratio = float(self.get_parameter("roi_top_y_ratio").value)
        self.roi_top_width_ratio = float(self.get_parameter("roi_top_width_ratio").value)
        self.roi_dynamic_shift_px = float(self.get_parameter("roi_dynamic_shift_px").value)

        self.bev_width = int(self.get_parameter("bev_width").value)
        self.bev_height = int(self.get_parameter("bev_height").value)
        self.bev_src_points = parse_points(self.get_parameter("bev_src_points").value)
        self.bev_dst_points = parse_points(self.get_parameter("bev_dst_points").value)
        self.M = cv2.getPerspectiveTransform(self.bev_src_points, self.bev_dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.bev_dst_points, self.bev_src_points)

        self.window_count = int(self.get_parameter("window_count").value)
        self.window_margin_base = int(self.get_parameter("window_margin_base").value)
        self.window_margin_curve_gain = float(self.get_parameter("window_margin_curve_gain").value)
        self.window_min_pixels = int(self.get_parameter("window_min_pixels").value)

        self.ransac_iterations = int(self.get_parameter("ransac_iterations").value)
        self.ransac_inlier_threshold_px = float(self.get_parameter("ransac_inlier_threshold_px").value)
        self.ransac_min_inliers = int(self.get_parameter("ransac_min_inliers").value)
        self.poly_degree_candidates = [int(value) for value in self.get_parameter("poly_degree_candidates").value]

        self.temporal_window = int(self.get_parameter("temporal_window").value)
        self.temporal_ema_alpha = float(self.get_parameter("temporal_ema_alpha").value)
        self.fallback_timeout_sec = float(self.get_parameter("fallback_timeout_sec").value)
        self.min_pair_confidence = float(self.get_parameter("min_pair_confidence").value)

        self.lane_width_min_px = float(self.get_parameter("lane_width_min_px").value)
        self.lane_width_max_px = float(self.get_parameter("lane_width_max_px").value)
        self.lane_width_var_max_px = float(self.get_parameter("lane_width_var_max_px").value)
        self.slope_diff_max = float(self.get_parameter("slope_diff_max").value)
        self.curvature_min_m = float(self.get_parameter("curvature_min_m").value)
        self.curvature_max_m = float(self.get_parameter("curvature_max_m").value)

        self.ym_per_px = float(self.get_parameter("ym_per_px").value)
        self.xm_per_px = float(self.get_parameter("xm_per_px").value)
        self.single_lane_center_offset_px = float(self.get_parameter("single_lane_center_offset_px").value)

    def _on_image(self, msg: Image) -> None:
        frame = self._image_msg_to_cv2(msg)
        if frame is None:
            return
        self._process(frame, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _on_compressed(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Compressed image decode failed")
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._process(frame, stamp)

    def _process(self, frame_bgr: np.ndarray, stamp_sec: float) -> None:
        start = time.monotonic()
        self.frame_counter += 1

        binary = threshold_lane_binary(
            frame_bgr,
            gray_threshold=self.gray_threshold,
            sat_threshold=self.sat_threshold,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            morph_kernel=self.morph_kernel,
        )

        curve_hint = lane_direction_hint_px(self.last_center_fit, self.bev_height - 1, int(self.bev_height * 0.35))
        roi_shift = np.clip(np.sign(curve_hint) * self.roi_dynamic_shift_px, -self.roi_dynamic_shift_px, self.roi_dynamic_shift_px)
        roi_mask = build_dynamic_roi_mask(
            shape=(binary.shape[0], binary.shape[1]),
            top_y_ratio=self.roi_top_y_ratio,
            top_width_ratio=self.roi_top_width_ratio,
            center_shift_px=float(roi_shift),
        )
        roi_binary = cv2.bitwise_and(binary, roi_mask)

        bev = cv2.warpPerspective(roi_binary, self.M, (self.bev_width, self.bev_height), flags=cv2.INTER_LINEAR)

        midpoint_bias = int(np.clip(curve_hint * 0.2, -80, 80))
        left_base, right_base = find_histogram_bases(bev, midpoint_bias_px=midpoint_bias)

        lx, ly, rx, ry, windows = adaptive_sliding_window(
            binary_bev=bev,
            left_base=left_base,
            right_base=right_base,
            window_count=self.window_count,
            margin_base=self.window_margin_base,
            margin_curve_gain=self.window_margin_curve_gain,
            min_pixels_recenter=self.window_min_pixels,
            curve_hint_px=curve_hint,
        )

        left_fit, left_conf = self._fit_lane_side(ly, lx, side_seed=11)
        right_fit, right_conf = self._fit_lane_side(ry, rx, side_seed=23)

        pair_confidence = 0.5 * (left_conf + right_conf)
        sanity_ok = False
        sanity_reason = "single_lane"
        metrics: Dict[str, float] = {}

        if left_fit is not None and right_fit is not None:
            sanity_ok, metrics, sanity_reason = sanity_check_lane_pair(
                left_fit=left_fit,
                right_fit=right_fit,
                image_height=self.bev_height,
                image_width=self.bev_width,
                lane_width_min_px=self.lane_width_min_px,
                lane_width_max_px=self.lane_width_max_px,
                lane_width_var_max_px=self.lane_width_var_max_px,
                slope_diff_max=self.slope_diff_max,
                curvature_min_m=self.curvature_min_m,
                curvature_max_m=self.curvature_max_m,
                ym_per_px=self.ym_per_px,
                xm_per_px=self.xm_per_px,
            )

        using_fallback = False
        fallback_reason = "none"

        if left_fit is not None and right_fit is not None and sanity_ok and pair_confidence >= self.min_pair_confidence:
            self.last_stable_left = left_fit.copy()
            self.last_stable_right = right_fit.copy()
            self.last_stable_stamp = time.monotonic()
            self.last_lane_width_px = metrics.get("lane_width_bottom_px")
        else:
            lost = (left_fit is None and right_fit is None) or not sanity_ok or pair_confidence < self.min_pair_confidence
            if lost and (time.monotonic() - self.last_stable_stamp) <= self.fallback_timeout_sec:
                using_fallback = True
                fallback_reason = sanity_reason if not sanity_ok else "low_confidence_or_missing"
                if left_fit is None:
                    left_fit = self.last_stable_left.copy() if self.last_stable_left is not None else None
                if right_fit is None:
                    right_fit = self.last_stable_right.copy() if self.last_stable_right is not None else None
                if left_fit is None and self.last_stable_left is not None:
                    left_fit = self.last_stable_left.copy()
                if right_fit is None and self.last_stable_right is not None:
                    right_fit = self.last_stable_right.copy()

        if left_fit is not None:
            self.left_history.append(left_fit)
        if right_fit is not None:
            self.right_history.append(right_fit)

        left_smoothed = smooth_fit_from_history(list(self.left_history), self.temporal_ema_alpha)
        right_smoothed = smooth_fit_from_history(list(self.right_history), self.temporal_ema_alpha)

        if left_smoothed is None and left_fit is not None:
            left_smoothed = left_fit
        if right_smoothed is None and right_fit is not None:
            right_smoothed = right_fit

        lane_center_px, lane_width_px = self._compute_lane_center_and_width(left_smoothed, right_smoothed)
        if lane_width_px is not None:
            self.last_lane_width_px = lane_width_px

        center_fit = build_center_fit(left_smoothed, right_smoothed, y_max=self.bev_height - 1)
        self.last_center_fit = center_fit if center_fit is not None else self.last_center_fit
        curve_direction_px = lane_direction_hint_px(self.last_center_fit, self.bev_height - 1, int(self.bev_height * 0.35))

        curvature_m = float(metrics.get("radius_m", 0.0)) if metrics else 0.0
        if curvature_m <= 0.0:
            curvature_m = self._estimate_curvature_from_available(left_smoothed, right_smoothed)

        confidence = float(np.clip(pair_confidence if sanity_ok else max(left_conf, right_conf) * 0.7, 0.0, 1.0))

        lane_info = {
            "stamp": stamp_sec,
            "confidence": round(confidence, 4),
            "using_fallback": using_fallback,
            "fallback_reason": fallback_reason,
            "sanity_ok": sanity_ok,
            "sanity_reason": sanity_reason,
            "left_detected": left_fit is not None,
            "right_detected": right_fit is not None,
            "lane_center_px": None if lane_center_px is None else round(float(lane_center_px), 2),
            "lane_width_px": None if lane_width_px is None else round(float(lane_width_px), 2),
            "curvature_m": round(float(curvature_m), 3),
            "curve_direction": "right" if curve_direction_px > 6.0 else "left" if curve_direction_px < -6.0 else "straight",
            "curve_direction_px": round(float(curve_direction_px), 2),
            "fps": round(1.0 / max(1e-4, time.monotonic() - start), 2),
        }
        lane_info.update({k: round(float(v), 3) for k, v in metrics.items()})

        lane_msg = String()
        lane_msg.data = json.dumps(lane_info)
        self.pub_lane_info.publish(lane_msg)

        curv_msg = Float32()
        curv_msg.data = float(curvature_m)
        self.pub_curvature.publish(curv_msg)

        if self.publish_annotated:
            annotated = self._build_annotated_image(
                frame_bgr=frame_bgr,
                binary=bev,
                windows=windows,
                left_fit=left_smoothed,
                right_fit=right_smoothed,
                lane_info=lane_info,
            )
            self.pub_annotated.publish(self._cv2_to_image_msg(annotated))

        now = time.monotonic()
        if now - self.last_log_time > 2.0:
            self.last_log_time = now
            self.get_logger().info(
                f"lane fps={lane_info['fps']} conf={lane_info['confidence']:.2f} "
                f"curv={lane_info['curvature_m']:.1f}m sanity={lane_info['sanity_ok']}"
            )

    def _fit_lane_side(self, y_vals: np.ndarray, x_vals: np.ndarray, side_seed: int) -> Tuple[Optional[np.ndarray], float]:
        results = []
        for degree in self.poly_degree_candidates:
            result = polyfit_ransac(
                y_vals=y_vals.astype(np.float64),
                x_vals=x_vals.astype(np.float64),
                degree=int(degree),
                iterations=self.ransac_iterations,
                inlier_threshold_px=self.ransac_inlier_threshold_px,
                min_inliers=self.ransac_min_inliers,
                random_state=side_seed + self.frame_counter,
            )
            results.append(result)
        best = choose_best_fit(results)
        return best.fit, best.confidence

    def _compute_lane_center_and_width(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
    ) -> Tuple[Optional[float], Optional[float]]:
        y_eval = float(self.bev_height - 1)

        if left_fit is not None and right_fit is not None:
            lx = float(np.polyval(left_fit, y_eval))
            rx = float(np.polyval(right_fit, y_eval))
            return 0.5 * (lx + rx), rx - lx

        if left_fit is not None:
            lx = float(np.polyval(left_fit, y_eval))
            width = self.last_lane_width_px if self.last_lane_width_px is not None else 2.0 * self.single_lane_center_offset_px
            return lx + 0.5 * width, width

        if right_fit is not None:
            rx = float(np.polyval(right_fit, y_eval))
            width = self.last_lane_width_px if self.last_lane_width_px is not None else 2.0 * self.single_lane_center_offset_px
            return rx - 0.5 * width, width

        return None, None

    def _estimate_curvature_from_available(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
    ) -> float:
        y_eval = float(self.bev_height - 1)
        values = []
        if left_fit is not None:
            values.append(curvature_radius_m(left_fit, y_eval, self.ym_per_px, self.xm_per_px))
        if right_fit is not None:
            values.append(curvature_radius_m(right_fit, y_eval, self.ym_per_px, self.xm_per_px))
        if not values:
            return 0.0
        return float(np.mean(values))

    def _build_annotated_image(
        self,
        frame_bgr: np.ndarray,
        binary: np.ndarray,
        windows,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        lane_info: Dict[str, object],
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        bev_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for xl, yl, xh, yh in windows:
            cv2.rectangle(bev_vis, (int(xl), int(yl)), (int(xh), int(yh)), (0, 180, 0), 1)

        ys = np.linspace(0, self.bev_height - 1, self.bev_height)
        lane_fill = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)

        if left_fit is not None:
            lx = np.polyval(left_fit, ys)
            left_pts = np.stack([lx, ys], axis=1).astype(np.int32)
            cv2.polylines(bev_vis, [left_pts], False, (255, 0, 0), 2)
        else:
            left_pts = None

        if right_fit is not None:
            rx = np.polyval(right_fit, ys)
            right_pts = np.stack([rx, ys], axis=1).astype(np.int32)
            cv2.polylines(bev_vis, [right_pts], False, (0, 0, 255), 2)
        else:
            right_pts = None

        if left_pts is not None and right_pts is not None:
            polygon = np.vstack([left_pts, right_pts[::-1]])
            cv2.fillPoly(lane_fill, [polygon], (0, 180, 0))

        unwarped = cv2.warpPerspective(lane_fill, self.M_inv, (w, h))
        overlay = cv2.addWeighted(frame_bgr, 1.0, unwarped, 0.35, 0.0)

        cv2.putText(
            overlay,
            f"conf={lane_info['confidence']:.2f} curv={lane_info['curvature_m']:.1f}m dir={lane_info['curve_direction']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"fallback={lane_info['using_fallback']} sanity={lane_info['sanity_ok']} fps={lane_info['fps']}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )

        bev_panel = cv2.resize(bev_vis, (w, h))
        return np.hstack([overlay, bev_panel])

    @staticmethod
    def _image_msg_to_cv2(msg: Image) -> Optional[np.ndarray]:
        try:
            if msg.encoding in ("bgr8", "8UC3"):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return frame.copy()
            if msg.encoding == "rgb8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if msg.encoding in ("mono8", "8UC1"):
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
        return None

    @staticmethod
    def _cv2_to_image_msg(image_bgr: np.ndarray) -> Image:
        msg = Image()
        msg.height = int(image_bgr.shape[0])
        msg.width = int(image_bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(image_bgr.shape[1] * 3)
        msg.data = image_bgr.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
