#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception Module — Team Cathı / BFMC
======================================
Encapsulates every vision task:
  1. Thread-safe YOLOv8 inference (background worker)
  2. Detection parsing into typed dataclasses
  3. Traffic-light colour classification via HSV
  4. Temporal smoothing of detections (EMA filter)

The background thread ensures that camera callbacks never block on
inference — the node always processes the *latest* frame.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from car_brain.config import (
    DrivingConfig,
    CLASS_TRAFFIC_LIGHT,
)

# ── Optional heavy imports (graceful degradation) ──────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Detection:
    """Lightweight, typed container for a single YOLO detection."""
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class PerceptionResult:
    """Everything the perception pipeline produces for one frame."""
    detections: List[Detection] = field(default_factory=list)
    traffic_light_colour: str = "unknown"  # "red" | "yellow" | "green" | "unknown"
    annotated_frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    inference_ms: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Image conversion helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None


def ros_image_to_cv2(msg) -> Optional[np.ndarray]:
    """Convert a sensor_msgs/Image to a BGR numpy array."""
    # Attempt cv_bridge first
    if _bridge is not None:
        try:
            img = _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass

    # Manual fallback
    try:
        enc = msg.encoding
        h, w = msg.height, msg.width
        if enc == "rgb8":
            frame = np.frombuffer(msg.data, np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            return np.frombuffer(msg.data, np.uint8).reshape(h, w, 3)
        elif enc in ("mono8", "8UC1"):
            frame = np.frombuffer(msg.data, np.uint8).reshape(h, w)
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif enc == "bgra8":
            frame = np.frombuffer(msg.data, np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif enc == "rgba8":
            frame = np.frombuffer(msg.data, np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            ch = len(msg.data) // (h * w)
            frame = np.frombuffer(msg.data, np.uint8).reshape(h, w, ch)
            if ch == 1:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame
    except Exception:
        return None


def cv2_to_ros_image(cv_image: np.ndarray, stamp=None) -> "Image":
    """Convert a BGR numpy image to sensor_msgs/Image."""
    from sensor_msgs.msg import Image as RosImage

    if _bridge is not None:
        try:
            return _bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        except Exception:
            pass

    msg = RosImage()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = "camera"
    msg.height, msg.width = cv_image.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = cv_image.tobytes()
    return msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Traffic-light colour classifier (HSV thresholding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def classify_traffic_light(
    frame: np.ndarray,
    det: Detection,
    cfg: DrivingConfig,
) -> str:
    """
    Crop the traffic-light bounding box from *frame*, convert to HSV
    and count red / yellow / green pixels.  Returns the dominant colour
    string or ``"unknown"`` if ambiguous.
    """
    x1 = max(0, int(det.x1))
    y1 = max(0, int(det.y1))
    x2 = min(frame.shape[1], int(det.x2))
    y2 = min(frame.shape[0], int(det.y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_px = hsv.shape[0] * hsv.shape[1]
    min_px = int(total_px * cfg.tl_min_pixel_ratio)

    # Red (wraps around 0° in HSV)
    mask_r1 = cv2.inRange(
        hsv,
        np.array(cfg.tl_red_lower, dtype=np.uint8),
        np.array(cfg.tl_red_upper, dtype=np.uint8),
    )
    mask_r2 = cv2.inRange(
        hsv,
        np.array(cfg.tl_red_lower2, dtype=np.uint8),
        np.array(cfg.tl_red_upper2, dtype=np.uint8),
    )
    red_count = int(cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2))

    # Green
    mask_g = cv2.inRange(
        hsv,
        np.array(cfg.tl_green_lower, dtype=np.uint8),
        np.array(cfg.tl_green_upper, dtype=np.uint8),
    )
    green_count = int(cv2.countNonZero(mask_g))

    # Yellow
    mask_y = cv2.inRange(
        hsv,
        np.array(cfg.tl_yellow_lower, dtype=np.uint8),
        np.array(cfg.tl_yellow_upper, dtype=np.uint8),
    )
    yellow_count = int(cv2.countNonZero(mask_y))

    counts = {"red": red_count, "yellow": yellow_count, "green": green_count}
    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]

    if counts[dominant] < min_px:
        return "unknown"
    return dominant


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Threaded perception engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PerceptionEngine:
    """
    Owns the YOLO model and runs inference in a daemon thread.

    Call flow (from ROS node):
        1. ``engine.push_frame(cv_frame)``   – non-blocking
        2. ``result = engine.get_result()``   – returns latest result or None
    """

    def __init__(self, cfg: DrivingConfig, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger
        self._model = None

        # ── Load model ──────────────────────────────────────────
        if not YOLO_AVAILABLE:
            self._log_warn("ultralytics not installed — CRUISE-ONLY mode.")
        elif not os.path.isfile(cfg.model_path):
            self._log_warn(f"Model not found: {cfg.model_path} — CRUISE-ONLY mode.")
        else:
            self._model = YOLO(cfg.model_path)
            self._log_info(f"YOLOv8 model loaded: {cfg.model_path}")

        # ── Thread-safe buffers ─────────────────────────────────
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[PerceptionResult] = None
        self._frame_available = threading.Event()

        # ── Worker thread ───────────────────────────────────────
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Public API ──────────────────────────────────────────────

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def push_frame(self, frame: np.ndarray) -> None:
        """Submit a new frame — only the latest is kept (drop-oldest)."""
        with self._frame_lock:
            self._latest_frame = frame
        self._frame_available.set()

    def get_result(self) -> Optional[PerceptionResult]:
        """Return the most recent inference result (or None)."""
        with self._result_lock:
            return self._latest_result

    def shutdown(self) -> None:
        self._running = False
        self._frame_available.set()
        self._thread.join(timeout=2.0)

    # ── Background worker ───────────────────────────────────────

    def _worker(self) -> None:
        """Continuously process the latest frame."""
        while self._running:
            self._frame_available.wait(timeout=0.5)
            self._frame_available.clear()

            # Grab latest frame (swap to None so we skip stale repeats)
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            result = self._run_inference(frame)

            with self._result_lock:
                self._latest_result = result

    def _run_inference(self, frame: np.ndarray) -> PerceptionResult:
        """Execute YOLO + post-processing on one frame."""
        result = PerceptionResult(timestamp=time.monotonic())

        if self._model is None:
            return result

        t0 = time.perf_counter()
        try:
            yolo_results = self._model(
                frame, conf=self._cfg.confidence_threshold, verbose=False,
            )
        except Exception as exc:
            self._log_error(f"YOLO inference failed: {exc}")
            return result

        result.inference_ms = (time.perf_counter() - t0) * 1000.0

        # ── Parse detections ────────────────────────────────────
        if yolo_results and len(yolo_results) > 0:
            res0 = yolo_results[0]
            boxes = res0.boxes
            names = res0.names or {}

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = (
                        boxes.xyxy[i].cpu().numpy().astype(float)
                    )
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = names.get(cls_id, f"class_{cls_id}")

                    result.detections.append(
                        Detection(
                            class_name=cls_name,
                            confidence=conf,
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                        )
                    )

            # ── Traffic-light colour ────────────────────────────
            tl_dets = [
                d for d in result.detections
                if d.class_name == CLASS_TRAFFIC_LIGHT
            ]
            if tl_dets:
                # Pick the largest (closest) traffic light
                biggest = max(tl_dets, key=lambda d: d.area)
                result.traffic_light_colour = classify_traffic_light(
                    frame, biggest, self._cfg,
                )

            # ── Annotated frame for visualisation ───────────────
            if self._cfg.publish_visualisation:
                try:
                    result.annotated_frame = res0.plot()
                except Exception:
                    result.annotated_frame = frame.copy()

        return result

    # ── Logging helpers (optional ROS logger) ───────────────────

    def _log_info(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger:
            self._logger.warn(msg)

    def _log_error(self, msg: str) -> None:
        if self._logger:
            self._logger.error(msg)
