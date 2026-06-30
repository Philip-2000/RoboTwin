from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..core import Detection, DetectionSet
from .base import BaseDetector


def _bbox(contour) -> tuple[float, float, float, float]:
    x, y, w, h = cv2.boundingRect(contour)
    return (float(x), float(y), float(x + w), float(y + h))


def _clean(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _contours(mask: np.ndarray, min_area: float = 80.0, max_area: float = 20000.0):
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for contour in found:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            contours.append((area, contour))
    return sorted(contours, key=lambda x: x[0], reverse=True)


def _mask_hsv(hsv: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(
        _mask_hsv(hsv, (0, 80, 80), (10, 255, 255)),
        _mask_hsv(hsv, (170, 80, 80), (180, 255, 255)),
    )


class SimpleCVDetector(BaseDetector):
    """Small OpenCV detector for early, low-object-count RoboTwin scenes.

    This is not intended to replace open-vocabulary detectors. It gives us a
    dependency-light path for testing the reconstruction data flow.
    """

    name = "simple_cv"

    def __init__(self, task_name: str):
        self.task_name = task_name

    def detect(self, image_path: str | Path, prompts: tuple[str, ...] = ()) -> DetectionSet:
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        if self.task_name == "place_empty_cup":
            detections = self._detect_place_empty_cup(hsv)
        elif self.task_name == "click_bell":
            detections = self._detect_click_bell(hsv)
        elif self.task_name == "stack_blocks_three":
            detections = self._detect_stack_blocks_three(hsv)
        elif self.task_name == "beat_block_hammer":
            detections = self._detect_beat_block_hammer(hsv)
        else:
            detections = ()

        return DetectionSet(
            image_path=image_path,
            detections=tuple(detections),
            source=self.name,
            metadata={"task_name": self.task_name, "prompts": list(prompts)},
        )

    def _detect_place_empty_cup(self, hsv: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        blue = _clean(_mask_hsv(hsv, (85, 35, 45), (125, 255, 255)))
        for area, contour in _contours(blue, min_area=120, max_area=8000)[:2]:
            detections.append(Detection("cup", _bbox(contour), score=min(1.0, area / 1800), source=self.name))

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray = np.where((sat < 45) & (val > 70) & (val < 230), 255, 0).astype(np.uint8)
        gray[:25, :] = 0
        gray = _clean(gray)
        candidates = []
        for area, contour in _contours(gray, min_area=180, max_area=6000):
            x, y, w, h = cv2.boundingRect(contour)
            if 0.55 <= w / max(h, 1) <= 1.8:
                candidates.append((area, contour))
        for area, contour in candidates[:2]:
            detections.append(Detection("coaster", _bbox(contour), score=min(1.0, area / 1600), source=self.name))
        return detections

    def _detect_click_bell(self, hsv: np.ndarray) -> list[Detection]:
        blue = _mask_hsv(hsv, (85, 35, 45), (125, 255, 255))
        yellow = _mask_hsv(hsv, (15, 60, 80), (40, 255, 255))
        mask = _clean(cv2.bitwise_or(blue, yellow))
        contours = _contours(mask, min_area=80, max_area=5000)
        if not contours:
            return []
        area, contour = contours[0]
        return [Detection("bell", _bbox(contour), score=min(1.0, area / 1200), source=self.name)]

    def _detect_stack_blocks_three(self, hsv: np.ndarray) -> list[Detection]:
        specs = [
            ("red_block", _red_mask(hsv)),
            ("green_block", _mask_hsv(hsv, (45, 80, 80), (85, 255, 255))),
            ("blue_block", _mask_hsv(hsv, (100, 80, 60), (130, 255, 255))),
        ]
        detections: list[Detection] = []
        for label, mask in specs:
            mask = _clean(mask)
            contours = _contours(mask, min_area=100, max_area=8000)
            if contours:
                area, contour = contours[0]
                detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 1400), source=self.name))
        return detections

    def _detect_beat_block_hammer(self, hsv: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        red = _clean(_red_mask(hsv))
        contours = _contours(red, min_area=100, max_area=5000)
        if contours:
            area, contour = contours[0]
            detections.append(Detection("red_block", _bbox(contour), score=min(1.0, area / 1200), source=self.name))

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        dark = np.where((val < 95) & (sat < 120), 255, 0).astype(np.uint8)
        dark[:25, :] = 0
        dark[:, :20] = 0
        dark[:, -20:] = 0
        dark = _clean(dark, ksize=3)
        hammer_candidates = []
        for area, contour in _contours(dark, min_area=120, max_area=6000):
            x, y, w, h = cv2.boundingRect(contour)
            if h > w * 1.8:
                hammer_candidates.append((area, contour))
        if hammer_candidates:
            area, contour = hammer_candidates[0]
            detections.append(Detection("hammer", _bbox(contour), score=min(1.0, area / 1500), source=self.name))
        return detections


def detection_set_to_json(detections: DetectionSet) -> dict:
    data = {
        "image_path": str(detections.image_path),
        "source": detections.source,
        "metadata": detections.metadata,
        "detections": [],
    }
    for det in detections.detections:
        data["detections"].append(
            {
                "label": det.label,
                "bbox_xyxy": list(det.bbox_xyxy),
                "score": det.score,
                "source": det.source,
                "mask_path": None if det.mask_path is None else str(det.mask_path),
                "attributes": det.attributes,
            }
        )
    return data


def draw_detections(image_path: str | Path, detections: DetectionSet, out_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    for det in detections.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label = det.label if det.score is None else f"{det.label} {det.score:.2f}"
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def write_detection_json(detections: DetectionSet, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(detection_set_to_json(detections), ensure_ascii=False, indent=2), encoding="utf-8")

