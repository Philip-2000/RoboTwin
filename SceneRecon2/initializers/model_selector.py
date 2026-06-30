from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .place_empty_cup import Detection


@dataclass(frozen=True)
class ModelSelection:
    modelname: str
    model_id: int
    candidates: list[int]
    source: str
    confidence: float
    scores: dict[int, float]
    features: dict[str, float]
    notes: list[str]
    model_id_to_asset: dict[int, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = {str(k): v for k, v in self.scores.items()}
        if self.model_id_to_asset is not None:
            data["model_id_to_asset"] = {str(k): v for k, v in self.model_id_to_asset.items()}
        return data


def select_model(first_frame: Path, detection: Detection, modelname: str, candidates: list[int]) -> ModelSelection:
    """Choose a model id from cheap visual features in the detected first-frame ROI.

    This is intentionally lightweight. It records scores and features so the
    selection can be audited, and can later be swapped for render-template
    scoring without changing initializer state shape.
    """

    image = Image.open(first_frame).convert("RGB")
    roi = _crop_detection(image, detection, pad=4)
    features = _roi_features(roi, detection)
    if modelname == "050_bell":
        return _select_bell(features, candidates)
    if modelname == "048_stapler":
        return _select_stapler(features, candidates)
    if modelname == "056_switch":
        return _select_switch(features, candidates)
    if modelname == "080_pillbottle":
        return _select_pillbottle(features, candidates)
    if modelname == "001_bottle":
        return _select_bottle_001(features, candidates)
    scores = {int(candidate): 0.0 for candidate in candidates}
    return ModelSelection(
        modelname=modelname,
        model_id=int(candidates[0]) if candidates else 0,
        candidates=[int(candidate) for candidate in candidates],
        source="heuristic_roi_v1",
        confidence=0.0,
        scores=scores,
        features=features,
        notes=[f"No selector rule for {modelname}; using first candidate."],
    )


def _crop_detection(image: Image.Image, detection: Detection, pad: int) -> Image.Image:
    x1, y1, x2, y2 = detection.bbox_xyxy
    left = max(0, int(np.floor(x1)) - pad)
    top = max(0, int(np.floor(y1)) - pad)
    right = min(image.width, int(np.ceil(x2)) + pad)
    bottom = min(image.height, int(np.ceil(y2)) + pad)
    if right <= left or bottom <= top:
        return image.crop((0, 0, min(1, image.width), min(1, image.height)))
    return image.crop((left, top, right, bottom))


def _roi_features(roi: Image.Image, detection: Detection) -> dict[str, float]:
    rgb = np.asarray(roi, dtype=np.float32) / 255.0
    if rgb.size == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.float32)
    hsv = _rgb_to_hsv(rgb)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    x1, y1, x2, y2 = detection.bbox_xyxy
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return {
        "mean_r": float(rgb[:, :, 0].mean()),
        "mean_g": float(rgb[:, :, 1].mean()),
        "mean_b": float(rgb[:, :, 2].mean()),
        "mean_h": float(hue.mean()),
        "mean_s": float(sat.mean()),
        "mean_v": float(val.mean()),
        "blue_ratio": float(((hue > 0.52) & (hue < 0.72) & (sat > 0.18) & (val > 0.18)).mean()),
        "red_ratio": float((((hue < 0.06) | (hue > 0.94)) & (sat > 0.18) & (val > 0.18)).mean()),
        "yellow_ratio": float(((hue > 0.09) & (hue < 0.18) & (sat > 0.16) & (val > 0.25)).mean()),
        "dark_ratio": float((val < 0.38).mean()),
        "pale_ratio": float((sat < 0.16).mean()),
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": float(w * h),
        "bbox_aspect": float(w / h),
    }


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    rmax = mask & (maxc == r)
    gmax = mask & (maxc == g)
    bmax = mask & (maxc == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue /= 6.0
    sat = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    sat[nonzero] = delta[nonzero] / maxc[nonzero]
    return np.stack([hue, sat, maxc], axis=2)


def _select_bell(features: dict[str, float], candidates: list[int]) -> ModelSelection:
    area = features["bbox_area"]
    # base0 is the larger 0.05-scale bell, base1 the smaller 0.04-scale bell.
    scores = {
        0: _triangular(area, 1300.0, 5200.0),
        1: _triangular(area, 450.0, 2600.0),
    }
    return _finish("050_bell", candidates, scores, features, ["Bell selector uses ROI size; base0 is larger than base1."])


def _select_stapler(features: dict[str, float], candidates: list[int]) -> ModelSelection:
    blue = features["blue_ratio"]
    red = features["red_ratio"]
    dark = features["dark_ratio"]
    pale = features["pale_ratio"]
    aspect = features["bbox_aspect"]
    area = features["bbox_area"]
    # These profiles are deliberately coarse, but color should dominate here:
    # base4/base5 are red, base0/base2/base6 are blue, base1 is dark, and
    # base3 is the pale/silver variant.  The ROI often includes table pixels,
    # so pale_ratio alone is not allowed to override clear red/blue evidence.
    scores = {
        0: 1.20 * _high(blue, 0.08, 0.45) + 0.20 * _triangular(aspect, 1.0, 5.0) + 0.10 * _triangular(area, 180.0, 2400.0),
        1: 1.40 * _high(dark, 0.12, 0.55) + 0.20 * _triangular(aspect, 0.8, 5.0) + 0.10 * _triangular(area, 160.0, 2400.0),
        2: 0.75 * _high(blue, 0.06, 0.40) + 0.55 * _high(dark, 0.08, 0.45) + 0.15 * _triangular(aspect, 0.8, 4.8),
        3: 0.80 * _high(pale - max(red, blue, dark), 0.35, 0.85) + 0.25 * _triangular(aspect, 0.45, 3.0),
        4: 1.45 * _high(red, 0.05, 0.38) + 0.20 * _triangular(aspect, 0.45, 4.5) + 0.10 * _triangular(area, 120.0, 2400.0),
        5: 1.15 * _high(red, 0.08, 0.45) + 0.20 * _triangular(aspect, 0.35, 2.5) + 0.10 * _triangular(area, 120.0, 2000.0),
        6: 1.15 * _high(blue, 0.10, 0.48) + 0.25 * _triangular(aspect, 0.45, 2.8) + 0.10 * _triangular(area, 120.0, 1800.0),
    }
    return _finish("048_stapler", candidates, scores, features, ["Stapler selector uses color-dominant profiles for red, blue, dark, and silver variants."])


def _select_switch(features: dict[str, float], candidates: list[int]) -> ModelSelection:
    yellow = features["yellow_ratio"]
    red = features["red_ratio"]
    dark = features["dark_ratio"]
    pale = features["pale_ratio"]
    aspect = features["bbox_aspect"]
    area = features["bbox_area"]
    scores = {
        0: 0.45 * _high(yellow, 0.08, 0.50) + 0.30 * _triangular(aspect, 0.8, 3.5) + 0.25 * _triangular(area, 180.0, 3500.0),
        1: 0.45 * _high(dark, 0.18, 0.75) + 0.30 * _triangular(aspect, 0.5, 2.4) + 0.25 * _triangular(area, 180.0, 3500.0),
        2: 0.45 * _high(pale, 0.55, 0.95) + 0.30 * _triangular(aspect, 0.7, 3.4) + 0.25 * _triangular(area, 180.0, 4200.0),
        3: 0.50 * _high(red, 0.08, 0.55) + 0.30 * _triangular(aspect, 0.7, 3.2) + 0.20 * _triangular(area, 180.0, 3500.0),
        4: 0.42 * _high(pale, 0.45, 0.95) + 0.28 * _triangular(aspect, 0.4, 1.9) + 0.30 * _triangular(area, 300.0, 5000.0),
        5: 0.40 * _high(yellow, 0.05, 0.45) + 0.35 * _triangular(aspect, 0.35, 1.8) + 0.25 * _triangular(area, 120.0, 2200.0),
        6: 0.40 * _high(pale, 0.55, 0.98) + 0.35 * _triangular(aspect, 1.6, 6.0) + 0.25 * _triangular(area, 120.0, 2200.0),
        7: 0.40 * _high(dark, 0.10, 0.65) + 0.35 * _triangular(aspect, 0.5, 2.7) + 0.25 * _triangular(area, 180.0, 3000.0),
    }
    return _finish(
        "056_switch",
        candidates,
        scores,
        features,
        ["Switch selector uses coarse face color, darkness, aspect, and size."],
        model_id_to_asset={
            0: "100880",
            1: "100901",
            2: "100905",
            3: "100906",
            4: "100907",
            5: "100914",
            6: "100933",
            7: "100937",
        },
    )


def _select_pillbottle(features: dict[str, float], candidates: list[int]) -> ModelSelection:
    red = features["red_ratio"]
    yellow = features["yellow_ratio"]
    blue = features["blue_ratio"]
    dark = features["dark_ratio"]
    pale = features["pale_ratio"]
    mean_g = features["mean_g"]
    mean_b = features["mean_b"]
    # base1=white/orange, base2=plain white, base3=orange body,
    # base4=brown, base5=teal/white.
    scores = {
        1: 0.45 + 0.85 * _high(red + yellow, 0.06, 0.45),
        2: 0.20 + 1.20 * _high(pale - max(red, yellow, blue, dark), 0.25, 0.85),
        3: 0.25 + 1.10 * _high(yellow + red, 0.10, 0.60),
        4: 0.25 + 1.25 * _high(dark, 0.12, 0.65) + 0.25 * _high(red, 0.05, 0.35),
        5: 0.25 + 1.10 * _high(blue, 0.05, 0.45) + 0.25 * max(0.0, mean_b - mean_g),
    }
    return _finish(
        "080_pillbottle",
        candidates,
        scores,
        features,
        ["Pill-bottle selector uses coarse ROI color profiles for orange, white, brown, and teal variants."],
    )


def _select_bottle_001(features: dict[str, float], candidates: list[int]) -> ModelSelection:
    red = features["red_ratio"]
    yellow = features["yellow_ratio"]
    blue = features["blue_ratio"]
    dark = features["dark_ratio"]
    pale = features["pale_ratio"]
    mean_r = features["mean_r"]
    mean_g = features["mean_g"]
    mean_b = features["mean_b"]
    # Coarse profiles for the common WorldArena bottle appearances.
    scores = {
        0: 0.35 + 1.00 * _high(red, 0.08, 0.45) + 0.35 * _high(yellow, 0.04, 0.35),
        1: 0.30 + 0.85 * _high(yellow, 0.08, 0.45) + 0.25 * _high(dark, 0.05, 0.35),
        4: 0.35 + 1.10 * _high(yellow + red, 0.08, 0.55),
        5: 0.30 + 0.75 * _high(yellow + red, 0.08, 0.50) + 0.25 * _high(dark, 0.05, 0.35),
        7: 0.30 + 0.70 * _high(yellow, 0.08, 0.45) + 0.35 * _high(dark, 0.08, 0.45),
        10: 0.25 + 0.90 * _high(dark, 0.10, 0.65) + 0.35 * _high(red, 0.04, 0.35),
        13: 0.35 + 1.10 * _high(dark, 0.12, 0.70) + 0.45 * _high(red, 0.05, 0.35),
        14: 0.30 + 0.95 * _high(blue, 0.06, 0.45) + 0.25 * _high(pale, 0.20, 0.70),
        16: 0.35 + 1.10 * _high(max(0.0, mean_g - max(mean_r, mean_b)), 0.04, 0.28),
        18: 0.25 + 1.15 * _high(pale - max(red, yellow, blue, dark), 0.25, 0.90),
        19: 0.25 + 1.05 * _high(blue, 0.08, 0.50),
    }
    # Keep all candidates represented, even those without a strong profile.
    for candidate in candidates:
        scores.setdefault(int(candidate), 0.0)
    return _finish(
        "001_bottle",
        candidates,
        scores,
        features,
        ["001_bottle selector uses coarse color profiles for coke, orange, green, white, and blue variants."],
    )


def _finish(
    modelname: str,
    candidates: list[int],
    scores: dict[int, float],
    features: dict[str, float],
    notes: list[str],
    model_id_to_asset: dict[int, str] | None = None,
) -> ModelSelection:
    allowed = [int(candidate) for candidate in candidates]
    filtered = {candidate: float(scores.get(candidate, 0.0)) for candidate in allowed}
    if not filtered:
        filtered = {0: 0.0}
        allowed = [0]
    ranked = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, best_score - second + 0.35 * best_score))
    return ModelSelection(
        modelname=modelname,
        model_id=int(best_id),
        candidates=allowed,
        source="heuristic_roi_v1",
        confidence=float(confidence),
        scores=filtered,
        features={key: float(value) for key, value in features.items()},
        notes=notes,
        model_id_to_asset=model_id_to_asset,
    )


def _triangular(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    mid = (low + high) * 0.5
    if value <= low or value >= high:
        return 0.0
    if value <= mid:
        return float((value - low) / (mid - low))
    return float((high - value) / (high - mid))


def _high(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))
