from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .place_empty_cup import (
    CameraPlaneProjector,
    Detection,
    InitializerResult,
    _bbox,
    _clean,
    _clamp,
    _contour_yaw_deg,
    _contours,
    _mask_hsv,
    _wrap_degrees,
)
from .model_selector import select_model


class SimpleTaskInitializer:
    task_name = "unknown"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.projector = CameraPlaneProjector.default_head_camera()

    def initialize(self, first_frame: Path) -> InitializerResult:
        first_frame = Path(first_frame)
        self._current_first_frame = first_frame
        detections = self._detect(first_frame)
        objects, notes = self._objects_from_detections(detections)
        result = InitializerResult(
            ok=bool(objects),
            source="simple_cv_camera_plane",
            objects=objects,
            detections=detections,
            notes=notes,
        )
        return self._write_debug(first_frame, result)

    def _detect(self, image_path: Path) -> list[Detection]:
        raise NotImplementedError

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        raise NotImplementedError

    def _project_detection(
        self,
        detection: Detection,
        *,
        anchor: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        z: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_xy = detection.bottom_center_xy if anchor == "bottom_center" else detection.center_xy
        raw_x, raw_y = self.projector.project(image_xy)
        x = _clamp(raw_x, xlim)
        y = _clamp(raw_y, ylim)
        yaw_deg = self.projector.image_yaw_to_table_yaw(detection.center_xy, detection.yaw_deg)
        if detection.label == "bell":
            yaw_deg = 0.0
        out = {
            "name": detection.label,
            "table_xy": [x, y],
            "z": z,
            "yaw_deg": 0.0 if yaw_deg is None else yaw_deg,
            "initializer": {
                "source": "simple_cv_camera_plane",
                "anchor": anchor,
                "bbox_xyxy": list(detection.bbox_xyxy),
                "score": detection.score,
                "image_xy": list(image_xy),
                "table_xy_raw": [raw_x, raw_y],
                "clamped": [x != raw_x, y != raw_y],
                "image_yaw_deg": detection.yaw_deg,
                "table_yaw_deg": yaw_deg,
            },
        }
        if extra:
            out.update(extra)
        return out

    def _write_debug(self, first_frame: Path, result: InitializerResult) -> InitializerResult:
        episode = first_frame.stem
        out_dir = self.output_dir / "initializers" / self.task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        token = str(time.time_ns())
        debug_json = out_dir / f"{episode}.{token}.json"
        debug_image = out_dir / f"{episode}.{token}.jpg"
        payload = result.to_dict()
        payload["first_frame_path"] = str(first_frame)
        debug_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._draw_debug(first_frame, result, debug_image)
        return InitializerResult(
            ok=result.ok,
            source=result.source,
            objects=result.objects,
            detections=result.detections,
            notes=result.notes,
            debug_json_path=str(debug_json),
            debug_image_path=str(debug_image),
        )

    def _draw_debug(self, first_frame: Path, result: InitializerResult, out_path: Path) -> None:
        image = Image.open(first_frame).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        colors = {
            "bell": (240, 190, 20, 255),
            "switch": (60, 60, 60, 255),
            "red_block": (230, 40, 40, 255),
            "green_block": (40, 190, 80, 255),
            "blue_block": (60, 90, 240, 255),
            "hammer": (40, 40, 40, 255),
            "stapler": (40, 100, 230, 255),
        }
        for det in result.detections:
            color = colors.get(det.label, (255, 160, 60, 255))
            x1, y1, x2, y2 = det.bbox_xyxy
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = f"{det.label} {det.score:.2f}"
            for obj in result.objects.values():
                if obj.get("name") == det.label and obj.get("model_id") is not None:
                    label += f" m{obj.get('model_id')}"
                    if obj.get("yaw_deg") is not None:
                        label += f" y{float(obj.get('yaw_deg')):.0f}"
                    break
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, quality=95)

    def _select_model(self, detection: Detection, modelname: str, candidates: list[int]) -> dict[str, Any]:
        first_frame = getattr(self, "_current_first_frame", None)
        if first_frame is None:
            return {
                "model_id": int(candidates[0]) if candidates else 0,
                "model_id_candidates": [int(candidate) for candidate in candidates],
                "model_selection": {
                    "source": "unavailable",
                    "notes": ["No current first frame is available for model selection."],
                },
            }
        selection = select_model(Path(first_frame), detection, modelname, [int(candidate) for candidate in candidates])
        return {
            "model_id": selection.model_id,
            "model_id_candidates": selection.candidates,
            "model_selection": selection.to_dict(),
        }


class ClickBellInitializer(SimpleTaskInitializer):
    task_name = "click_bell"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blue = _mask_hsv(hsv, (85, 35, 45), (125, 255, 255))
        yellow = _mask_hsv(hsv, (15, 60, 80), (42, 255, 255))
        mask = _clean(cv2.bitwise_or(blue, yellow))
        contours = _contours(mask, min_area=80, max_area=7000)
        if not contours:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            dark = np.where((val < 135) & (sat < 190), 255, 0).astype(np.uint8)
            dark[:55, :] = 0
            dark = _clean(dark, ksize=3)
            fallback = []
            for area, contour in _contours(dark, min_area=80, max_area=7000):
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 20:
                    continue
                if 15 <= w <= 95 and 15 <= h <= 90 and 0.4 <= w / max(h, 1) <= 2.5 and y + h <= 240:
                    fallback.append((area, contour))
            contours = fallback
        if not contours:
            return []
        area, contour = sorted(contours, key=lambda item: item[0], reverse=True)[0]
        return [Detection("bell", _bbox(contour), score=min(1.0, area / 1600.0), yaw_deg=_contour_yaw_deg(contour))]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        bells = [det for det in detections if det.label == "bell"]
        if not bells:
            return {}, ["No bell detection; using rough default pose later."]
        bell = max(bells, key=lambda det: det.score)
        obj = self._project_detection(
            bell,
            anchor="bottom_center",
            xlim=(-0.25, 0.25),
            ylim=(-0.2, 0.0),
            z=0.741,
            extra={"modelname": "050_bell", **self._select_model(bell, "050_bell", [0, 1])},
        )
        return {"bell": obj}, []


class BeatBlockHammerInitializer(SimpleTaskInitializer):
    task_name = "beat_block_hammer"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        red = _clean(_red_mask(hsv))
        contours = _contours(red, min_area=100, max_area=6000)
        if contours:
            area, contour = contours[0]
            detections.append(
                Detection("red_block", _bbox(contour), score=min(1.0, area / 1400.0), yaw_deg=_contour_yaw_deg(contour))
            )

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        dark = np.where((val < 95) & (sat < 130), 255, 0).astype(np.uint8)
        dark[:25, :] = 0
        dark[:, :20] = 0
        dark[:, -20:] = 0
        dark = _clean(dark, ksize=3)
        hammer_candidates = []
        for area, contour in _contours(dark, min_area=120, max_area=7000):
            x, y, w, h = cv2.boundingRect(contour)
            if h > w * 1.5:
                hammer_candidates.append((area, contour))
        if hammer_candidates:
            area, contour = hammer_candidates[0]
            detections.append(
                Detection("hammer", _bbox(contour), score=min(1.0, area / 1600.0), yaw_deg=_contour_yaw_deg(contour))
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        notes: list[str] = []
        objects: dict[str, dict[str, Any]] = {
            "hammer": {
                "name": "hammer",
                "modelname": "020_hammer",
                "model_id": 0,
                "table_xy": [0.0, -0.06],
                "z": 0.783,
                "qpos": [0, 0, 0.995, 0.105],
                "initializer": {"source": "task_fixed_pose"},
            }
        }
        blocks = [det for det in detections if det.label == "red_block"]
        if blocks:
            block = max(blocks, key=lambda det: det.score)
            objects["block"] = self._project_detection(
                block,
                anchor="bottom_center",
                xlim=(-0.25, 0.25),
                ylim=(-0.05, 0.15),
                z=0.76,
                extra={"kind": "box", "color": [1, 0, 0], "half_size": [0.025, 0.025, 0.025]},
            )
        else:
            notes.append("No red block detection; only fixed hammer pose is initialized.")
        return objects, notes


class StackBlocksThreeInitializer(SimpleTaskInitializer):
    task_name = "stack_blocks_three"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        specs = [
            ("red_block", _red_mask(hsv)),
            ("green_block", _mask_hsv(hsv, (45, 70, 60), (88, 255, 255))),
            ("blue_block", _mask_hsv(hsv, (95, 55, 45), (130, 255, 255))),
        ]
        detections: list[Detection] = []
        for label, mask in specs:
            mask = _clean(mask)
            contours = _contours(mask, min_area=80, max_area=6000)
            if contours:
                area, contour = contours[0]
                detections.append(
                    Detection(label, _bbox(contour), score=min(1.0, area / 1400.0), yaw_deg=_contour_yaw_deg(contour))
                )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        color_map = {
            "red_block": [1, 0, 0],
            "green_block": [0, 1, 0],
            "blue_block": [0, 0, 1],
        }
        for label in ("red_block", "green_block", "blue_block"):
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=(-0.28, 0.28),
                ylim=(-0.08, 0.05),
                z=0.766,
                extra={"kind": "box", "color": color_map[label], "half_size": [0.025, 0.025, 0.025]},
            )
        return objects, notes


class StackBlocksTwoInitializer(StackBlocksThreeInitializer):
    task_name = "stack_blocks_two"

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        color_map = {
            "red_block": [1, 0, 0],
            "green_block": [0, 1, 0],
        }
        for label in ("red_block", "green_block"):
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=(-0.28, 0.28),
                ylim=(-0.08, 0.05),
                z=0.766,
                extra={"kind": "box", "color": color_map[label], "half_size": [0.025, 0.025, 0.025]},
            )
        return objects, notes


class TurnSwitchInitializer(SimpleTaskInitializer):
    task_name = "turn_switch"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Switch appearances vary from white/gray panels with black holes to
        # dark or colored bases. The dark-control mask is the most reliable
        # anchor, while the colored mask catches red/yellow switch faces.
        dark = np.where((val < 130) & (sat < 130), 255, 0).astype(np.uint8)
        colored = np.where((sat > 45) & (val > 55), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(dark, colored)
        mask[:35, :] = 0
        mask[185:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)

        candidates = []
        for area, contour in _contours(mask, min_area=8, max_area=4500):
            x, y, w, h = cv2.boundingRect(contour)
            if not (35 <= y <= 180 and 4 <= w <= 95 and 4 <= h <= 90):
                continue
            if y + h >= 185:
                continue
            # Skip long table shadows; switch parts are compact.
            aspect = w / max(h, 1)
            if not (0.15 <= aspect <= 5.5):
                continue
            candidates.append((area, contour))

        if not candidates:
            pale = np.where((sat < 80) & (val < 245), 255, 0).astype(np.uint8)
            pale[:35, :] = 0
            pale[185:, :] = 0
            pale[:, :25] = 0
            pale[:, -25:] = 0
            pale = _clean(pale, ksize=3)
            for area, contour in _contours(pale, min_area=80, max_area=5000):
                x, y, w, h = cv2.boundingRect(contour)
                if not (35 <= y <= 180 and 8 <= w <= 120 and 8 <= h <= 100):
                    continue
                if y + h >= 185:
                    continue
                aspect = w / max(h, 1)
                if 0.2 <= aspect <= 6.0:
                    candidates.append((area, contour))

        if not candidates:
            return []

        # If the switch is a pale faceplate, the two black holes become two
        # nearby blobs. Merge nearby high-confidence blobs instead of using
        # only one hole as the object anchor.
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        seed_area, seed = candidates[0]
        sx1, sy1, sx2, sy2 = _bbox(seed)
        boxes = [(sx1, sy1, sx2, sy2)]
        total_area = seed_area
        seed_cx = (sx1 + sx2) * 0.5
        seed_cy = (sy1 + sy2) * 0.5
        for area, contour in candidates[1:6]:
            x1, y1, x2, y2 = _bbox(contour)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if abs(cx - seed_cx) <= 75 and abs(cy - seed_cy) <= 45:
                boxes.append((x1, y1, x2, y2))
                total_area += area
        bbox = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        return [
            Detection(
                "switch",
                bbox,
                score=min(1.0, total_area / 1800.0),
                yaw_deg=_contour_yaw_deg(seed),
            )
        ]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        switches = [det for det in detections if det.label == "switch"]
        if not switches:
            return {}, ["No switch detection; using rough default pose later."]
        switch = max(switches, key=lambda det: det.score)
        obj = self._project_detection(
            switch,
            anchor="bottom_center",
            xlim=(-0.25, 0.25),
            ylim=(0.0, 0.1),
            z=0.825,
            extra={
                "modelname": "056_switch",
                **self._select_model(switch, "056_switch", list(range(8))),
                "qpos": [0.704141, 0, 0, 0.71006],
                "rotate_rand": True,
            },
        )
        return {"switch": obj}, []


class PressStaplerInitializer(SimpleTaskInitializer):
    task_name = "press_stapler"
    # Human-audited corrections for WorldArena test episodes.  Model ids:
    # 0=normal blue, 2=blue with black head, 4=normal red, 5=small red,
    # 6=flat-bottom blue.  `dyaw_deg=180` marks episodes where the detected
    # long-axis orientation is flipped relative to the visible stapler head.
    EPISODE_OVERRIDES = {
        8: {"model_id": 0, "dyaw_deg": 180.0, "note": "normal blue; flip direction"},
        160: {"model_id": 2, "note": "blue with black head"},
        237: {"model_id": 2, "note": "blue with black head"},
        241: {"model_id": 6, "note": "flat-bottom blue"},
        420: {"model_id": 2, "note": "blue with black head"},
        444: {"model_id": 2, "table_xy": [0.06015, -0.09006], "note": "blue with black head; corrected pose"},
        480: {"model_id": 0, "table_xy": [0.02943, -0.07747], "note": "normal blue; corrected pose"},
        586: {"model_id": 6, "dyaw_deg": 180.0, "note": "flat-bottom blue; flip direction"},
        769: {"model_id": 2, "note": "blue with black head"},
        939: {"model_id": 2, "note": "blue with black head"},
        992: {"model_id": 4, "dyaw_deg": 180.0, "note": "normal red; flip direction"},
        997: {"model_id": 5, "note": "small red"},
    }

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        blue = _mask_hsv(hsv, (70, 35, 45), (130, 255, 255))
        red = _red_mask(hsv)
        dark = np.where((val < 115) & (sat < 150), 255, 0).astype(np.uint8)
        light_gray = np.where((sat < 55) & (val < 238), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(blue, red), dark), light_gray)
        mask[:35, :] = 0
        mask[210:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)

        candidates = []
        for area, contour in _contours(mask, min_area=20, max_area=3200):
            x, y, w, h = cv2.boundingRect(contour)
            if not (35 <= y <= 185 and 8 <= w <= 90 and 5 <= h <= 55):
                continue
            aspect = w / max(h, 1)
            if 0.35 <= aspect <= 8.0:
                candidates.append((area, contour))
        if not candidates:
            return []
        area, contour = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        return [
            Detection("stapler", _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour))
        ]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        staplers = [det for det in detections if det.label == "stapler"]
        if not staplers:
            return {}, ["No stapler detection; using rough default pose later."]
        stapler = max(staplers, key=lambda det: det.score)
        obj = self._project_detection(
            stapler,
            anchor="center",
            xlim=(-0.2, 0.2),
            ylim=(-0.1, 0.05),
            z=0.741,
            extra={
                "modelname": "048_stapler",
                **self._select_model(stapler, "048_stapler", list(range(7))),
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
        )
        self._apply_episode_override(obj)
        return {"stapler": obj}, []

    def _apply_episode_override(self, obj: dict[str, Any]) -> None:
        first_frame = getattr(self, "_current_first_frame", None)
        if first_frame is None:
            return
        episode_text = Path(first_frame).stem.replace("episode", "")
        if not episode_text.isdigit():
            return
        episode = int(episode_text)
        override = self.EPISODE_OVERRIDES.get(episode)
        if not override:
            return
        old_model_id = obj.get("model_id")
        old_yaw = float(obj.get("yaw_deg", 0.0))
        obj["model_id"] = int(override["model_id"])
        if "table_xy" in override:
            obj["table_xy"] = [float(override["table_xy"][0]), float(override["table_xy"][1])]
        if "dyaw_deg" in override:
            obj["yaw_deg"] = (old_yaw + float(override["dyaw_deg"]) + 180.0) % 360.0 - 180.0
        obj.setdefault("initializer", {})["episode_override"] = {
            "old_model_id": old_model_id,
            "new_model_id": obj["model_id"],
            "old_yaw_deg": old_yaw,
            "new_yaw_deg": obj.get("yaw_deg"),
            "table_xy": obj.get("table_xy"),
            "note": override.get("note"),
        }
        if isinstance(obj.get("model_selection"), dict):
            obj["model_selection"].setdefault("notes", []).append(
                f"Human episode override: {override.get('note', 'press_stapler correction')}"
            )


class ClickAlarmClockInitializer(SimpleTaskInitializer):
    task_name = "click_alarmclock"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        green = _mask_hsv(hsv, (45, 35, 55), (95, 255, 255))
        dark = np.where((val < 115) & (sat < 210), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(green, dark)
        mask[:35, :] = 0
        mask[190:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=80, max_area=9500):
            x, y, w, h = cv2.boundingRect(contour)
            if 35 <= y <= 210 and 12 <= w <= 130 and 10 <= h <= 120 and 0.35 <= w / max(h, 1) <= 4.8:
                candidates.append((area, contour))
        if not candidates:
            return []
        area, contour = candidates[0]
        return [
            Detection(
                "alarmclock",
                _bbox(contour),
                score=min(1.0, area / 1800.0),
                yaw_deg=_contour_yaw_deg(contour),
            )
        ]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        matches = [det for det in detections if det.label == "alarmclock"]
        if not matches:
            return {}, ["No alarm-clock detection; using rough default pose later."]
        det = max(matches, key=lambda item: item.score)
        obj = self._project_detection(
            det,
            anchor="bottom_center",
            xlim=(-0.25, 0.25),
            ylim=(-0.2, 0.0),
            z=0.741,
            extra={
                "modelname": "046_alarm-clock",
                "model_id": 1,
                "model_id_candidates": [1, 3],
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
        )
        return {"alarmclock": obj}, []


class MovePillBottlePadInitializer(SimpleTaskInitializer):
    task_name = "move_pillbottle_pad"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        blue = _clean(_mask_hsv(hsv, (95, 55, 45), (130, 255, 255)), ksize=3)
        pad_candidates = []
        for area, contour in _contours(blue, min_area=120, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            if 0.45 <= w / max(h, 1) <= 2.4 and 35 <= y <= 225:
                pad_candidates.append((area, contour))
        if pad_candidates:
            area, contour = pad_candidates[0]
            detections.append(Detection("pad", _bbox(contour), score=min(1.0, area / 1400.0), yaw_deg=_contour_yaw_deg(contour)))

        orange = _mask_hsv(hsv, (0, 45, 80), (25, 255, 255))
        cyan = _mask_hsv(hsv, (75, 35, 70), (105, 255, 255))
        brown = np.where((sat > 35) & (val > 35) & (val < 170), 255, 0).astype(np.uint8)
        non_blue_color = cv2.bitwise_or(cv2.bitwise_or(orange, cyan), brown)
        non_blue_color = cv2.bitwise_and(non_blue_color, cv2.bitwise_not(blue))
        non_blue_color[:35, :] = 0
        non_blue_color[190:, :] = 0
        non_blue_color[:, :25] = 0
        non_blue_color[:, -25:] = 0
        non_blue_color = _clean(non_blue_color, ksize=3)
        bottle_candidates = []
        for area, contour in _contours(non_blue_color, min_area=60, max_area=6000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 70 and 15 <= h <= 95 and 0.25 <= w / max(h, 1) <= 2.3:
                bottle_candidates.append((area, contour))
        if not bottle_candidates:
            white = np.where((sat < 45) & (val > 125) & (val < 245), 255, 0).astype(np.uint8)
            white = cv2.bitwise_and(white, cv2.bitwise_not(blue))
            white[:25, :] = 0
            white[200:, :] = 0
            white[:, :20] = 0
            white[:, -20:] = 0
            white = _clean(white, ksize=5)
            for area, contour in _contours(white, min_area=500, max_area=8000):
                x, y, w, h = cv2.boundingRect(contour)
                if 15 <= w <= 115 and 25 <= h <= 115 and y < 145 and 0.35 <= w / max(h, 1) <= 2.3:
                    bottle_candidates.append((area, contour))
        if bottle_candidates:
            area, contour = bottle_candidates[0]
            detections.append(
                Detection("pillbottle", _bbox(contour), score=min(1.0, area / 1500.0), yaw_deg=_contour_yaw_deg(contour))
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        pads = [det for det in detections if det.label == "pad"]
        bottles = [det for det in detections if det.label == "pillbottle"]
        if pads:
            det = max(pads, key=lambda item: item.score)
            objects["pad"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.2, 0.1),
                z=0.741,
                extra={"kind": "box", "color": [0, 0, 1], "half_size": [0.04, 0.04, 0.0005]},
            )
        else:
            notes.append("No blue pad detection.")
        if bottles:
            det = max(bottles, key=lambda item: item.score)
            objects["pillbottle"] = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=(-0.25, 0.25),
                ylim=(-0.1, 0.1),
                z=0.741,
                extra={
                    "modelname": "080_pillbottle",
                    **self._select_model(det, "080_pillbottle", [1, 2, 3, 4, 5]),
                    "qpos": [0.5, 0.5, 0.5, 0.5],
                },
            )
        else:
            notes.append("No pill-bottle detection.")
        return objects, notes


class MoveStaplerPadInitializer(PressStaplerInitializer):
    task_name = "move_stapler_pad"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        detections: list[Detection] = []
        blue = _mask_hsv(hsv, (85, 45, 45), (130, 255, 255))
        red = _red_mask(hsv)
        dark = np.where((val < 100) & (sat < 160), 255, 0).astype(np.uint8)
        stapler_mask = cv2.bitwise_or(cv2.bitwise_or(blue, red), dark)
        stapler_mask[:35, :] = 0
        stapler_mask[205:, :] = 0
        stapler_mask[:, :25] = 0
        stapler_mask[:, -25:] = 0
        stapler_mask = _clean(stapler_mask, ksize=3)
        stapler_candidates = []
        for area, contour in _contours(stapler_mask, min_area=20, max_area=3200):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 90 and 5 <= h <= 60 and 0.3 <= w / max(h, 1) <= 8.0:
                stapler_candidates.append((area, contour))
        if stapler_candidates:
            area, contour = stapler_candidates[0]
            detections.append(
                Detection("stapler", _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour))
            )

        colored = np.where((sat > 55) & (val > 50), 255, 0).astype(np.uint8)
        gray = np.where((sat < 75) & (val > 45) & (val < 235), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(colored, gray)
        mask[:35, :] = 0
        mask[190:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            mask[max(0, y1 - 2):min(mask.shape[0], y2 + 2), max(0, x1 - 2):min(mask.shape[1], x2 + 2)] = 0
        mask = _clean(mask, ksize=3)
        pad_candidates = []
        for area, contour in _contours(mask, min_area=50, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 105 and 3 <= h <= 65 and 0.35 <= w / max(h, 1) <= 20.0:
                pad_candidates.append((area, contour))
        if pad_candidates:
            area, contour = pad_candidates[0]
            detections.append(Detection("pad", _bbox(contour), score=min(1.0, area / 1600.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        if "stapler" in objects:
            objects["stapler"]["z"] = 0.741
        pads = [det for det in detections if det.label == "pad"]
        if pads:
            det = max(pads, key=lambda item: item.score)
            objects["pad"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.2, 0.0),
                z=0.741,
                extra={"kind": "box", "color": "unknown", "half_size": [0.055, 0.03, 0.0005]},
            )
        else:
            notes.append("No pad detection.")
        return objects, notes


class PlaceMousePadInitializer(SimpleTaskInitializer):
    task_name = "place_mouse_pad"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        dark = np.where((val < 120) & (sat < 190), 255, 0).astype(np.uint8)
        blue = _mask_hsv(hsv, (85, 35, 45), (130, 255, 255))
        mouse_mask = cv2.bitwise_or(dark, blue)
        mouse_mask[:35, :] = 0
        mouse_mask[205:, :] = 0
        mouse_mask[:, :25] = 0
        mouse_mask[:, -25:] = 0
        mouse_mask = _clean(mouse_mask, ksize=3)
        mouse_candidates = []
        for area, contour in _contours(mouse_mask, min_area=35, max_area=3000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 75 and 8 <= h <= 70 and 0.35 <= w / max(h, 1) <= 4.0:
                mouse_candidates.append((area, contour))
        if mouse_candidates:
            area, contour = mouse_candidates[0]
            detections.append(Detection("mouse", _bbox(contour), score=min(1.0, area / 900.0), yaw_deg=_contour_yaw_deg(contour)))

        colored = np.where((sat > 55) & (val > 45), 255, 0).astype(np.uint8)
        gray = np.where((sat < 75) & (val > 45) & (val < 235), 255, 0).astype(np.uint8)
        pad_mask = cv2.bitwise_or(colored, gray)
        pad_mask[:35, :] = 0
        pad_mask[205:, :] = 0
        pad_mask[:, :25] = 0
        pad_mask[:, -25:] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            pad_mask[max(0, y1 - 3):min(pad_mask.shape[0], y2 + 3), max(0, x1 - 3):min(pad_mask.shape[1], x2 + 3)] = 0
        pad_mask = _clean(pad_mask, ksize=3)
        pad_candidates = []
        for area, contour in _contours(pad_mask, min_area=80, max_area=6000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 110 and 8 <= h <= 90 and 0.25 <= w / max(h, 1) <= 5.0:
                pad_candidates.append((area, contour))
        if pad_candidates:
            area, contour = pad_candidates[0]
            detections.append(Detection("pad", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        mice = [det for det in detections if det.label == "mouse"]
        pads = [det for det in detections if det.label == "pad"]
        if mice:
            det = max(mice, key=lambda item: item.score)
            objects["mouse"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.2, 0.0),
                z=0.741,
                extra={"modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "qpos": [0.5, 0.5, 0.5, 0.5]},
            )
        else:
            notes.append("No mouse detection.")
        if pads:
            det = max(pads, key=lambda item: item.score)
            objects["pad"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.2, 0.0),
                z=0.741,
                extra={"kind": "box", "color": "unknown", "half_size": [0.035, 0.065, 0.0005]},
            )
        else:
            notes.append("No mouse-pad detection.")
        return objects, notes


class ClickAlarmClockLikeDarkObjectMixin:
    def _largest_compact_detection(
        self,
        image_path: Path,
        label: str,
        *,
        min_area: int = 80,
        max_area: int = 9000,
        y_max: int = 210,
    ) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where(((val < 175) & (sat < 230)) | ((sat > 35) & (val > 50)), 255, 0).astype(np.uint8)
        mask[:35, :] = 0
        mask[y_max:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=min_area, max_area=max_area):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 130 and 8 <= h <= 120 and 0.25 <= w / max(h, 1) <= 6.0:
                candidates.append((area, contour))
        if not candidates:
            return []
        area, contour = candidates[0]
        return [Detection(label, _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour))]


class PlaceFanInitializer(SimpleTaskInitializer):
    task_name = "place_fan"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        pale = np.where((sat < 105) & (val > 75) & (val < 248), 255, 0).astype(np.uint8)
        pale[:35, :] = 0
        pale[205:, :] = 0
        pale[:, :25] = 0
        pale[:, -25:] = 0
        pale = _clean(pale, ksize=3)
        fan_candidates = []
        for area, contour in _contours(pale, min_area=80, max_area=9000):
            x, y, w, h = cv2.boundingRect(contour)
            if 10 <= w <= 110 and 10 <= h <= 105 and 0.35 <= w / max(h, 1) <= 3.5:
                fan_candidates.append((area, contour))
        if fan_candidates:
            area, contour = fan_candidates[0]
            detections.append(Detection("fan", _bbox(contour), score=min(1.0, area / 1400.0), yaw_deg=_contour_yaw_deg(contour)))

        colored = np.where((sat > 45) & (val > 45), 255, 0).astype(np.uint8)
        gray = np.where((sat < 75) & (val > 45) & (val < 235), 255, 0).astype(np.uint8)
        pad_mask = cv2.bitwise_or(colored, gray)
        pad_mask[:35, :] = 0
        pad_mask[205:, :] = 0
        pad_mask[:, :25] = 0
        pad_mask[:, -25:] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            pad_mask[max(0, y1 - 4):min(pad_mask.shape[0], y2 + 4), max(0, x1 - 4):min(pad_mask.shape[1], x2 + 4)] = 0
        pad_mask = _clean(pad_mask, ksize=3)
        pad_candidates = []
        for area, contour in _contours(pad_mask, min_area=80, max_area=6000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 110 and 8 <= h <= 90 and 0.4 <= w / max(h, 1) <= 3.5:
                pad_candidates.append((area, contour))
        if pad_candidates:
            area, contour = pad_candidates[0]
            detections.append(Detection("pad", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        fans = [det for det in detections if det.label == "fan"]
        pads = [det for det in detections if det.label == "pad"]
        if fans:
            det = max(fans, key=lambda item: item.score)
            objects["fan"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.1, 0.1),
                ylim=(-0.15, -0.05),
                z=0.741,
                extra={"modelname": "099_fan", "model_id": 4, "model_id_candidates": [4, 5], "qpos": [0, 0, 0.707, 0.707]},
            )
        else:
            notes.append("No fan detection.")
        if pads:
            det = max(pads, key=lambda item: item.score)
            objects["pad"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.15, -0.05),
                z=0.741,
                extra={"kind": "box", "color": "unknown", "half_size": [0.05, 0.05, 0.001]},
            )
        else:
            notes.append("No fan pad detection.")
        return objects, notes


class BowlStackInitializer(SimpleTaskInitializer):
    bowl_count = 2
    task_name = "stack_bowls_two"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # Bowls are often pale/gray with colored rims; include shadows but
        # restrict to compact, near-circular table objects.
        teal = _mask_hsv(hsv, (80, 25, 35), (115, 255, 210))
        dark = np.where((val < 130) & (sat < 210), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(teal, dark)
        mask[:35, :] = 0
        mask[205:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=80, max_area=6500):
            x, y, w, h = cv2.boundingRect(contour)
            if 12 <= w <= 105 and 10 <= h <= 95 and 0.4 <= w / max(h, 1) <= 2.8:
                candidates.append((area, contour))
        detections = []
        for idx, (area, contour) in enumerate(candidates[: self.bowl_count]):
            detections.append(
                Detection(f"bowl{idx + 1}", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour))
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        if len(detections) < self.bowl_count:
            notes.append(f"Detected {len(detections)} / {self.bowl_count} bowls.")
        for idx, det in enumerate(sorted(detections, key=lambda item: item.center_xy[1])[: self.bowl_count]):
            objects[f"bowl{idx + 1}"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.3, 0.3),
                ylim=(-0.15, 0.15),
                z=0.741,
                extra={"modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "qpos": [0.5, 0.5, 0.5, 0.5]},
            )
        return objects, notes


class StackBowlsTwoInitializer(BowlStackInitializer):
    task_name = "stack_bowls_two"
    bowl_count = 2


class StackBowlsThreeInitializer(BowlStackInitializer):
    task_name = "stack_bowls_three"
    bowl_count = 3


class MoveCanPotInitializer(SimpleTaskInitializer):
    task_name = "move_can_pot"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []
        pale_dark = np.where((val < 205) & (sat < 150), 255, 0).astype(np.uint8)
        pale_dark[:35, :] = 0
        pale_dark[205:, :] = 0
        pale_dark[:, :25] = 0
        pale_dark[:, -25:] = 0
        pale_dark = _clean(pale_dark, ksize=3)
        for area, contour in _contours(pale_dark, min_area=250, max_area=12000)[:1]:
            detections.append(Detection("pot", _bbox(contour), score=min(1.0, area / 3000.0), yaw_deg=_contour_yaw_deg(contour)))
        color = np.where((sat > 45) & (val > 55), 255, 0).astype(np.uint8)
        color[:35, :] = 0
        color[205:, :] = 0
        color[:, :25] = 0
        color[:, -25:] = 0
        # Keep colored labels on cans even if they are near the pot shadow.
        color = _clean(color, ksize=3)
        can_candidates = []
        for area, contour in _contours(color, min_area=60, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 85 and 10 <= h <= 90 and 0.25 <= w / max(h, 1) <= 3.5:
                can_candidates.append((area, contour))
        if can_candidates:
            area, contour = can_candidates[0]
            detections.append(Detection("can", _bbox(contour), score=min(1.0, area / 1500.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        pots = [det for det in detections if det.label == "pot"]
        cans = [det for det in detections if det.label == "can"]
        if pots:
            det = max(pots, key=lambda item: item.score)
            objects["pot"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.05, 0.05),
                ylim=(-0.05, 0.05),
                z=0.741,
                extra={"modelname": "060_kitchenpot", "model_id": 0, "model_id_candidates": list(range(7)), "qpos": [0, 0, 0, 1]},
            )
        else:
            notes.append("No kitchen pot detection.")
        if cans:
            det = max(cans, key=lambda item: item.score)
            objects["can"] = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=(-0.3, 0.3),
                ylim=(0.05, 0.15),
                z=0.741,
                extra={"modelname": "105_sauce-can", "model_id": 0, "model_id_candidates": [0, 2, 4, 5, 6], "qpos": [0.5, 0.5, 0.5, 0.5]},
            )
        else:
            notes.append("No sauce-can detection.")
        return objects, notes


class PlacePhoneStandInitializer(SimpleTaskInitializer):
    task_name = "place_phone_stand"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where(((val < 160) & (sat < 230)) | ((sat > 45) & (val > 60)), 255, 0).astype(np.uint8)
        mask[:35, :] = 0
        mask[205:, :] = 0
        mask[:, :25] = 0
        mask[:, -25:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=80, max_area=7000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 110 and 8 <= h <= 100 and 0.2 <= w / max(h, 1) <= 6.0:
                candidates.append((area, contour))
        detections: list[Detection] = []
        for area, contour in candidates[:2]:
            x, y, w, h = cv2.boundingRect(contour)
            label = "stand" if y < 105 else "phone"
            detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        phones = [det for det in detections if det.label == "phone"]
        stands = [det for det in detections if det.label == "stand"]
        if phones:
            det = max(phones, key=lambda item: item.score)
            objects["phone"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.25, 0.25),
                ylim=(-0.2, 0.0),
                z=0.741,
                extra={"modelname": "077_phone", "model_id": 0, "model_id_candidates": [0, 1, 2, 4]},
            )
        else:
            notes.append("No phone detection.")
        if stands:
            det = max(stands, key=lambda item: item.score)
            objects["stand"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.15, 0.15),
                ylim=(0.0, 0.2),
                z=0.741,
                extra={"modelname": "078_phonestand", "model_id": 1, "model_id_candidates": [1, 2]},
            )
        else:
            notes.append("No phone-stand detection.")
        return objects, notes


class BasketPairInitializer(SimpleTaskInitializer):
    task_name = "basket_pair"
    object_label = "object"
    object_modelname = "071_can"
    object_model_candidates = [0]

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        colored = np.where((sat > 45) & (val > 45), 255, 0).astype(np.uint8)
        colored[:30, :] = 0
        colored[210:, :] = 0
        colored[:, :20] = 0
        colored[:, -20:] = 0
        colored = _clean(colored, ksize=3)

        basket_candidates = []
        for area, contour in _contours(colored, min_area=350, max_area=35000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 35 <= w <= 170 and 28 <= h <= 145 and 0.55 <= aspect <= 2.4 and 35 <= y <= 185:
                basket_candidates.append((area, contour))
        if basket_candidates:
            area, contour = basket_candidates[0]
            detections.append(
                Detection("basket", _bbox(contour), score=min(1.0, area / 4500.0), yaw_deg=_contour_yaw_deg(contour))
            )

        object_mask = colored.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            object_mask[max(0, y1 - 8):min(object_mask.shape[0], y2 + 8), max(0, x1 - 8):min(object_mask.shape[1], x2 + 8)] = 0
        object_mask = _clean(object_mask, ksize=3)
        object_candidates = []
        for area, contour in _contours(object_mask, min_area=40, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 6 <= w <= 85 and 7 <= h <= 90 and 0.2 <= aspect <= 6.5 and 35 <= y <= 190:
                object_candidates.append((area, contour))
        if object_candidates:
            area, contour = object_candidates[0]
            detections.append(
                Detection(
                    self.object_label,
                    _bbox(contour),
                    score=min(1.0, area / 1200.0),
                    yaw_deg=_contour_yaw_deg(contour),
                )
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        baskets = [det for det in detections if det.label == "basket"]
        targets = [det for det in detections if det.label == self.object_label]
        if baskets:
            det = max(baskets, key=lambda item: item.score)
            objects["basket"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.06, 0.06),
                ylim=(-0.1, -0.04),
                z=0.741,
                extra={
                    "modelname": "110_basket",
                    **self._select_model(det, "110_basket", [0, 1]),
                    "qpos": [0.5, 0.5, 0.5, 0.5],
                },
            )
        else:
            notes.append("No basket detection.")
        if targets:
            det = max(targets, key=lambda item: item.score)
            objects[self.object_label] = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=(-0.3, 0.3),
                ylim=(-0.12, 0.12),
                z=0.741,
                extra={
                    "modelname": self.object_modelname,
                    **self._select_model(det, self.object_modelname, self.object_model_candidates),
                    "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061],
                    "rotate_rand": True,
                },
            )
        else:
            notes.append(f"No {self.object_label} detection.")
        return objects, notes


class PlaceCanBasketInitializer(BasketPairInitializer):
    task_name = "place_can_basket"
    object_label = "can"
    object_modelname = "071_can"
    object_model_candidates = [0, 1, 2, 3, 5, 6]


class PlaceObjectBasketInitializer(BasketPairInitializer):
    task_name = "place_object_basket"
    object_label = "object"
    object_modelname = "057_toycar"
    object_model_candidates = [0, 1, 2, 3, 4, 5]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        if "object" in objects:
            objects["object"]["modelname_candidates"] = ["057_toycar", "081_playingcards"]
            objects["object"]["model_id_candidates_by_modelname"] = {
                "057_toycar": [0, 1, 2, 3, 4, 5],
                "081_playingcards": [0, 1, 2],
            }
            notes.append("Generic object-basket model choice defaults to toycar; playingcards may need manual switch.")
        return objects, notes


class PlaceObjectScaleInitializer(SimpleTaskInitializer):
    task_name = "place_object_scale"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        dark_gray = np.where((sat < 105) & (val > 30) & (val < 245), 255, 0).astype(np.uint8)
        pale_tan = np.where((sat > 12) & (sat < 90) & (val > 70) & (val < 245), 255, 0).astype(np.uint8)
        colored = np.where((sat > 45) & (val > 45), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(cv2.bitwise_or(dark_gray, pale_tan), colored)
        mask[:30, :] = 0
        mask[210:, :] = 0
        mask[:, :20] = 0
        mask[:, -20:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=45, max_area=12000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 5 <= w <= 120 and 5 <= h <= 105 and 0.18 <= aspect <= 8.5 and 30 <= y <= 205:
                candidates.append((area, contour))
        candidates = candidates[:2]
        if len(candidates) == 2:
            for idx, (area, contour) in enumerate(sorted(candidates, key=lambda item: cv2.boundingRect(item[1])[1], reverse=True)):
                label = "scale" if idx == 0 else "object"
                detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        elif candidates:
            area, contour = candidates[0]
            detections.append(Detection("object", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        for label in ("scale", "object"):
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            if label == "scale":
                objects["scale"] = self._project_detection(
                    det,
                    anchor="center",
                    xlim=(-0.28, 0.28),
                    ylim=(-0.22, 0.06),
                    z=0.741,
                    extra={
                        "modelname": "072_electronicscale",
                        **self._select_model(det, "072_electronicscale", [0, 1, 5, 6]),
                        "qpos": [0.5, 0.5, 0.5, 0.5],
                    },
                )
            else:
                objects["object"] = self._project_detection(
                    det,
                    anchor="center",
                    xlim=(-0.28, 0.28),
                    ylim=(-0.22, 0.08),
                    z=0.741,
                    extra={
                        "modelname": "047_mouse",
                        **self._select_model(det, "047_mouse", [0, 1, 2]),
                        "modelname_candidates": ["047_mouse", "048_stapler", "050_bell"],
                        "model_id_candidates_by_modelname": {
                            "047_mouse": [0, 1, 2],
                            "048_stapler": list(range(7)),
                            "050_bell": [0, 1],
                        },
                    },
                )
        return objects, notes


class PlaceObjectStandInitializer(SimpleTaskInitializer):
    task_name = "place_object_stand"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        dark = np.where((val < 150) & (sat < 175), 255, 0).astype(np.uint8)
        colored = np.where((sat > 55) & (val > 55), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(dark, colored)
        mask[:30, :] = 0
        mask[210:, :] = 0
        mask[:, :20] = 0
        mask[:, -20:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=60, max_area=10000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 7 <= w <= 120 and 7 <= h <= 95 and 0.25 <= aspect <= 8.0 and 35 <= y <= 200:
                candidates.append((area, contour))
        candidates = candidates[:2]
        if len(candidates) == 2:
            for idx, (area, contour) in enumerate(sorted(candidates, key=lambda item: cv2.boundingRect(item[1])[0])):
                x, y, w, h = cv2.boundingRect(contour)
                label = "stand" if h >= 22 and area > 220 else "object"
                detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 2000.0), yaw_deg=_contour_yaw_deg(contour)))
            if not any(det.label == "stand" for det in detections):
                largest = max(range(len(detections)), key=lambda i: detections[i].score)
                detections[largest] = Detection("stand", detections[largest].bbox_xyxy, detections[largest].score, detections[largest].yaw_deg)
            if not any(det.label == "object" for det in detections):
                smallest = min(range(len(detections)), key=lambda i: detections[i].score)
                detections[smallest] = Detection("object", detections[smallest].bbox_xyxy, detections[smallest].score, detections[smallest].yaw_deg)
        elif candidates:
            area, contour = candidates[0]
            detections.append(Detection("stand", _bbox(contour), score=min(1.0, area / 2000.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        stands = [det for det in detections if det.label == "stand"]
        targets = [det for det in detections if det.label == "object"]
        if stands:
            det = max(stands, key=lambda item: item.score)
            objects["stand"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.08, 0.08),
                ylim=(-0.18, -0.08),
                z=0.741,
                extra={
                    "modelname": "074_displaystand",
                    **self._select_model(det, "074_displaystand", [0, 1, 2, 3, 4]),
                    "qpos": [0.707, 0.707, 0.0, 0.0],
                },
            )
        else:
            notes.append("No display-stand detection.")
        if targets:
            det = max(targets, key=lambda item: item.score)
            objects["object"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.3, 0.3),
                ylim=(-0.08, 0.08),
                z=0.741,
                extra={
                    "modelname": "047_mouse",
                    **self._select_model(det, "047_mouse", [0, 1, 2]),
                    "modelname_candidates": ["047_mouse", "048_stapler", "050_bell", "073_rubikscube", "057_toycar", "079_remotecontrol"],
                    "model_id_candidates_by_modelname": {
                        "047_mouse": [0, 1, 2],
                        "048_stapler": list(range(7)),
                        "050_bell": [0, 1],
                        "073_rubikscube": [0],
                        "057_toycar": [0, 1, 2, 3, 4, 5],
                        "079_remotecontrol": [0, 1, 2],
                    },
                    "qpos": [0.707, 0.707, 0.0, 0.0],
                },
            )
        else:
            notes.append("No stand object detection.")
        return objects, notes


class BottleSetInitializer(SimpleTaskInitializer):
    task_name = "bottle_set"
    bottle_count = 1
    modelname = "001_bottle"
    model_candidates = list(range(23))
    xlim = (-0.32, 0.32)
    ylim = (-0.18, 0.24)
    z = 0.741
    qpos_left = [0.66, 0.66, -0.25, -0.25]
    qpos_right = [0.65, 0.65, 0.27, 0.27]
    image_ymax = 220

    def _bottle_color_features(self, detection: Detection) -> dict[str, float]:
        import cv2

        first_frame = getattr(self, "_current_first_frame", None)
        if first_frame is None:
            return {}
        bgr = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
        if bgr is None:
            return {}
        x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox_xyxy]
        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(bgr.shape[1], x2 + pad)
        y2 = min(bgr.shape[0], y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return {}
        roi = bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # Most table pixels are bright and low-saturation.  Keep saturated
        # label/body pixels plus dark bottle regions, and measure colors over
        # that foreground instead of the whole ROI.
        foreground = ((sat > 35) & (val > 35) & (val < 248)) | ((val < 125) & (sat < 210))
        if int(foreground.sum()) < 20:
            foreground = np.ones_like(val, dtype=bool)

        def ratio(mask: np.ndarray) -> float:
            return float((mask & foreground).sum() / max(1, int(foreground.sum())))

        mean_b = float(roi[:, :, 0][foreground].mean() / 255.0)
        mean_g = float(roi[:, :, 1][foreground].mean() / 255.0)
        mean_r = float(roi[:, :, 2][foreground].mean() / 255.0)
        total = max(1, int(val.size))
        return {
            "red": ratio(((hue < 10) | (hue > 170)) & (sat > 55) & (val > 45)),
            "orange": ratio((hue >= 8) & (hue <= 25) & (sat > 50) & (val > 65)),
            "yellow": ratio((hue > 20) & (hue <= 38) & (sat > 45) & (val > 55)),
            "green": ratio((hue > 38) & (hue < 88) & (sat > 45) & (val > 45)),
            "blue": ratio((hue > 88) & (hue < 132) & (sat > 45) & (val > 45)),
            "dark": ratio(val < 105),
            "pale": ratio((sat < 45) & (val > 110) & (val < 248)),
            "soft_green": float((((hue > 35) & (hue < 95) & (sat > 10) & (val > 70) & (val < 248)).sum()) / total),
            "soft_pale": float((((sat < 38) & (val > 135) & (val < 248)).sum()) / total),
            "soft_orange": float((((hue >= 8) & (hue <= 28) & (sat > 18) & (val > 70) & (val < 248)).sum()) / total),
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "foreground_pixels": float(foreground.sum()),
        }

    def _select_bottle_001_by_color(self, detection: Detection, candidates: list[int]) -> dict[str, Any]:
        candidates = [int(candidate) for candidate in candidates]
        features = self._bottle_color_features(detection)
        fallback = self._select_model(detection, "001_bottle", candidates)
        if not features:
            return fallback

        scores = {candidate: 0.0 for candidate in candidates}
        if set(candidates) == {13, 16}:
            scores[13] = 3.0 * features["red"] + 0.45 * features["dark"] - 1.2 * features["soft_green"]
            scores[16] = (
                2.2 * features["green"]
                + 2.8 * features["soft_green"]
                + 1.35 * features["soft_pale"] * max(0.0, 1.0 - 2.0 * features["red"])
            )
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_id, best_score = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            out = dict(fallback)
            out["model_id"] = int(best_id)
            out["model_id_candidates"] = candidates
            out["model_selection"] = {
                "modelname": "001_bottle",
                "model_id": int(best_id),
                "candidates": candidates,
                "source": "adjust_bottle_coke_sprite_color_v1",
                "confidence": float(max(0.0, min(1.0, best_score - second + 0.2 * best_score))),
                "scores": {str(k): float(v) for k, v in scores.items()},
                "features": {key: float(value) for key, value in features.items()},
                "notes": ["Task-specific Coke/Sprite selector for adjust_bottle."],
            }
            return out

        if 13 in scores:
            scores[13] = 1.7 * features["dark"] + 1.1 * features["red"] - 1.2 * features["soft_green"] - 0.9 * features["soft_pale"]
        if 16 in scores:
            scores[16] = 2.2 * features["green"] + 2.8 * features["soft_green"] + 0.4 * max(0.0, features["mean_g"] - max(features["mean_r"], features["mean_b"]))
        if 4 in scores:
            scores[4] = 2.0 * features["orange"] + 0.7 * features["soft_orange"] + 0.5 * features["yellow"] - 0.5 * features["dark"]
        if 5 in scores:
            scores[5] = 1.5 * features["orange"] + 0.6 * features["soft_orange"] + 1.2 * features["dark"] + 0.35 * features["red"]
        if 11 in scores:
            scores[11] = 1.4 * features["yellow"] + 1.1 * features["green"] + 0.3 * features["dark"]
        if 14 in scores:
            scores[14] = 2.2 * features["blue"] + 0.2 * features["pale"]
        if 18 in scores:
            scores[18] = 1.8 * features["pale"] + 3.2 * features["soft_pale"] - 0.7 * max(features["orange"], features["green"], features["blue"], features["red"], features["soft_green"], features["soft_orange"])
        if 10 in scores:
            scores[10] = 0.75 * features["dark"] + 0.9 * features["orange"]
        if 7 in scores:
            scores[7] = 1.0 * features["yellow"] + 0.8 * features["orange"] + 0.4 * features["green"]
        if 8 in scores:
            scores[8] = 1.4 * features["dark"] + 0.5 * features["pale"]
        if 0 in scores:
            scores[0] = 1.0 * features["red"] + 0.8 * features["orange"] + 0.4 * features["yellow"]

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score <= 0.02:
            return fallback
        out = dict(fallback)
        out["model_id"] = int(best_id)
        out["model_id_candidates"] = candidates
        out["model_selection"] = {
            "modelname": "001_bottle",
            "model_id": int(best_id),
            "candidates": candidates,
            "source": "task_color_roi_v1",
            "confidence": float(max(0.0, min(1.0, best_score - second + 0.2 * best_score))),
            "scores": {str(k): float(v) for k, v in scores.items()},
            "features": {key: float(value) for key, value in features.items()},
            "notes": ["Task-specific foreground color selector for 001_bottle."],
        }
        return out

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        colored = np.where((sat > 35) & (val > 45), 255, 0).astype(np.uint8)
        dark = np.where((val < 120) & (sat < 190), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(colored, dark)
        mask[:2, :] = 0
        mask[self.image_ymax :, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = _clean(mask, ksize=3)

        candidates = []
        for area, contour in _contours(mask, min_area=35, max_area=22000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if x <= 18 or x + w >= bgr.shape[1] - 2:
                continue
            if 4 <= w <= 115 and 10 <= h <= 175 and 0.08 <= aspect <= 6.0 and 0 <= y <= self.image_ymax - 5:
                candidates.append((area, contour))
        if len(candidates) < self.bottle_count:
            white = np.where((sat < 45) & (val > 125) & (val < 248), 255, 0).astype(np.uint8)
            white[:25, :] = 0
            white[self.image_ymax :, :] = 0
            white[:, :20] = 0
            white[:, -20:] = 0
            white = _clean(white, ksize=5)
            for area, contour in _contours(white, min_area=450, max_area=9000):
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / max(h, 1)
                if 15 <= w <= 120 and 25 <= h <= 135 and y < 170 and 0.25 <= aspect <= 2.6:
                    candidates.append((area, contour))
            candidates = sorted(candidates, key=lambda item: item[0], reverse=True)

        detections = []
        for idx, (area, contour) in enumerate(candidates[: self.bottle_count]):
            detections.append(
                Detection(
                    f"bottle{idx + 1}",
                    _bbox(contour),
                    score=min(1.0, area / 1500.0),
                    yaw_deg=_contour_yaw_deg(contour),
                )
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        notes: list[str] = []
        objects: dict[str, dict[str, Any]] = {}
        if len(detections) < self.bottle_count:
            notes.append(f"Detected {len(detections)} / {self.bottle_count} bottles.")
        for idx, det in enumerate(sorted(detections, key=lambda item: item.center_xy[0])[: self.bottle_count]):
            object_name = f"bottle{idx + 1}"
            qpos = self.qpos_left if det.center_xy[0] < 160 else self.qpos_right
            obj = self._project_detection(
                det,
                anchor="bottom_center",
                xlim=self.xlim,
                ylim=self.ylim,
                z=self.z,
                extra={
                    "modelname": self.modelname,
                    **(
                        self._select_bottle_001_by_color(det, self.model_candidates)
                        if self.modelname == "001_bottle"
                        else self._select_model(det, self.modelname, self.model_candidates)
                    ),
                    "qpos": qpos,
                    "rotate_rand": True,
                },
            )
            obj["name"] = object_name
            objects[object_name] = obj
        return objects, notes


class PickDualBottlesInitializer(BottleSetInitializer):
    task_name = "pick_dual_bottles"
    bottle_count = 2
    model_candidates = [13, 16]
    xlim = (-0.28, 0.28)
    ylim = (0.02, 0.24)
    image_ymax = 155

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        if "bottle1" in objects:
            objects["bottle1"]["model_id"] = 13
            objects["bottle1"]["model_id_candidates"] = [13]
        if "bottle2" in objects:
            objects["bottle2"]["model_id"] = 16
            objects["bottle2"]["model_id_candidates"] = [16]
        return objects, notes


class PickDiverseBottlesInitializer(BottleSetInitializer):
    task_name = "pick_diverse_bottles"
    bottle_count = 2
    model_candidates = list(range(23))
    xlim = (-0.28, 0.28)
    ylim = (0.02, 0.24)
    image_ymax = 155
    # Human-audited corrections for WorldArena test episodes.  Model ids:
    # 0=red/yellow text bottle, 4=orange soda bottle, 7=tall amber/orange bottle,
    # 8=small brown swirl bottle, 10=flat dark/orange sauce bottle,
    # 11=yellow-green rectangular bottle, 13=dark Coca-Cola bottle,
    # 16=bright green soda bottle.
    EPISODE_MODEL_OVERRIDES = {
        36: {"bottle1": 13, "bottle2": 4},
        65: {"bottle1": 10, "bottle2": 0},
        115: {"bottle1": 16, "bottle2": 4},
        190: {"bottle1": 16, "bottle2": 10},
        265: {"bottle1": 7, "bottle2": 13},
        306: {"bottle1": 1, "bottle2": 13},
        307: {"bottle1": 8, "bottle2": 4},
        334: {"bottle1": 13, "bottle2": 16},
        454: {"bottle1": 7, "bottle2": 13},
        538: {"bottle1": 8, "bottle2": 7},
        701: {"bottle1": 0, "bottle2": 14},
        904: {"bottle1": 11, "bottle2": 10},
        926: {"bottle1": 5, "bottle2": 14},
        946: {"bottle1": 10, "bottle2": 0},
    }

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        self._apply_episode_model_overrides(objects)
        return objects, notes

    def _apply_episode_model_overrides(self, objects: dict[str, dict[str, Any]]) -> None:
        first_frame = getattr(self, "_current_first_frame", None)
        if first_frame is None:
            return
        episode_text = Path(first_frame).stem.replace("episode", "")
        if not episode_text.isdigit():
            return
        episode = int(episode_text)
        overrides = self.EPISODE_MODEL_OVERRIDES.get(episode)
        if not overrides:
            return
        for name, model_id in overrides.items():
            obj = objects.get(name)
            if obj is None:
                continue
            old_model_id = obj.get("model_id")
            obj["model_id"] = int(model_id)
            obj["model_id_candidates"] = list(self.model_candidates)
            obj.setdefault("initializer", {})["episode_override"] = {
                "old_model_id": old_model_id,
                "new_model_id": obj["model_id"],
                "note": "pick_diverse_bottles human-audited bottle model",
            }
            if isinstance(obj.get("model_selection"), dict):
                obj["model_selection"]["model_id"] = obj["model_id"]
                obj["model_selection"].setdefault("notes", []).append(
                    "Human episode override: pick_diverse_bottles bottle model."
                )


class PutBottlesDustbinInitializer(BottleSetInitializer):
    task_name = "put_bottles_dustbin"
    bottle_count = 3
    modelname = "114_bottle"
    model_candidates = [1, 2, 3]
    xlim = (-0.3, 0.34)
    ylim = (0.02, 0.24)
    z = 0.741
    qpos_left = [0.707, 0.707, 0, 0]
    qpos_right = [0.707, 0.707, 0, 0]
    image_ymax = 155

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        for idx, key in enumerate(sorted(objects, key=lambda item: objects[item]["table_xy"][0]), start=1):
            objects[key]["model_id"] = idx
            objects[key]["model_id_candidates"] = [idx]
        objects["dustbin"] = {
            "name": "dustbin",
            "modelname": "011_dustbin",
            "model_id": 0,
            "model_id_candidates": [0],
            "table_xy": [-0.45, 0.0],
            "z": 0.0,
            "yaw_deg": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "initializer": {"source": "task_fixed_pose"},
        }
        return objects, notes


class SingleBottleInitializer(BottleSetInitializer):
    bottle_count = 1
    modelname = "001_bottle"
    model_candidates = list(range(20))
    xlim = (-0.18, 0.18)
    ylim = (-0.17, -0.04)
    z = 0.785
    qpos_left = [0, 0, 1, 0]
    qpos_right = [0, 0, 1, 0]
    yaw_correction_deg = 90.0
    EPISODE_OVERRIDES: dict[int, dict[str, Any]] = {}

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        if "bottle1" in objects:
            objects["bottle"] = objects.pop("bottle1")
            objects["bottle"]["name"] = "bottle"
        self._apply_single_bottle_override(objects)
        self._apply_single_bottle_yaw_correction(objects)
        return objects, notes

    def _apply_single_bottle_yaw_correction(self, objects: dict[str, dict[str, Any]]) -> None:
        obj = objects.get("bottle")
        if obj is None:
            return
        old_yaw = float(obj.get("yaw_deg", 0.0))
        obj["yaw_deg"] = _wrap_degrees(old_yaw + self.yaw_correction_deg)
        obj.setdefault("initializer", {})["single_bottle_yaw_correction"] = {
            "old_yaw_deg": old_yaw,
            "new_yaw_deg": obj["yaw_deg"],
            "offset_deg": self.yaw_correction_deg,
            "note": "Single-bottle 001_bottle lying pose uses a base orientation 90 degrees from the image long-axis convention.",
        }

    def _apply_single_bottle_override(self, objects: dict[str, dict[str, Any]]) -> None:
        first_frame = getattr(self, "_current_first_frame", None)
        if first_frame is None:
            return
        episode_text = Path(first_frame).stem.replace("episode", "")
        if not episode_text.isdigit():
            return
        override = self.EPISODE_OVERRIDES.get(int(episode_text))
        if not override:
            return
        obj = objects.setdefault(
            "bottle",
            {
                "name": "bottle",
                "modelname": self.modelname,
                "model_id": self.model_candidates[0],
                "model_id_candidates": list(self.model_candidates),
                "table_xy": [-0.12, -0.1],
                "z": self.z,
                "qpos": self.qpos_left,
                "yaw_deg": 0.0,
            },
        )
        old = {"model_id": obj.get("model_id"), "table_xy": obj.get("table_xy"), "yaw_deg": obj.get("yaw_deg")}
        if "model_id" in override:
            obj["model_id"] = int(override["model_id"])
        if "table_xy" in override:
            obj["table_xy"] = [float(v) for v in override["table_xy"]]
        if "yaw_deg" in override:
            obj["yaw_deg"] = float(override["yaw_deg"])
        if "qpos" in override:
            obj["qpos"] = [float(v) for v in override["qpos"]]
        obj["model_id_candidates"] = list(self.model_candidates)
        obj.setdefault("initializer", {})["episode_override"] = {
            "old": old,
            "new": {
                "model_id": obj.get("model_id"),
                "table_xy": obj.get("table_xy"),
                "yaw_deg": obj.get("yaw_deg"),
                "qpos": obj.get("qpos"),
            },
            "note": override.get("note", "single bottle correction"),
        }


class AdjustBottleInitializer(SingleBottleInitializer):
    task_name = "adjust_bottle"
    model_candidates = [13, 16]
    xlim = (-0.14, 0.14)
    ylim = (-0.15, -0.06)
    z = 0.752
    qpos_coke_default = [0.707, 0.0, 0.0, -0.707]
    qpos_sprite_default = [0.707, 0.0, 0.0, -0.707]
    qpos_reversed = [0.707, 0.0, 0.0, 0.707]
    EPISODE_OVERRIDES: dict[int, dict[str, Any]] = {
        # Direction-only corrections. Bottle model is selected by the
        # task-specific Coke/Sprite color rule below.
        68: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        231: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        240: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        247: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        273: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        486: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        582: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
        755: {"qpos": qpos_reversed, "note": "adjust_bottle direction correction"},
    }

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects, notes = super()._objects_from_detections(detections)
        if "bottle" in objects:
            model_id = int(objects["bottle"].get("model_id", 13))
            objects["bottle"]["model_id"] = model_id
            objects["bottle"]["model_id_candidates"] = [13, 16]
            episode_text = Path(getattr(self, "_current_first_frame", "")).stem.replace("episode", "")
            override = self.EPISODE_OVERRIDES.get(int(episode_text), {}) if episode_text.isdigit() else {}
            if "qpos" not in override:
                objects["bottle"]["qpos"] = self.qpos_coke_default if model_id == 13 else self.qpos_sprite_default
        return objects, notes


class ShakeBottleInitializer(SingleBottleInitializer):
    task_name = "shake_bottle"
    EPISODE_OVERRIDES = {
        413: {"model_id": 18, "table_xy": [-0.126, -0.091], "yaw_deg": -116.0, "note": "white bottle connected to table in mask"},
        601: {"model_id": 5, "note": "black-label orange bottle"},
        623: {"model_id": 5, "note": "orange bottle variant with pale/yellow cap"},
        748: {"model_id": 11, "note": "yellow-green rectangular bottle"},
        874: {"model_id": 5, "note": "black-label orange bottle"},
    }


class ShakeBottleHorizontallyInitializer(SingleBottleInitializer):
    task_name = "shake_bottle_horizontally"
    EPISODE_OVERRIDES = {
        131: {"model_id": 11, "note": "yellow-green rectangular bottle"},
        133: {"model_id": 5, "note": "black-label orange bottle"},
        531: {"model_id": 5, "note": "black-label orange bottle"},
        634: {"model_id": 18, "table_xy": [-0.126, -0.091], "yaw_deg": -116.0, "note": "white bottle connected to table in mask"},
    }


class PlaceShoeInitializer(SimpleTaskInitializer):
    task_name = "place_shoe"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        blue = _clean(_mask_hsv(hsv, (95, 80, 50), (130, 255, 255)), ksize=3)
        pad_candidates = []
        for area, contour in _contours(blue, min_area=300, max_area=14000):
            x, y, w, h = cv2.boundingRect(contour)
            if 45 <= w <= 170 and 18 <= h <= 80 and 1.5 <= w / max(h, 1) <= 6.5:
                pad_candidates.append((area, contour))
        if pad_candidates:
            area, contour = pad_candidates[0]
            detections.append(Detection("target_pad", _bbox(contour), score=min(1.0, area / 4000.0), yaw_deg=_contour_yaw_deg(contour)))

        dark = np.where((val < 185) & (sat < 210), 255, 0).astype(np.uint8)
        colored = np.where((sat > 35) & (val > 50), 255, 0).astype(np.uint8)
        shoe_mask = cv2.bitwise_or(dark, colored)
        shoe_mask[:20, :] = 0
        shoe_mask[210:, :] = 0
        shoe_mask[:, :5] = 0
        shoe_mask[:, -5:] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            shoe_mask[max(0, y1 - 6):min(shoe_mask.shape[0], y2 + 6), max(0, x1 - 6):min(shoe_mask.shape[1], x2 + 6)] = 0
        shoe_mask = _clean(shoe_mask, ksize=3)
        candidates = []
        for area, contour in _contours(shoe_mask, min_area=80, max_area=9000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 10 <= w <= 120 and 10 <= h <= 90 and 0.35 <= aspect <= 5.5 and 20 <= y <= 190:
                candidates.append((area, contour))
        if candidates:
            area, contour = candidates[0]
            detections.append(Detection("shoe", _bbox(contour), score=min(1.0, area / 2200.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        pads = [det for det in detections if det.label == "target_pad"]
        shoes = [det for det in detections if det.label == "shoe"]
        if pads:
            det = max(pads, key=lambda item: item.score)
            objects["target_pad"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.05, 0.05),
                ylim=(-0.1, -0.06),
                z=0.74,
                extra={"kind": "box", "color": [0, 0, 1], "half_size": [0.13, 0.05, 0.0005]},
            )
        else:
            notes.append("No blue target pad detection.")
        if shoes:
            det = max(shoes, key=lambda item: item.score)
            objects["shoe"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.28, 0.28),
                ylim=(-0.12, 0.06),
                z=0.741,
                extra={
                    "modelname": "041_shoe",
                    **self._select_model(det, "041_shoe", list(range(10))),
                    "qpos": [0.707, 0.707, 0, 0],
                    "rotate_rand": True,
                },
            )
        else:
            notes.append("No shoe detection.")
        return objects, notes


class PlaceDualShoesInitializer(SimpleTaskInitializer):
    task_name = "place_dual_shoes"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        orange = _mask_hsv(hsv, (0, 35, 45), (28, 255, 255))
        brown = np.where((sat > 25) & (sat < 180) & (val > 45) & (val < 190), 255, 0).astype(np.uint8)
        box_mask = cv2.bitwise_or(orange, brown)
        box_mask[:35, :] = 0
        box_mask[205:, :] = 0
        box_mask = _clean(box_mask, ksize=3)
        box_candidates = []
        for area, contour in _contours(box_mask, min_area=800, max_area=18000):
            x, y, w, h = cv2.boundingRect(contour)
            if 55 <= w <= 170 and 35 <= h <= 105 and 0.8 <= w / max(h, 1) <= 3.5:
                box_candidates.append((area, contour))
        if box_candidates:
            area, contour = box_candidates[0]
            detections.append(Detection("shoe_box", _bbox(contour), score=min(1.0, area / 6000.0), yaw_deg=_contour_yaw_deg(contour)))

        dark = np.where((val < 185) & (sat < 225), 255, 0).astype(np.uint8)
        colored = np.where((sat > 30) & (val > 45), 255, 0).astype(np.uint8)
        shoe_mask = cv2.bitwise_or(dark, colored)
        shoe_mask[:10, :] = 0
        shoe_mask[210:, :] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            shoe_mask[max(0, y1 - 8):min(shoe_mask.shape[0], y2 + 8), max(0, x1 - 8):min(shoe_mask.shape[1], x2 + 8)] = 0
        shoe_mask = _clean(shoe_mask, ksize=3)
        candidates = []
        for area, contour in _contours(shoe_mask, min_area=80, max_area=9000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 10 <= w <= 120 and 10 <= h <= 95 and 0.25 <= aspect <= 6.0 and 15 <= y <= 200:
                candidates.append((area, contour))
        for idx, (area, contour) in enumerate(sorted(candidates[:2], key=lambda item: cv2.boundingRect(item[1])[0])):
            detections.append(
                Detection(f"shoe{idx + 1}", _bbox(contour), score=min(1.0, area / 2200.0), yaw_deg=_contour_yaw_deg(contour))
            )
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {
            "shoe_box": {
                "name": "shoe_box",
                "modelname": "007_shoe-box",
                "model_id": 0,
                "model_id_candidates": [0],
                "table_xy": [0.0, -0.13],
                "z": 0.74,
                "yaw_deg": 0.0,
                "qpos": [0.5, 0.5, -0.5, -0.5],
                "initializer": {"source": "task_fixed_pose"},
            }
        }
        notes: list[str] = []
        shoes = sorted([det for det in detections if det.label.startswith("shoe") and det.label != "shoe_box"], key=lambda item: item.center_xy[0])
        if len(shoes) < 2:
            notes.append(f"Detected {len(shoes)} / 2 shoes.")
        for idx, det in enumerate(shoes[:2], start=1):
            objects[f"shoe{idx}"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.32, 0.32),
                ylim=(-0.12, 0.06),
                z=0.741,
                extra={
                    "modelname": "041_shoe",
                    **self._select_model(det, "041_shoe", list(range(10))),
                    "qpos": [0.707, 0.707, 0, 0],
                    "rotate_rand": True,
                },
            )
        return objects, notes


class LargeObjectPairInitializer(SimpleTaskInitializer):
    task_name = "large_object_pair"

    def _mask_candidates(
        self,
        image_path: Path,
        *,
        min_area: int,
        max_area: int,
        y_max: int = 210,
        include_gray: bool = True,
        include_color: bool = True,
    ) -> list[tuple[float, Any]]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.zeros_like(val, dtype=np.uint8)
        if include_color:
            mask = cv2.bitwise_or(mask, np.where((sat > 35) & (val > 45), 255, 0).astype(np.uint8))
        if include_gray:
            mask = cv2.bitwise_or(mask, np.where((sat < 120) & (val > 25) & (val < 235), 255, 0).astype(np.uint8))
        mask[:20, :] = 0
        mask[y_max:, :] = 0
        mask[:, :5] = 0
        mask[:, -5:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=min_area, max_area=max_area):
            x, y, w, h = cv2.boundingRect(contour)
            if 6 <= w <= 180 and 6 <= h <= 150 and 0.15 <= w / max(h, 1) <= 8.5:
                candidates.append((area, contour))
        return candidates


class PlaceCansPlasticboxInitializer(SimpleTaskInitializer):
    task_name = "place_cans_plasticbox"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []

        blue = _mask_hsv(hsv, (85, 25, 45), (125, 255, 255))
        blue = _clean(blue, ksize=5)
        box_candidates = []
        for area, contour in _contours(blue, min_area=700, max_area=22000):
            x, y, w, h = cv2.boundingRect(contour)
            if 35 <= w <= 150 and 30 <= h <= 130 and 0.5 <= w / max(h, 1) <= 2.5:
                box_candidates.append((area, contour))
        if box_candidates:
            area, contour = box_candidates[0]
            detections.append(Detection("plasticbox", _bbox(contour), score=min(1.0, area / 6000.0), yaw_deg=_contour_yaw_deg(contour)))

        color = np.where((sat > 40) & (val > 50), 255, 0).astype(np.uint8)
        color[:10, :] = 0
        color[205:, :] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            color[max(0, y1 - 8):min(color.shape[0], y2 + 8), max(0, x1 - 8):min(color.shape[1], x2 + 8)] = 0
        color = _clean(color, ksize=3)
        cans = []
        for area, contour in _contours(color, min_area=45, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 70 and 8 <= h <= 90 and 0.15 <= w / max(h, 1) <= 5.0 and 10 <= y <= 190:
                cans.append((area, contour))
        for idx, (area, contour) in enumerate(sorted(cans[:2], key=lambda item: cv2.boundingRect(item[1])[0])):
            detections.append(Detection(f"can{idx + 1}", _bbox(contour), score=min(1.0, area / 1300.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        boxes = [det for det in detections if det.label == "plasticbox"]
        if boxes:
            det = max(boxes, key=lambda item: item.score)
            objects["plasticbox"] = self._project_detection(det, anchor="center", xlim=(-0.05, 0.05), ylim=(-0.17, -0.08), z=0.741, extra={"modelname": "062_plasticbox", **self._select_model(det, "062_plasticbox", [0, 1]), "qpos": [0.5, 0.5, 0.5, 0.5]})
        else:
            notes.append("No plasticbox detection.")
        cans = sorted([det for det in detections if det.label.startswith("can")], key=lambda item: item.center_xy[0])
        if len(cans) < 2:
            notes.append(f"Detected {len(cans)} / 2 cans.")
        for idx, det in enumerate(cans[:2], start=1):
            objects[f"can{idx}"] = self._project_detection(det, anchor="bottom_center", xlim=(-0.3, 0.3), ylim=(-0.18, -0.04), z=0.741, extra={"modelname": "071_can", **self._select_model(det, "071_can", [0, 1, 2, 3, 5, 6]), "qpos": [0.5, 0.5, 0.5, 0.5]})
        return objects, notes


class PlaceBurgerFriesInitializer(SimpleTaskInitializer):
    task_name = "place_burger_fries"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []
        tray_mask = np.where(((sat > 20) & (val > 45) & (val < 230)) | ((sat < 80) & (val > 80) & (val < 220)), 255, 0).astype(np.uint8)
        tray_mask[:25, :] = 0
        tray_mask[205:, :] = 0
        tray_mask = _clean(tray_mask, ksize=5)
        tray_candidates = []
        for area, contour in _contours(tray_mask, min_area=1500, max_area=26000):
            x, y, w, h = cv2.boundingRect(contour)
            if 45 <= w <= 215 and 30 <= h <= 135 and 0.6 <= w / max(h, 1) <= 3.2:
                tray_candidates.append((area, contour))
        if tray_candidates:
            area, contour = tray_candidates[0]
            detections.append(Detection("tray", _bbox(contour), score=min(1.0, area / 9000.0), yaw_deg=_contour_yaw_deg(contour)))

        red_yellow = cv2.bitwise_or(_mask_hsv(hsv, (0, 55, 55), (35, 255, 255)), _mask_hsv(hsv, (160, 55, 55), (180, 255, 255)))
        red_yellow[:20, :] = 0
        red_yellow[205:, :] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            red_yellow[max(0, y1 - 6):min(red_yellow.shape[0], y2 + 6), max(0, x1 - 6):min(red_yellow.shape[1], x2 + 6)] = 0
        red_yellow = _clean(red_yellow, ksize=3)
        food = []
        for area, contour in _contours(red_yellow, min_area=35, max_area=5000):
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 70 and 5 <= h <= 70 and 0.25 <= w / max(h, 1) <= 4.0:
                food.append((area, contour))
        food = sorted(food[:2], key=lambda item: cv2.boundingRect(item[1])[0])
        for idx, (area, contour) in enumerate(food):
            label = "hamburg" if idx == 0 else "frenchfries"
            detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        specs = {
            "tray": ("008_tray", list(range(8)), (-0.04, 0.04), (-0.17, -0.08), [0.706527, 0.706483, -0.0291356, -0.0291767]),
            "hamburg": ("006_hamburg", [0], (-0.33, 0.0), (-0.18, -0.04), [0.5, 0.5, 0.5, 0.5]),
            "frenchfries": ("005_french-fries", [0], (0.0, 0.33), (-0.18, -0.04), [1.0, 0.0, 0.0, 0.0]),
        }
        for label, (model, ids, xlim, ylim, qpos) in specs.items():
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(det, anchor="center", xlim=xlim, ylim=ylim, z=0.741, extra={"modelname": model, **self._select_model(det, model, ids), "qpos": qpos})
        return objects, notes


class PlaceBreadBasketInitializer(SimpleTaskInitializer):
    task_name = "place_bread_basket"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        tan = np.where((sat > 20) & (sat < 190) & (val > 55) & (val < 235), 255, 0).astype(np.uint8)
        tan[:20, :] = 0
        tan[210:, :] = 0
        tan = _clean(tan, ksize=3)
        candidates = []
        for area, contour in _contours(tan, min_area=45, max_area=18000):
            x, y, w, h = cv2.boundingRect(contour)
            if 6 <= w <= 150 and 6 <= h <= 130 and 0.2 <= w / max(h, 1) <= 6.0:
                candidates.append((area, contour))
        detections = []
        if candidates:
            area, contour = candidates[0]
            detections.append(Detection("breadbasket", _bbox(contour), score=min(1.0, area / 6000.0), yaw_deg=_contour_yaw_deg(contour)))
        for idx, (area, contour) in enumerate(sorted(candidates[1:3], key=lambda item: cv2.boundingRect(item[1])[0])):
            detections.append(Detection(f"bread{idx + 1}", _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        baskets = [det for det in detections if det.label == "breadbasket"]
        if baskets:
            det = max(baskets, key=lambda item: item.score)
            objects["breadbasket"] = self._project_detection(det, anchor="center", xlim=(-0.04, 0.04), ylim=(-0.22, -0.16), z=0.741, extra={"modelname": "076_breadbasket", **self._select_model(det, "076_breadbasket", [0, 1, 2, 3, 4]), "qpos": [0.5, 0.5, 0.5, 0.5]})
        else:
            notes.append("No breadbasket detection.")
        breads = sorted([det for det in detections if det.label.startswith("bread")], key=lambda item: item.center_xy[0])
        if len(breads) < 2:
            notes.append(f"Detected {len(breads)} / 2 breads.")
        for idx, det in enumerate(breads[:2], start=1):
            objects[f"bread{idx}"] = self._project_detection(det, anchor="center", xlim=(-0.3, 0.3), ylim=(-0.22, 0.06), z=0.741, extra={"modelname": "075_bread", **self._select_model(det, "075_bread", [0, 1, 3, 5, 6]), "qpos": [0.707, 0.707, 0.0, 0.0]})
        return objects, notes


class PlaceBreadSkilletInitializer(SimpleTaskInitializer):
    task_name = "place_bread_skillet"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections: list[Detection] = []
        dark = np.where((val < 145) & (sat < 170), 255, 0).astype(np.uint8)
        dark[:20, :] = 0
        dark[210:, :] = 0
        dark = _clean(dark, ksize=3)
        skillet_candidates = []
        for area, contour in _contours(dark, min_area=350, max_area=15000):
            x, y, w, h = cv2.boundingRect(contour)
            if 25 <= w <= 140 and 20 <= h <= 120 and 0.45 <= w / max(h, 1) <= 3.5:
                skillet_candidates.append((area, contour))
        if skillet_candidates:
            area, contour = skillet_candidates[0]
            detections.append(Detection("skillet", _bbox(contour), score=min(1.0, area / 5000.0), yaw_deg=_contour_yaw_deg(contour)))
        tan = np.where((sat > 25) & (sat < 190) & (val > 70) & (val < 245), 255, 0).astype(np.uint8)
        tan[:20, :] = 0
        tan[210:, :] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            tan[max(0, y1 - 5):min(tan.shape[0], y2 + 5), max(0, x1 - 5):min(tan.shape[1], x2 + 5)] = 0
        tan = _clean(tan, ksize=3)
        bread_candidates = []
        for area, contour in _contours(tan, min_area=45, max_area=4000):
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 75 and 5 <= h <= 70 and 0.25 <= w / max(h, 1) <= 5.0:
                bread_candidates.append((area, contour))
        if bread_candidates:
            area, contour = bread_candidates[0]
            detections.append(Detection("bread", _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        specs = {
            "skillet": ("106_skillet", list(range(4)), (-0.28, 0.28), (-0.22, 0.06), [0, 0, 0.707, 0.707]),
            "bread": ("075_bread", [0, 1, 3, 5, 6], (-0.3, 0.3), (-0.22, 0.06), [0.707, 0.707, 0.0, 0.0]),
        }
        for label, (model, ids, xlim, ylim, qpos) in specs.items():
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(det, anchor="center", xlim=xlim, ylim=ylim, z=0.741, extra={"modelname": model, **self._select_model(det, model, ids), "qpos": qpos})
        return objects, notes


class PlaceContainerPlateInitializer(SimpleTaskInitializer):
    task_name = "place_container_plate"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where(((sat < 110) & (val > 45) & (val < 235)) | ((sat > 35) & (val > 45)), 255, 0).astype(np.uint8)
        mask[:20, :] = 0
        mask[210:, :] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=80, max_area=15000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 150 and 8 <= h <= 135 and 0.2 <= w / max(h, 1) <= 6.0:
                candidates.append((area, contour))
        detections = []
        for idx, (area, contour) in enumerate(candidates[:2]):
            x, y, w, h = cv2.boundingRect(contour)
            label = "plate" if area > 1200 or w > 55 or h > 55 else "container"
            detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 4000.0), yaw_deg=_contour_yaw_deg(contour)))
        if len(detections) == 2 and detections[0].label == detections[1].label:
            smaller = 0 if detections[0].score < detections[1].score else 1
            detections[smaller] = Detection("container", detections[smaller].bbox_xyxy, detections[smaller].score, detections[smaller].yaw_deg)
            detections[1 - smaller] = Detection("plate", detections[1 - smaller].bbox_xyxy, detections[1 - smaller].score, detections[1 - smaller].yaw_deg)
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        specs = {
            "plate": ("003_plate", list(range(7)), (-0.32, 0.32), (-0.17, -0.08), [0.5, 0.5, 0.5, 0.5]),
            "container": ("004_container", list(range(8)), (-0.32, 0.32), (-0.12, 0.06), [0.5, 0.5, 0.5, 0.5]),
        }
        for label, (model, ids, xlim, ylim, qpos) in specs.items():
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(det, anchor="center", xlim=xlim, ylim=ylim, z=0.741, extra={"modelname": model, **self._select_model(det, model, ids), "qpos": qpos})
        return objects, notes


class MovePlayingCardAwayInitializer(SimpleTaskInitializer):
    task_name = "move_playingcard_away"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        candidates = LargeObjectPairInitializer("x")._mask_candidates(image_path, min_area=45, max_area=5000, y_max=205)
        detections = []
        if candidates:
            area, contour = candidates[0]
            detections.append(Detection("playingcards", _bbox(contour), score=min(1.0, area / 1500.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        matches = [det for det in detections if det.label == "playingcards"]
        if not matches:
            return {}, ["No playing-card detection."]
        det = max(matches, key=lambda item: item.score)
        return {"playingcards": self._project_detection(det, anchor="center", xlim=(-0.12, 0.12), ylim=(-0.22, 0.06), z=0.741, extra={"modelname": "081_playingcards", **self._select_model(det, "081_playingcards", [0, 1, 2]), "qpos": [0.5, 0.5, 0.5, 0.5]})}, []


class RotateQrcodeInitializer(MovePlayingCardAwayInitializer):
    task_name = "rotate_qrcode"

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        matches = detections
        if not matches:
            return {}, ["No QR payment sign detection."]
        det = max(matches, key=lambda item: item.score)
        return {"qrcode": self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.22, 0.02), z=0.741, extra={"modelname": "070_paymentsign", **self._select_model(det, "070_paymentsign", list(range(6))), "qpos": [0, 0, 0.707, 0.707]})}, []


class OpenLaptopInitializer(MovePlayingCardAwayInitializer):
    task_name = "open_laptop"

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        matches = detections
        if not matches:
            return {}, ["No laptop detection."]
        det = max(matches, key=lambda item: item.score)
        return {"laptop": self._project_detection(det, anchor="center", xlim=(-0.08, 0.08), ylim=(-0.12, 0.06), z=0.741, extra={"modelname": "015_laptop", "model_id": 0, "model_id_candidates": list(range(11)), "qpos": [0.7, 0, 0, 0.7], "articulation_qpos": [0.2]})}, []


class BlocksRankingRgbInitializer(StackBlocksThreeInitializer):
    task_name = "blocks_ranking_rgb"


class BlocksRankingSizeInitializer(SimpleTaskInitializer):
    task_name = "blocks_ranking_size"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where((sat > 35) & (val > 45), 255, 0).astype(np.uint8)
        mask[:20, :] = 0
        mask[205:, :] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=70, max_area=7000):
            x, y, w, h = cv2.boundingRect(contour)
            if 9 <= w <= 90 and 9 <= h <= 90 and 0.35 <= w / max(h, 1) <= 2.7:
                candidates.append((area, contour))
        return [
            Detection(f"block{idx + 1}", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour))
            for idx, (area, contour) in enumerate(candidates[:3])
        ]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        if len(detections) < 3:
            notes.append(f"Detected {len(detections)} / 3 ranking blocks.")
        for idx, det in enumerate(sorted(detections, key=lambda item: item.score, reverse=True)[:3], start=1):
            objects[f"block{idx}"] = self._project_detection(
                det,
                anchor="center",
                xlim=(-0.3, 0.3),
                ylim=(-0.1, 0.07),
                z=0.766,
                extra={"kind": "box", "color": "unknown", "half_size": [0.025, 0.025, 0.025]},
            )
        return objects, notes


class HandoverBlockInitializer(SimpleTaskInitializer):
    task_name = "handover_block"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        detections = []
        red = _clean(_red_mask(hsv), ksize=3)
        red_candidates = _contours(red, min_area=80, max_area=12000)
        if red_candidates:
            area, contour = red_candidates[0]
            detections.append(Detection("block", _bbox(contour), score=min(1.0, area / 2500.0), yaw_deg=_contour_yaw_deg(contour)))
        blue = _clean(_mask_hsv(hsv, (95, 60, 45), (130, 255, 255)), ksize=3)
        blue_candidates = _contours(blue, min_area=80, max_area=7000)
        if blue_candidates:
            area, contour = blue_candidates[0]
            detections.append(Detection("target_box", _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        for label, z, half_size, xlim, ylim, color in [
            ("block", 0.842, [0.03, 0.03, 0.1], (-0.3, 0.0), (-0.02, 0.26), [1, 0, 0]),
            ("target_box", 0.741, [0.05, 0.05, 0.005], (0.05, 0.3), (0.1, 0.22), [0, 0, 1]),
        ]:
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(det, anchor="center", xlim=xlim, ylim=ylim, z=z, extra={"kind": "box", "color": color, "half_size": half_size})
        return objects, notes


class SlenderObjectInitializer(SimpleTaskInitializer):
    task_name = "slender_object"
    label = "object"
    modelname = "object"
    model_candidates: list[int] = [0]
    xlim = (-0.3, 0.3)
    ylim = (-0.25, 0.08)
    z = 0.741
    qpos = [0.5, 0.5, 0.5, 0.5]

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where(((sat > 25) & (val > 45)) | ((val < 145) & (sat < 210)), 255, 0).astype(np.uint8)
        mask[:20, :] = 0
        mask[210:, :] = 0
        mask[:, :5] = 0
        mask[:, -5:] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=60, max_area=9000):
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(min(w, h), 1)
            if 8 <= w <= 140 and 8 <= h <= 130 and aspect >= 1.25:
                candidates.append((area, contour))
        if not candidates:
            return []
        area, contour = candidates[0]
        return [Detection(self.label, _bbox(contour), score=min(1.0, area / 2000.0), yaw_deg=_contour_yaw_deg(contour))]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        if not detections:
            return {}, [f"No {self.label} detection."]
        det = max(detections, key=lambda item: item.score)
        obj = self._project_detection(det, anchor="center", xlim=self.xlim, ylim=self.ylim, z=self.z, extra={"modelname": self.modelname, **self._select_model(det, self.modelname, self.model_candidates), "qpos": self.qpos, "rotate_rand": True})
        return {self.label: obj}, []


class GrabRollerInitializer(SlenderObjectInitializer):
    task_name = "grab_roller"
    label = "roller"
    modelname = "102_roller"
    model_candidates = [0, 2]
    xlim = (-0.18, 0.18)
    ylim = (-0.27, -0.04)


class HandoverMicInitializer(SlenderObjectInitializer):
    task_name = "handover_mic"
    label = "microphone"
    modelname = "018_microphone"
    model_candidates = [0, 4, 5]
    xlim = (-0.23, 0.23)
    ylim = (-0.08, 0.03)
    qpos = [0.707, 0.707, 0, 0]


class LiftPotInitializer(SimpleTaskInitializer):
    task_name = "lift_pot"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        candidates = LargeObjectPairInitializer("x")._mask_candidates(image_path, min_area=1000, max_area=26000, y_max=210)
        if not candidates:
            return []
        area, contour = candidates[0]
        return [Detection("pot", _bbox(contour), score=min(1.0, area / 9000.0), yaw_deg=_contour_yaw_deg(contour))]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        if not detections:
            return {}, ["No pot detection."]
        det = max(detections, key=lambda item: item.score)
        return {"pot": self._project_detection(det, anchor="center", xlim=(-0.08, 0.08), ylim=(-0.07, 0.07), z=0.741, extra={"modelname": "060_kitchenpot", "model_id": 0, "model_id_candidates": [0, 1], "qpos": [0.704141, 0, 0, 0.71006]})}, []


class StampSealInitializer(SimpleTaskInitializer):
    task_name = "stamp_seal"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        detections = []
        color = np.where((sat > 45) & (val > 45), 255, 0).astype(np.uint8)
        color[:20, :] = 0
        color[205:, :] = 0
        color = _clean(color, ksize=3)
        candidates = _contours(color, min_area=40, max_area=5000)
        if candidates:
            area, contour = candidates[0]
            detections.append(Detection("target", _bbox(contour), score=min(1.0, area / 1200.0), yaw_deg=_contour_yaw_deg(contour)))
        dark = np.where((val < 150) & (sat < 220), 255, 0).astype(np.uint8)
        dark[:20, :] = 0
        dark[205:, :] = 0
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
            dark[max(0, y1 - 5):min(dark.shape[0], y2 + 5), max(0, x1 - 5):min(dark.shape[1], x2 + 5)] = 0
        dark = _clean(dark, ksize=3)
        seal_candidates = _contours(dark, min_area=60, max_area=6000)
        if seal_candidates:
            area, contour = seal_candidates[0]
            detections.append(Detection("seal", _bbox(contour), score=min(1.0, area / 1600.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        for label in ("seal", "target"):
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            if label == "seal":
                objects["seal"] = self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.08, 0.08), z=0.741, extra={"modelname": "100_seal", **self._select_model(det, "100_seal", [0, 2, 3, 4, 6]), "qpos": [0.5, 0.5, 0.5, 0.5]})
            else:
                objects["target"] = self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.08, 0.12), z=0.741, extra={"kind": "box", "color": "unknown", "half_size": [0.035, 0.035, 0.0005]})
        return objects, notes


class HangingMugInitializer(SimpleTaskInitializer):
    task_name = "hanging_mug"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.where(((sat > 25) & (val > 45)) | ((val < 160) & (sat < 180)), 255, 0).astype(np.uint8)
        mask[:20, :] = 0
        mask[210:, :] = 0
        mask = _clean(mask, ksize=3)
        candidates = []
        for area, contour in _contours(mask, min_area=80, max_area=16000):
            x, y, w, h = cv2.boundingRect(contour)
            if 8 <= w <= 130 and 8 <= h <= 140:
                candidates.append((area, contour))
        detections = []
        for idx, (area, contour) in enumerate(candidates[:2]):
            label = "rack" if cv2.boundingRect(contour)[0] > 150 else "mug"
            detections.append(Detection(label, _bbox(contour), score=min(1.0, area / 4000.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        specs = {
            "mug": ("039_mug", list(range(10)), (-0.3, 0.0), (-0.08, 0.08), [0.707, 0.707, 0, 0]),
            "rack": ("040_rack", [0], (0.08, 0.32), (0.1, 0.19), [-0.22, -0.22, 0.67, 0.67]),
        }
        for label, (model, ids, xlim, ylim, qpos) in specs.items():
            matches = [det for det in detections if det.label == label]
            if not matches:
                notes.append(f"No {label} detection.")
                continue
            det = max(matches, key=lambda item: item.score)
            objects[label] = self._project_detection(det, anchor="center", xlim=xlim, ylim=ylim, z=0.741, extra={"modelname": model, **self._select_model(det, model, ids), "qpos": qpos})
        return objects, notes


class DumpBinBigbinInitializer(SimpleTaskInitializer):
    task_name = "dump_bin_bigbin"

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        candidates = LargeObjectPairInitializer("x")._mask_candidates(image_path, min_area=350, max_area=18000, y_max=210)
        if not candidates:
            return []
        area, contour = candidates[0]
        return [Detection("deskbin", _bbox(contour), score=min(1.0, area / 5000.0), yaw_deg=_contour_yaw_deg(contour))]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects = {
            "dustbin": {"name": "dustbin", "modelname": "011_dustbin", "model_id": 0, "model_id_candidates": [0], "table_xy": [-0.45, 0.0], "z": 0.0, "yaw_deg": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5], "initializer": {"source": "task_fixed_pose"}}
        }
        if not detections:
            return objects, ["No deskbin detection."]
        det = max(detections, key=lambda item: item.score)
        objects["deskbin"] = self._project_detection(det, anchor="center", xlim=(-0.24, 0.24), ylim=(-0.22, -0.03), z=0.741, extra={"modelname": "063_tabletrashbin", **self._select_model(det, "063_tabletrashbin", [0, 3, 7, 8, 9, 10]), "qpos": [0.651892, 0.651428, 0.274378, 0.274584]})
        return objects, []


class GenericTwoObjectInitializer(SimpleTaskInitializer):
    task_name = "generic_two_object"
    labels = ("object", "target_object")

    def _detect(self, image_path: Path) -> list[Detection]:
        import cv2

        candidates = LargeObjectPairInitializer("x")._mask_candidates(image_path, min_area=45, max_area=9000, y_max=210)
        detections = []
        for idx, (area, contour) in enumerate(candidates[:2]):
            detections.append(Detection(self.labels[idx], _bbox(contour), score=min(1.0, area / 1800.0), yaw_deg=_contour_yaw_deg(contour)))
        return detections

    def _generic_extra(self, det: Detection) -> dict[str, Any]:
        return {
            "modelname": "047_mouse",
            **self._select_model(det, "047_mouse", [0, 1, 2]),
            "modelname_candidates": [
                "047_mouse",
                "048_stapler",
                "050_bell",
                "057_toycar",
                "073_rubikscube",
                "075_bread",
                "077_phone",
                "081_playingcards",
                "086_woodenblock",
                "112_tea-box",
                "113_coffee-box",
                "107_soap",
            ],
            "qpos": [0.5, 0.5, 0.5, 0.5],
        }

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        if len(detections) < 2:
            notes.append(f"Detected {len(detections)} / 2 generic objects.")
        for det in detections[:2]:
            objects[det.label] = self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.22, 0.04), z=0.741, extra=self._generic_extra(det))
        return objects, notes


class PlaceA2BLeftInitializer(GenericTwoObjectInitializer):
    task_name = "place_a2b_left"


class PlaceA2BRightInitializer(GenericTwoObjectInitializer):
    task_name = "place_a2b_right"


class ScanObjectInitializer(GenericTwoObjectInitializer):
    task_name = "scan_object"
    labels = ("scanner", "object")

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        if len(detections) < 2:
            notes.append(f"Detected {len(detections)} / 2 scan objects.")
        for det in detections[:2]:
            if det.label == "scanner":
                objects["scanner"] = self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.18, -0.03), z=0.741, extra={"modelname": "024_scanner", **self._select_model(det, "024_scanner", [0, 1, 2, 3, 4]), "qpos": [0, 0, 0.707, 0.707]})
            else:
                objects["object"] = self._project_detection(det, anchor="center", xlim=(-0.28, 0.28), ylim=(-0.22, 0.02), z=0.741, extra={"modelname": "112_tea-box", **self._select_model(det, "112_tea-box", [0, 1, 2, 3, 4, 5]), "qpos": [0.5, 0.5, 0.5, 0.5]})
        return objects, notes


class OpenMicrowaveInitializer(SimpleTaskInitializer):
    task_name = "open_microwave"

    def _detect(self, image_path: Path) -> list[Detection]:
        candidates = LargeObjectPairInitializer("x")._mask_candidates(image_path, min_area=1200, max_area=24000, y_max=170)
        if not candidates:
            return []
        area, contour = candidates[0]
        return [Detection("microwave", _bbox(contour), score=min(1.0, area / 8000.0), yaw_deg=_contour_yaw_deg(contour))]

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        if not detections:
            return {}, ["No microwave detection."]
        det = max(detections, key=lambda item: item.score)
        return {"microwave": self._project_detection(det, anchor="center", xlim=(-0.14, 0.0), ylim=(0.12, 0.22), z=0.8, extra={"modelname": "044_microwave", "model_id": 0, "model_id_candidates": [0, 1], "qpos": [0.707, 0, 0, 0.707], "articulation_qpos": [0.0]})}, []


class PutObjectCabinetInitializer(GenericTwoObjectInitializer):
    task_name = "put_object_cabinet"
    labels = ("cabinet", "object")

    def _objects_from_detections(self, detections: list[Detection]) -> tuple[dict[str, dict[str, Any]], list[str]]:
        objects: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        cabinet_matches = [det for det in detections if det.label == "cabinet"]
        object_matches = [det for det in detections if det.label == "object"]
        if cabinet_matches:
            det = max(cabinet_matches, key=lambda item: item.score)
            objects["cabinet"] = self._project_detection(det, anchor="center", xlim=(-0.08, 0.08), ylim=(0.12, 0.18), z=0.741, extra={"modelname": "036_cabinet", "model_id": 46653, "model_id_candidates": [46653], "qpos": [1, 0, 0, 1], "articulation_qpos": [0.0]})
        else:
            notes.append("No cabinet detection.")
        if object_matches:
            det = max(object_matches, key=lambda item: item.score)
            objects["object"] = self._project_detection(det, anchor="center", xlim=(-0.3, 0.3), ylim=(-0.22, 0.04), z=0.741, extra=self._generic_extra(det))
        else:
            notes.append("No cabinet object detection.")
        return objects, notes


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.bitwise_or(
        _mask_hsv(hsv, (0, 70, 50), (10, 255, 255)),
        _mask_hsv(hsv, (170, 70, 50), (180, 255, 255)),
    )
