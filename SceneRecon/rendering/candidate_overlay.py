from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core import CandidateScene, EpisodeRef
from ..geometry import TablePoint, TableProjector


PALETTE = {
    "cup": (40, 120, 255),
    "coaster": (180, 180, 180),
    "bell": (240, 190, 20),
    "red_block": (230, 40, 40),
    "green_block": (40, 190, 80),
    "blue_block": (60, 90, 240),
    "block": (230, 40, 40),
    "hammer": (40, 40, 40),
}


@dataclass(frozen=True)
class RenderableObject:
    name: str
    target_bbox: tuple[float, float, float, float] | None
    render_bbox: tuple[float, float, float, float] | None
    color: tuple[int, int, int]


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(float(v) for v in value)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_with_center(
    bbox: tuple[float, float, float, float],
    center: tuple[float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = center
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _object_entries(parameters: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for key, value in parameters.items():
        if isinstance(value, dict):
            if "bbox_xyxy" in value or "image_center_xy" in value or "table_xy" in value:
                entries.append((key, value))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and ("bbox_xyxy" in item or "image_center_xy" in item or "table_xy" in item):
                    name = str(item.get("name", f"{key}_{index}"))
                    entries.append((name, item))
    return entries


class CandidateOverlayRenderer:
    """Fast visual render proxy for the render-search loop.

    This renderer does not replace SAPIEN. It projects candidate table-plane
    positions back to the first-frame image and draws object-sized proxies. It is
    useful for debugging geometry, candidate search ranges, and report plumbing
    before the slower task-specific SAPIEN renderer is connected.
    """

    def __init__(self, projector: TableProjector | None = None):
        self.projector = projector

    def render_objects(self, candidate: CandidateScene) -> list[RenderableObject]:
        objects: list[RenderableObject] = []
        for name, params in _object_entries(candidate.parameters):
            target_bbox = _as_bbox(params.get("bbox_xyxy"))
            render_bbox = target_bbox
            table_xy = params.get("table_xy")
            if (
                self.projector is not None
                and target_bbox is not None
                and isinstance(table_xy, (list, tuple))
                and len(table_xy) == 2
            ):
                image_point = self.projector.unproject(TablePoint(float(table_xy[0]), float(table_xy[1])))
                render_bbox = _bbox_with_center(target_bbox, (image_point.u, image_point.v))
            elif target_bbox is not None and params.get("image_center_xy") is not None:
                render_bbox = _bbox_with_center(target_bbox, tuple(params["image_center_xy"]))
            color = PALETTE.get(name, (255, 120, 40))
            objects.append(RenderableObject(name=name, target_bbox=target_bbox, render_bbox=render_bbox, color=color))
        return objects

    def score(self, candidate: CandidateScene) -> dict[str, Any]:
        objects = self.render_objects(candidate)
        per_object = []
        scores = []
        for obj in objects:
            if obj.target_bbox is None or obj.render_bbox is None:
                continue
            iou = _iou(obj.target_bbox, obj.render_bbox)
            tc = np.asarray(_bbox_center(obj.target_bbox), dtype=np.float64)
            rc = np.asarray(_bbox_center(obj.render_bbox), dtype=np.float64)
            center_error_px = float(np.linalg.norm(tc - rc))
            score = 0.7 * iou + 0.3 * float(np.exp(-center_error_px / 35.0))
            scores.append(score)
            per_object.append(
                {
                    "name": obj.name,
                    "iou": iou,
                    "center_error_px": center_error_px,
                    "score": score,
                    "target_bbox_xyxy": list(obj.target_bbox),
                    "render_bbox_xyxy": list(obj.render_bbox),
                }
            )
        return {
            "score": float(np.mean(scores)) if scores else 0.0,
            "per_object": per_object,
            "renderer": "candidate_overlay",
        }

    def save_visualization(
        self,
        episode: EpisodeRef,
        candidate: CandidateScene,
        out_path: Path,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        first = Image.open(episode.first_frame_path).convert("RGB")
        w, h = first.size
        target = first.copy()
        render = Image.new("RGB", (w, h), (238, 238, 232))
        overlay = first.copy()

        objects = self.render_objects(candidate)
        self._draw_target(target, objects)
        self._draw_render(render, objects)
        self._draw_overlay(overlay, objects)

        pad = 18
        title_h = 30
        canvas = Image.new("RGB", (w * 3 + pad * 4, h + title_h + pad * 2), (250, 250, 248))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        labels = ["target first frame", "candidate render proxy", "overlay"]
        for i, panel in enumerate((target, render, overlay)):
            x = pad + i * (w + pad)
            y = title_h + pad
            canvas.paste(panel, (x, y))
            draw.text((x, pad), labels[i], fill=(20, 20, 20), font=font)
        draw.text((pad, h + title_h + pad + 2), title[:140], fill=(20, 20, 20), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=95)
        if metadata is not None:
            out_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _draw_target(image: Image.Image, objects: list[RenderableObject]) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        for obj in objects:
            if obj.target_bbox is None:
                continue
            draw.rectangle(obj.target_bbox, outline=(*obj.color, 255), width=3)
            draw.text((obj.target_bbox[0], max(0, obj.target_bbox[1] - 12)), obj.name, fill=(*obj.color, 255))

    @staticmethod
    def _draw_render(image: Image.Image, objects: list[RenderableObject]) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        for obj in objects:
            if obj.render_bbox is None:
                continue
            draw.rectangle(obj.render_bbox, fill=(*obj.color, 110), outline=(*obj.color, 255), width=3)
            draw.text((obj.render_bbox[0], max(0, obj.render_bbox[1] - 12)), obj.name, fill=(0, 0, 0, 255))

    @staticmethod
    def _draw_overlay(image: Image.Image, objects: list[RenderableObject]) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        for obj in objects:
            if obj.target_bbox is not None:
                draw.rectangle(obj.target_bbox, outline=(255, 255, 255, 230), width=5)
                draw.rectangle(obj.target_bbox, outline=(*obj.color, 230), width=2)
            if obj.render_bbox is not None:
                draw.rectangle(obj.render_bbox, fill=(*obj.color, 55), outline=(0, 0, 0, 230), width=2)
