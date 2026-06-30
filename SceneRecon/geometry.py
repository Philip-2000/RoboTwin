from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImagePoint:
    u: float
    v: float


@dataclass(frozen=True)
class TablePoint:
    x: float
    y: float


@dataclass(frozen=True)
class BBoxProjection:
    bbox_xyxy: tuple[float, float, float, float]
    center: TablePoint
    bottom_center: TablePoint
    top_center: TablePoint
    corners: tuple[TablePoint, TablePoint, TablePoint, TablePoint]
    anchor: str
    selected: TablePoint

    def to_dict(self) -> dict:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "anchor": self.anchor,
            "selected": {"x": self.selected.x, "y": self.selected.y},
            "center": {"x": self.center.x, "y": self.center.y},
            "bottom_center": {"x": self.bottom_center.x, "y": self.bottom_center.y},
            "top_center": {"x": self.top_center.x, "y": self.top_center.y},
            "corners": [{"x": point.x, "y": point.y} for point in self.corners],
        }


class TableProjector:
    """Project image points to table coordinates.

    This is a deliberately small abstraction. The first implementation can be a
    hand-calibrated homography; later strategies can swap in camera-matrix based
    projection without changing task-specific reconstructors.
    """

    def project(self, point: ImagePoint) -> TablePoint:
        raise NotImplementedError

    def unproject(self, point: TablePoint) -> ImagePoint:
        raise NotImplementedError

    def project_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        anchor: str = "bottom_center",
    ) -> BBoxProjection:
        x1, y1, x2, y2 = bbox_xyxy
        center = self.project(ImagePoint((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        bottom_center = self.project(ImagePoint((x1 + x2) * 0.5, y2))
        top_center = self.project(ImagePoint((x1 + x2) * 0.5, y1))
        corners = (
            self.project(ImagePoint(x1, y1)),
            self.project(ImagePoint(x2, y1)),
            self.project(ImagePoint(x2, y2)),
            self.project(ImagePoint(x1, y2)),
        )
        selected_by_anchor = {
            "center": center,
            "bottom_center": bottom_center,
            "top_center": top_center,
        }
        if anchor not in selected_by_anchor:
            raise ValueError(f"Unsupported bbox projection anchor: {anchor}")
        return BBoxProjection(
            bbox_xyxy=bbox_xyxy,
            center=center,
            bottom_center=bottom_center,
            top_center=top_center,
            corners=corners,
            anchor=anchor,
            selected=selected_by_anchor[anchor],
        )


class IdentityTableProjector(TableProjector):
    """Debug projector that keeps pixel coordinates as pseudo table coordinates."""

    def project(self, point: ImagePoint) -> TablePoint:
        return TablePoint(x=point.u, y=point.v)

    def unproject(self, point: TablePoint) -> ImagePoint:
        return ImagePoint(u=point.x, v=point.y)


class HomographyTableProjector(TableProjector):
    """Map image pixels to table x/y using a 3x3 homography."""

    def __init__(self, image_to_table: np.ndarray):
        image_to_table = np.asarray(image_to_table, dtype=np.float64)
        if image_to_table.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 homography, got {image_to_table.shape}")
        self.image_to_table = image_to_table
        self.table_to_image = np.linalg.inv(image_to_table)

    def project(self, point: ImagePoint) -> TablePoint:
        vec = np.array([point.u, point.v, 1.0], dtype=np.float64)
        out = self.image_to_table @ vec
        out = out / out[2]
        return TablePoint(x=float(out[0]), y=float(out[1]))

    def unproject(self, point: TablePoint) -> ImagePoint:
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        out = self.table_to_image @ vec
        out = out / out[2]
        return ImagePoint(u=float(out[0]), v=float(out[1]))

    @classmethod
    def from_points(
        cls,
        image_points: list[tuple[float, float]],
        table_points: list[tuple[float, float]],
    ) -> "HomographyTableProjector":
        if len(image_points) < 4 or len(table_points) < 4 or len(image_points) != len(table_points):
            raise ValueError("Homography calibration needs at least 4 paired points")
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("OpenCV is required for homography calibration") from exc
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(table_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise RuntimeError("cv2.findHomography failed")
        return cls(matrix)


def default_head_table_projector() -> HomographyTableProjector:
    """Rough 320x240 head-camera table calibration.

    This is intentionally approximate. It maps the central visible table region
    to the RoboTwin table-plane workspace used by most tasks, and should be
    refined by render-and-compare optimization.
    """

    image_points = [
        (35.0, 205.0),   # front-left visible table region
        (285.0, 205.0),  # front-right
        (70.0, 55.0),    # back-left
        (250.0, 55.0),   # back-right
    ]
    table_points = [
        (-0.32, -0.22),
        (0.32, -0.22),
        (-0.32, 0.25),
        (0.32, 0.25),
    ]
    return HomographyTableProjector.from_points(image_points, table_points)


def load_table_projector(path: str | Path | None, use_default: bool = False) -> TableProjector | None:
    if path is None and not use_default:
        return None
    if path is None:
        return default_head_table_projector()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "image_to_table" in data:
        return HomographyTableProjector(np.asarray(data["image_to_table"], dtype=np.float64))
    if "image_points" in data and "table_points" in data:
        return HomographyTableProjector.from_points(data["image_points"], data["table_points"])
    raise ValueError(f"Unsupported calibration JSON schema: {path}")
