from __future__ import annotations

from dataclasses import replace

from .core import CandidateScene, DetectionSet
from .geometry import ImagePoint, TableProjector


def _fill_value(value, detections: DetectionSet, projector: TableProjector | None):
    if isinstance(value, dict):
        return {k: _fill_value(v, detections, projector) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_value(v, detections, projector) for v in value]
    if value != "project_center_with_calibrated_table_projector":
        return value
    if projector is None:
        return value
    # The caller replaces this placeholder only inside an object dict that also
    # contains image_center_xy; handled in fill_projected_table_xy.
    return value


def fill_projected_table_xy(
    candidates: tuple[CandidateScene, ...],
    projector: TableProjector | None,
) -> tuple[CandidateScene, ...]:
    if projector is None:
        return candidates

    def walk(obj):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if key == "table_xy" and value == "project_center_with_calibrated_table_projector":
                    center = obj.get("image_center_xy")
                    if center is None:
                        out[key] = value
                    else:
                        point = projector.project(ImagePoint(u=float(center[0]), v=float(center[1])))
                        out[key] = [point.x, point.y]
                else:
                    out[key] = walk(value)
            return out
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    return tuple(
        replace(
            candidate,
            parameters=walk(candidate.parameters),
            notes=tuple(candidate.notes) + ("Projected image centers to table_xy.",),
        )
        for candidate in candidates
    )


def _default_projection_anchor(obj: dict) -> str:
    explicit = obj.get("projection_anchor")
    if explicit is not None:
        return str(explicit)
    name = str(obj.get("name", obj.get("modelname", ""))).lower()
    if any(word in name for word in ("coaster", "pad", "mat", "plate", "basket")):
        return "center"
    return "bottom_center"


def _default_z(obj: dict) -> float:
    if "z" in obj:
        return float(obj["z"])
    name = str(obj.get("name", obj.get("modelname", obj.get("kind", "")))).lower()
    if "block" in name or obj.get("kind") == "box":
        return 0.76
    if "hammer" in name:
        return 0.783
    return 0.741


def _clamp(value: float, bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return value, False
    lo, hi = float(bounds[0]), float(bounds[1])
    return min(max(value, lo), hi), value < lo or value > hi


def _add_bbox_projection(obj: dict, projector: TableProjector) -> dict:
    bbox = obj.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return obj

    out = dict(obj)
    anchor = _default_projection_anchor(out)
    projection = projector.project_bbox(tuple(float(v) for v in bbox), anchor=anchor)
    selected = projection.selected
    x, x_clamped = _clamp(selected.x, out.get("xlim") or out.get("prior_xlim"))
    y, y_clamped = _clamp(selected.y, out.get("ylim") or out.get("prior_ylim"))
    out["bbox_projection"] = projection.to_dict()
    out["table_xy_raw"] = [selected.x, selected.y]
    out["table_xy"] = [x, y]
    out["table_pose_guess"] = {
        "p": [x, y, _default_z(out)],
        "yaw": float(out.get("yaw", 0.0)) if out.get("yaw") != "fit" else 0.0,
        "qpos": out.get("qpos"),
        "anchor": anchor,
        "clamped_to_prior": {"x": x_clamped, "y": y_clamped},
    }
    return out


def fill_projected_bbox_3d(
    candidates: tuple[CandidateScene, ...],
    projector: TableProjector | None,
) -> tuple[CandidateScene, ...]:
    """Project 2D detections to table-plane 3D pose guesses.

    A single RGB bbox cannot determine full 6DoF. This uses the task's table
    plane assumption to recover x/y, then fills z/qpos/yaw from task priors.
    """
    if projector is None:
        return candidates

    def walk(obj):
        if isinstance(obj, dict):
            out = {key: walk(value) for key, value in obj.items()}
            return _add_bbox_projection(out, projector)
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    return tuple(
        replace(
            candidate,
            parameters=walk(candidate.parameters),
            notes=tuple(candidate.notes) + ("Projected bbox anchors to table-plane 3D pose guesses.",),
        )
        for candidate in candidates
    )


def add_local_search_ranges(
    candidates: tuple[CandidateScene, ...],
    xy_radius: float = 0.035,
    yaw_radius: float = 0.4,
) -> tuple[CandidateScene, ...]:
    def walk(obj):
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                out[key] = walk(value)
            table_xy = out.get("table_xy")
            if isinstance(table_xy, list) and len(table_xy) == 2:
                out.setdefault(
                    "search",
                    {
                        "x": [table_xy[0] - xy_radius, table_xy[0] + xy_radius],
                        "y": [table_xy[1] - xy_radius, table_xy[1] + xy_radius],
                    },
                )
            if out.get("yaw") == "fit":
                out["yaw"] = 0.0
                out.setdefault("search", {})["yaw"] = [-yaw_radius, yaw_radius]
            return out
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        return obj

    return tuple(
        replace(
            candidate,
            parameters=walk(candidate.parameters),
            notes=tuple(candidate.notes) + ("Added local render-search ranges.",),
        )
        for candidate in candidates
    )
