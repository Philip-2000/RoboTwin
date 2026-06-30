from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeRef:
    split: str
    episode: int
    instruction_path: Path
    first_frame_path: Path
    hdf5_path: Path
    instruction: str
    trajectory_length: int | None = None


@dataclass(frozen=True)
class ObjectPrior:
    name: str
    modelname: str | None = None
    model_ids: tuple[int, ...] = ()
    kind: str = "asset"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    z: float | None = None
    qpos: tuple[float, ...] | None = None
    color: tuple[float, float, float] | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenePrior:
    task_name: str
    difficulty: str
    objects: tuple[ObjectPrior, ...]
    constraints: tuple[str, ...] = ()
    strategy_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateScene:
    task_name: str
    parameters: dict[str, Any]
    confidence: float = 0.0
    source: str = "prior_only"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Detection:
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float | None = None
    source: str = "unknown"
    mask_path: Path | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(frozen=True)
class DetectionSet:
    image_path: Path
    detections: tuple[Detection, ...] = ()
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_keywords(self, keywords: tuple[str, ...]) -> tuple[Detection, ...]:
        normalized = tuple(k.lower() for k in keywords)
        return tuple(
            det
            for det in self.detections
            if any(keyword in det.label.lower() for keyword in normalized)
        )


@dataclass(frozen=True)
class ReconstructionReport:
    episode: EpisodeRef
    task_name: str
    strategy: str
    prior: ScenePrior
    candidates: tuple[CandidateScene, ...] = ()
    detections: DetectionSet | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("instruction_path", "first_frame_path", "hdf5_path"):
            data["episode"][key] = str(data["episode"][key])
        if data.get("detections") is not None:
            data["detections"]["image_path"] = str(data["detections"]["image_path"])
            for det in data["detections"]["detections"]:
                if det.get("mask_path") is not None:
                    det["mask_path"] = str(det["mask_path"])
        return data
