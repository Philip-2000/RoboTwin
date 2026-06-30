from __future__ import annotations

from pathlib import Path

from ..core import DetectionSet
from .base import BaseDetector


class YOLOWorldDetector(BaseDetector):
    """Placeholder adapter for YOLO-World style open-vocabulary detection.

    The class is intentionally dependency-light. Once weights and the chosen
    implementation are installed, this adapter should translate model outputs
    into SceneRecon `DetectionSet`.
    """

    name = "yolo_world"

    def __init__(self, model_path: str | Path | None = None, **kwargs):
        self.model_path = None if model_path is None else Path(model_path)
        self.kwargs = kwargs

    def detect(self, image_path: str | Path, prompts: tuple[str, ...] = ()) -> DetectionSet:
        raise NotImplementedError(
            "YOLOWorldDetector is a placeholder. Install/select a YOLO-World implementation "
            "and wire its outputs to SceneRecon.core.DetectionSet."
        )

