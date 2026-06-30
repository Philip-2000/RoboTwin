from __future__ import annotations

from pathlib import Path

from ..core import DetectionSet


class QwenVLVerifier:
    """Placeholder for Qwen-VL based detection verification.

    Expected future role:
    - identify which detection corresponds to instruction entities
    - assign semantic attributes such as color/material/target role
    - reject false positives before geometric fitting
    """

    def __init__(self, model_path: str | Path | None = None, **kwargs):
        self.model_path = None if model_path is None else Path(model_path)
        self.kwargs = kwargs

    def verify(self, detections: DetectionSet, instruction: str) -> DetectionSet:
        raise NotImplementedError(
            "QwenVLVerifier is a placeholder. After model weights are available, "
            "implement semantic verification and return an updated DetectionSet."
        )

