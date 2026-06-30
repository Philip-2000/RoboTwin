from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..core import DetectionSet


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, image_path: str | Path, prompts: tuple[str, ...] = ()) -> DetectionSet:
        raise NotImplementedError

