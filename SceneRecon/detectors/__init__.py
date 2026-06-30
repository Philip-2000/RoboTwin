"""Detector adapters for SceneRecon.

Concrete model adapters should convert their output into `DetectionSet`.
"""

from .base import BaseDetector
from .json_detector import JsonDetector
from .simple_cv import SimpleCVDetector

__all__ = ["BaseDetector", "JsonDetector", "SimpleCVDetector"]
