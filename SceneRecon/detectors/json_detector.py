from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import Detection, DetectionSet
from .base import BaseDetector


class JsonDetector(BaseDetector):
    """Load detections from a JSON file.

    Supported schema:

    ```json
    {
      "image_path": "...optional...",
      "source": "manual",
      "detections": [
        {"label": "cup", "bbox_xyxy": [10, 20, 30, 40], "score": 0.9}
      ]
    }
    ```

    A bare list of detection dictionaries is also accepted.
    """

    name = "json"

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)

    def detect(self, image_path: str | Path, prompts: tuple[str, ...] = ()) -> DetectionSet:
        image_path = Path(image_path)
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            source = "json"
            detections_data = data
            json_image_path = image_path
            metadata: dict[str, Any] = {}
        else:
            source = str(data.get("source", "json"))
            detections_data = data.get("detections", [])
            json_image_path = Path(data.get("image_path", image_path))
            metadata = dict(data.get("metadata", {}))

        detections = []
        for item in detections_data:
            bbox = item.get("bbox_xyxy") or item.get("bbox")
            if bbox is None or len(bbox) != 4:
                raise ValueError(f"Detection is missing bbox_xyxy: {item}")
            mask_path = item.get("mask_path")
            detections.append(
                Detection(
                    label=str(item["label"]),
                    bbox_xyxy=tuple(float(v) for v in bbox),
                    score=None if item.get("score") is None else float(item["score"]),
                    source=str(item.get("source", source)),
                    mask_path=None if mask_path is None else Path(mask_path),
                    attributes=dict(item.get("attributes", {})),
                )
            )
        return DetectionSet(
            image_path=json_image_path,
            detections=tuple(detections),
            source=source,
            metadata=metadata | {"json_path": str(self.json_path), "prompts": list(prompts)},
        )

