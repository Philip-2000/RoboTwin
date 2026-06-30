from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskMatch:
    episode: int
    task_name: str
    score: float | None = None
    reason: str | None = None


def default_search_gt_path(worldarena_split: str) -> Path:
    repo = Path("/home/users/liang01.yue/C/WorldArena")
    base = repo / "yl_outputs" / "search_gt"
    if worldarena_split.startswith("val"):
        return base / "worldarena_val_to_robotwin_clean50_ollama_task_gttrace_top10.json"
    if worldarena_split.startswith("test"):
        return base / "worldarena_test_to_robotwin_clean50_ollama_task_gttrace_top10.json"
    raise ValueError(f"Unsupported split for default search_gt path: {worldarena_split}")


class TaskTop1Mapping:
    def __init__(self, matches: dict[int, TaskMatch]):
        self.matches = matches

    @classmethod
    def from_search_gt(cls, path: str | Path) -> "TaskTop1Mapping":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        matches: dict[int, TaskMatch] = {}
        for item in data.get("matches", []):
            episode = int(item["worldarena"]["episode"])
            top = item["task_candidates"][0]
            matches[episode] = TaskMatch(
                episode=episode,
                task_name=str(top["task"]),
                score=top.get("task_score"),
                reason=top.get("ollama_reason"),
            )
        return cls(matches)

    def get(self, episode: int) -> TaskMatch:
        if episode not in self.matches:
            raise KeyError(f"No task top1 mapping for episode{episode}")
        return self.matches[episode]

