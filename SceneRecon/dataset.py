from __future__ import annotations

import json
import re
from pathlib import Path

from .core import EpisodeRef


def episode_number(path: Path) -> int:
    match = re.search(r"episode_?(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse episode number from {path}")
    return int(match.group(1))


def read_instruction(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "instruction" not in data:
        raise KeyError(f"Missing 'instruction' in {path}")
    return str(data["instruction"])


def strip_worldarena_prefix(text: str) -> str:
    marker = "enters the frame to "
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def hdf5_length(path: Path, dataset: str = "joint_action/vector") -> int | None:
    try:
        import h5py
    except ModuleNotFoundError:
        return None
    with h5py.File(path, "r") as f:
        if dataset not in f:
            return None
        return int(f[dataset].shape[0])


class WorldArenaDataset:
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split

    def episode(self, episode: int) -> EpisodeRef:
        rel = Path("fixed_scene_task") / f"episode{episode}.json"
        instruction_path = self.split_root / "instructions" / rel
        first_frame_path = self.split_root / "first_frame" / rel.with_suffix(".png")
        hdf5_path = self.split_root / "data" / rel.with_suffix(".hdf5")
        missing = [p for p in (instruction_path, first_frame_path, hdf5_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing WorldArena files: " + ", ".join(str(p) for p in missing))
        full_instruction = read_instruction(instruction_path)
        return EpisodeRef(
            split=self.split,
            episode=episode,
            instruction_path=instruction_path,
            first_frame_path=first_frame_path,
            hdf5_path=hdf5_path,
            instruction=strip_worldarena_prefix(full_instruction),
            trajectory_length=hdf5_length(hdf5_path),
        )

    def episodes(self) -> list[int]:
        inst_dir = self.split_root / "instructions" / "fixed_scene_task"
        return sorted(episode_number(p) for p in inst_dir.glob("episode*.json"))

