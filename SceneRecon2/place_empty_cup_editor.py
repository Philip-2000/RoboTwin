from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import re
import sys
import time
import traceback
import types
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "SceneRecon2" / "outputs" / "place_empty_cup_editor"
BASE_CAMERA_POSITION = [-0.032, -0.45, 1.35]
BASE_CAMERA_FORWARD = [0.0, 0.6, -0.8]
BASE_CAMERA_LEFT = [-1.0, 0.0, 0.0]
EDITOR_TASK_NAME = "place_empty_cup"
BASE_TABLE_OBJECT_QPOS = [0.5, 0.5, 0.5, 0.5]


def _default_first_frame() -> Path:
    candidates = [
        Path("/home/users/liang01.yue/D/WorldArena_Robotwin2.0/test_dataset/first_frame/fixed_scene_task/episode37.png"),
        Path("/home/users/liang01.yue/D/WorldArena_Robotwin2.0/val_dataset/first_frame/fixed_scene_task/episode37.png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _now_token() -> str:
    return str(time.time_ns())


def _episode_name(path: Path) -> str:
    match = re.search(r"(episode\d+)", path.stem)
    return match.group(1) if match else path.stem


def _episode_number(path: Path) -> int | None:
    match = re.search(r"episode(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _split_from_path(path: Path) -> str:
    parts = path.parts
    if "val_dataset" in parts:
        return "val_dataset"
    if "test_dataset" in parts:
        return "test_dataset"
    return "val_dataset"


def _load_task_matches(split: str) -> dict[int, str]:
    from SceneRecon.task_mapping import TaskTop1Mapping, default_search_gt_path

    mapping = TaskTop1Mapping.from_search_gt(default_search_gt_path(split))
    return {episode: match.task_name for episode, match in mapping.matches.items()}


def _all_robotwin_tasks() -> list[str]:
    tasks = []
    for path in (ROOT / "envs").glob("*.py"):
        name = path.stem
        if name.startswith("_") or name in {"__init__", "empty_green_table"}:
            continue
        tasks.append(name)
    return sorted(tasks)


def _state_path_for_episode(state_dir: Path, episode: int, task_name: str) -> Path:
    return state_dir / f"episode{episode}.{task_name}.json"


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _yaw_quat(base_qpos: list[float] | tuple[float, ...], yaw_deg: float) -> list[float]:
    base = np.asarray(base_qpos, dtype=np.float64)
    half = math.radians(float(yaw_deg)) * 0.5
    yaw = np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
    w1, x1, y1, z1 = yaw
    w2, x2, y2, z2 = base
    q = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    return [float(v) for v in q]


def _geometry_signature(objects: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    signature = []
    for name, obj in sorted(objects.items()):
        half_size = obj.get("half_size", [])
        if isinstance(half_size, (list, tuple)):
            half_size_sig = tuple(round(float(v), 6) for v in half_size)
        else:
            half_size_sig = ()
        signature.append(
            (
                str(name),
                str(obj.get("kind", "model")),
                str(obj.get("modelname", "")),
                int(obj.get("model_id", 0)),
                half_size_sig,
            )
        )
    return tuple(signature)


def _model_ids(modelname: str | None) -> list[int]:
    if not modelname:
        return []
    model_dir = Path("assets/objects") / str(modelname)
    if not model_dir.exists():
        return []
    ids = []
    for path in model_dir.glob("model_data*.json"):
        match = re.fullmatch(r"model_data(\d+)\.json", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(set(ids))


def _shift_model_id(modelname: str | None, current_id: int, delta: int) -> int:
    ids = _model_ids(modelname)
    if not ids:
        return max(0, int(current_id) + int(delta))
    if current_id in ids:
        index = ids.index(current_id)
    else:
        index = min(range(len(ids)), key=lambda i: abs(ids[i] - current_id))
    return ids[(index + int(delta)) % len(ids)]


def _object_qpos(params: dict[str, Any], name: str) -> list[float]:
    item = params.get(name, {})
    if not isinstance(item, dict):
        return list(BASE_TABLE_OBJECT_QPOS)
    base_qpos = item.get("qpos", BASE_TABLE_OBJECT_QPOS)
    return _yaw_quat(list(base_qpos), float(item.get("yaw_deg", 0.0)))


def _install_editor_curobo_stub() -> None:
    """Let the editor import RoboTwin rendering code without curobo planning."""

    module_name = "envs.robot.planner"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "CuroboPlanner"):
        return

    planner_stub = types.ModuleType(module_name)

    class _DisabledPlanner:
        def __init__(self, *args, **kwargs):
            pass

        def plan_grippers(self, now_val, target_val):
            vals = np.linspace(now_val, target_val, 2)
            return {"num_step": 2, "per_step": float(target_val - now_val) / 2.0, "result": vals}

        def __getattr__(self, name):
            raise RuntimeError(f"Robot planner method {name!r} is disabled in the SceneRecon2 editor.")

    planner_stub.CuroboPlanner = _DisabledPlanner
    planner_stub.MplibPlanner = _DisabledPlanner
    sys.modules[module_name] = planner_stub


class PlaceEmptyCupSession:
    def __init__(
        self,
        first_frame: Path,
        display_scale: float = 1.0,
        jpeg_quality: int = 95,
        auto_initialize: bool = True,
    ):
        self.display_scale = display_scale
        self.jpeg_quality = jpeg_quality
        self.auto_initialize = auto_initialize
        self.output_dir = OUTPUT_DIR
        self.split = _split_from_path(first_frame)
        self.state_dir = self.output_dir / "states" / self.split
        self.auto_state_dir = self.output_dir / "auto_states" / self.split
        self.initializer_state_dir = self.output_dir / "initializer_states" / self.split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_state_dir.mkdir(parents=True, exist_ok=True)
        self.initializer_state_dir.mkdir(parents=True, exist_ok=True)
        self.task_matches = _load_task_matches(self.split)
        requested_episode = _episode_number(first_frame)
        self.current_task_name = self.task_matches.get(requested_episode, EDITOR_TASK_NAME)
        self.first_frame = self._initial_first_frame(first_frame)
        self.render_index = 0
        self.error: str | None = None
        self.scene = None
        self.latest_render_jpeg: bytes | None = None
        self.target_jpeg: bytes | None = None
        self.state: dict[str, Any]
        if not self._load_saved_state(self.first_frame):
            self.state = self._default_state(self.first_frame)

    def _initial_first_frame(self, requested: Path) -> Path:
        episode = _episode_number(requested)
        if episode is None or not self._is_episode_done(episode):
            return requested

        episodes = self._episodes_for_current_task()
        for candidate in episodes:
            if candidate >= episode and not self._is_episode_done(candidate):
                return self._episode_frame_from_base(requested, candidate)
        for candidate in episodes:
            if not self._is_episode_done(candidate):
                return self._episode_frame_from_base(requested, candidate)
        return requested

    @staticmethod
    def _episode_frame_from_base(base: Path, episode: int) -> Path:
        return base.with_name(f"episode{episode}.png")

    def _default_state(self, first_frame: Path) -> dict[str, Any]:
        state = {
            "task": self.current_task_name,
            "episode": _episode_name(first_frame),
            "first_frame_path": str(first_frame),
            "rendering": {
                "mode": "interactive",
                "shader_request": "rt",
                "display_scale": self.display_scale,
                "jpeg_quality": self.jpeg_quality,
                "transport": "memory-jpeg",
            },
            "objects": {
                "cup": {
                    "name": "cup",
                    "modelname": "021_cup",
                    "model_id": 0,
                    "table_xy": [0.22, -0.08],
                    "z": 0.741,
                    "yaw_deg": 0.0,
                },
                "coaster": {
                    "name": "coaster",
                    "modelname": "019_coaster",
                    "model_id": 0,
                    "table_xy": [0.02, -0.08],
                    "z": 0.741,
                    "yaw_deg": 0.0,
                },
            },
            "camera": {
                "name": "head_camera",
                "position": list(BASE_CAMERA_POSITION),
                "base_position": list(BASE_CAMERA_POSITION),
                "pitch_deg": 0.0,
                "fovy_deg": 37.0,
            },
            "edit_status": "initializer" if self.auto_initialize else "editing",
        }
        self._apply_rough_initializer(state, first_frame)
        return state

    def _apply_rough_initializer(self, state: dict[str, Any], first_frame: Path) -> None:
        if not self.auto_initialize:
            state["initialization"] = {
                "enabled": False,
                "ok": False,
                "source": "disabled",
                "notes": ["Automatic rough initialization is disabled."],
            }
            return
        try:
            from SceneRecon2.initializers import PlaceEmptyCupInitializer

            initializer = PlaceEmptyCupInitializer(self.output_dir / self.split)
            result = initializer.initialize(first_frame)
            for name, obj in result.objects.items():
                if name in state["objects"]:
                    state_obj = state["objects"][name]
                    state_obj["modelname"] = obj.get("modelname", state_obj.get("modelname"))
                    state_obj["model_id"] = int(obj.get("model_id", state_obj.get("model_id", 0)))
                    state_obj["table_xy"] = list(obj.get("table_xy", state_obj.get("table_xy")))
                    state_obj["z"] = float(obj.get("z", state_obj.get("z", 0.741)))
                    state_obj["yaw_deg"] = float(obj.get("yaw_deg", state_obj.get("yaw_deg", 0.0)))
                    state_obj["initializer"] = obj.get("initializer")
            state["initialization"] = {
                "enabled": True,
                "ok": result.ok,
                "source": result.source,
                "notes": result.notes,
                "debug_json_path": result.debug_json_path,
                "debug_image_path": result.debug_image_path,
            }
        except Exception as exc:
            state["initialization"] = {
                "enabled": True,
                "ok": False,
                "source": "fallback_default",
                "error": repr(exc),
                "notes": ["Automatic rough initialization failed; using rough default poses."],
            }

    def initialize(self) -> None:
        if self.scene is not None:
            return

        old_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
        os.environ["ROBOTWIN_RT_DENOISER"] = ""
        try:
            if self._can_render_static_place_empty_cup():
                self.scene = _StaticPlaceEmptyCupScene(self._renderer_params(), "aloha-agilex")
            else:
                self.scene = _StaticGenericObjectsScene(self.state, "aloha-agilex")
            self.state["rendering"]["shader_actual"] = self.scene.shader_mode
            self.render()
        finally:
            if old_denoiser is None:
                os.environ.pop("ROBOTWIN_RT_DENOISER", None)
            else:
                os.environ["ROBOTWIN_RT_DENOISER"] = old_denoiser

    def _renderer_params(self) -> dict[str, Any]:
        if not self._can_render_static_place_empty_cup():
            raise RuntimeError("Static renderer currently supports place_empty_cup objects only.")
        return {
            "cup": {
                "name": "cup",
                "modelname": "021_cup",
                "model_id": int(self.state["objects"]["cup"]["model_id"]),
                "table_xy": list(self.state["objects"]["cup"]["table_xy"]),
                "yaw_deg": float(self.state["objects"]["cup"].get("yaw_deg", 0.0)),
            },
            "coaster": {
                "name": "coaster",
                "modelname": "019_coaster",
                "model_id": int(self.state["objects"]["coaster"]["model_id"]),
                "table_xy": list(self.state["objects"]["coaster"]["table_xy"]),
                "yaw_deg": float(self.state["objects"]["coaster"].get("yaw_deg", 0.0)),
            },
        }

    def _can_render_static_place_empty_cup(self) -> bool:
        objects = self.state.get("objects", {})
        return "cup" in objects and "coaster" in objects

    def _scene_matches_state(self) -> bool:
        if self.scene is None:
            return False
        if self._can_render_static_place_empty_cup():
            return (
                isinstance(self.scene, _StaticPlaceEmptyCupScene)
                and getattr(self.scene, "geometry_signature", None) == _geometry_signature(self._renderer_params())
            )
        return (
            isinstance(self.scene, _StaticGenericObjectsScene)
            and getattr(self.scene, "geometry_signature", None) == _geometry_signature(self.state.get("objects", {}))
        )

    def render(self) -> Path:
        if self.scene is None:
            raise RuntimeError("Scene is not initialized")
        self._apply_camera()
        self.render_index += 1
        if self._can_render_static_place_empty_cup() and isinstance(self.scene, _StaticPlaceEmptyCupScene):
            rgb = self.scene.render_array(self._renderer_params())
        else:
            rgb = self.scene.render_array(self.state)
        self.latest_render_jpeg = self._image_to_jpeg(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
        self.error = None
        return self.output_dir / "latest.jpg"

    def _image_to_jpeg(self, image: Image.Image) -> bytes:
        image = image.convert("RGB")
        if self.display_scale != 1.0:
            width = max(1, int(round(image.width * self.display_scale)))
            height = max(1, int(round(image.height * self.display_scale)))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=False)
        return buffer.getvalue()

    def target_image_bytes(self) -> bytes:
        if self.target_jpeg is None:
            self.target_jpeg = self._image_to_jpeg(Image.open(self.first_frame))
        return self.target_jpeg

    def save_state(self, *, manual: bool) -> Path:
        state_dir = self.state_dir if manual else self.auto_state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._state_path_for(self.first_frame, state_dir=state_dir)
        payload = {
            "saved_at_unix": time.time(),
            "save_kind": "manual" if manual else "auto_draft",
            "first_frame_path": str(self.first_frame),
            "state": copy.deepcopy(self.state),
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        return out_path

    def _state_path_for(self, first_frame: Path, *, state_dir: Path | None = None) -> Path:
        state_dir = self.state_dir if state_dir is None else state_dir
        episode_number = _episode_number(first_frame)
        if episode_number is not None:
            return _state_path_for_episode(state_dir, episode_number, self.current_task_name)
        return state_dir / f"{_episode_name(first_frame)}.{self.current_task_name}.json"

    def _load_saved_state(self, first_frame: Path) -> bool:
        state_path = self._state_path_for(first_frame, state_dir=self.state_dir)
        save_kind = "manual"
        if not state_path.exists():
            state_path = self._state_path_for(first_frame, state_dir=self.auto_state_dir)
            save_kind = "auto_draft"
        if not state_path.exists():
            state_path = self._state_path_for(first_frame, state_dir=self.initializer_state_dir)
            save_kind = "auto_initializer"
        if not state_path.exists():
            return False
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        save_kind = str(payload.get("save_kind") or save_kind)
        loaded_state = copy.deepcopy(payload.get("state", payload))
        loaded_state["task"] = self.current_task_name
        loaded_state["episode"] = _episode_name(first_frame)
        loaded_state["first_frame_path"] = str(first_frame)
        loaded_state.setdefault("rendering", {})
        loaded_state["rendering"].update(
            {
                "mode": "interactive",
                "shader_request": "rt",
                "display_scale": self.display_scale,
                "jpeg_quality": self.jpeg_quality,
                "transport": "memory-jpeg",
            }
        )
        loaded_state["loaded_save"] = {"kind": save_kind, "path": str(state_path)}
        loaded_state["edit_status"] = self._edit_status_from_save_kind(save_kind)
        for obj in loaded_state.get("objects", {}).values():
            if isinstance(obj, dict):
                obj.setdefault("yaw_deg", 0.0)
        self.state = loaded_state
        return True

    @staticmethod
    def _edit_status_from_save_kind(save_kind: str | None) -> str:
        if save_kind == "manual":
            return "manual_saved"
        if save_kind == "auto_initializer":
            return "initializer"
        return "editing"

    def mark_editing(self) -> None:
        self.state["edit_status"] = "editing"

    def _episodes_for_current_task(self) -> list[int]:
        return sorted(
            episode for episode, task_name in self.task_matches.items() if task_name == self.current_task_name
        )

    def _episode_frame(self, episode: int) -> Path:
        return self.first_frame.with_name(f"episode{episode}.png")

    def _is_episode_done(self, episode: int) -> bool:
        return _state_path_for_episode(self.state_dir, episode, self.current_task_name).exists()

    def _episode_frame_for_number(self, episode: int) -> Path:
        return self.first_frame.with_name(f"episode{episode}.png")

    def load_episode(self, task_name: str, episode: int, *, save_draft: bool = True) -> dict[str, Any]:
        if episode not in self.task_matches:
            raise FileNotFoundError(f"Unknown episode: {episode}")
        mapped_task = self.task_matches[episode]
        if mapped_task != task_name:
            raise ValueError(f"Episode {episode} belongs to {mapped_task!r}, not {task_name!r}")
        if save_draft and self.state.get("edit_status") == "editing":
            self.save_state(manual=False)

        next_frame = self._episode_frame_for_number(episode)
        if not next_frame.exists():
            raise FileNotFoundError(f"Target first frame does not exist: {next_frame}")

        self.current_task_name = task_name
        self.first_frame = next_frame
        self.target_jpeg = None
        self.latest_render_jpeg = None
        if not self._load_saved_state(next_frame):
            self.state = self._default_state(next_frame)
        if not self._scene_matches_state():
            self.scene = None
            self.initialize()
        else:
            self.render()
        return self.payload()

    def task_payload(self, task_name: str, *, kind: str = "manual") -> dict[str, Any]:
        episodes = sorted(episode for episode, mapped_task in self.task_matches.items() if mapped_task == task_name)
        if not episodes:
            raise FileNotFoundError(f"No mapped episodes for task {task_name!r} in {self.split}")
        items = []
        for episode in episodes:
            manual_done = _state_path_for_episode(self.state_dir, episode, task_name).exists()
            initializer_done = _state_path_for_episode(self.initializer_state_dir, episode, task_name).exists()
            items.append(
                {
                    "episode": episode,
                    "task": task_name,
                    "done": manual_done,
                    "manual_done": manual_done,
                    "initializer_done": initializer_done,
                    "entry_url": f"/task/{quote(task_name)}/{episode}",
                    "image_url": f"/file?kind=episode&episode={episode}",
                }
            )
        unfinished = [item for item in items if not item["done"]]
        finished = [item for item in items if item["done"]]
        return {
            "kind": "manual",
            "split": self.split,
            "task": task_name,
            "total": len(items),
            "done": len(finished),
            "percent": (len(finished) / len(items) * 100.0) if items else 0.0,
            "unfinished": unfinished,
            "finished": finished,
        }

    def _find_neighbor_episode(self, delta: int) -> tuple[int, bool]:
        current_number = _episode_number(self.first_frame)
        if current_number is None:
            raise ValueError(f"Cannot switch from non-numeric episode path: {self.first_frame}")

        episodes = self._episodes_for_current_task()
        if not episodes:
            raise FileNotFoundError(f"No mapped episodes for task {self.current_task_name!r} in {self.split}")
        if current_number not in episodes:
            episodes.append(current_number)
            episodes.sort()

        index = episodes.index(current_number)
        scan = range(index + delta, len(episodes), 1) if delta > 0 else range(index + delta, -1, -1)
        first_done_episode: int | None = None
        for candidate_index in scan:
            candidate = episodes[candidate_index]
            if first_done_episode is None:
                first_done_episode = candidate
            if not self._is_episode_done(candidate):
                return candidate, True
        if first_done_episode is not None:
            return first_done_episode, False
        raise FileNotFoundError(
            f"No {'next' if delta > 0 else 'previous'} episode for task {self.current_task_name!r}"
        )

    def switch_episode(self, delta: int) -> dict[str, Any]:
        saved_previous_path = None
        if self.state.get("edit_status") == "editing":
            saved_previous_path = self.save_state(manual=False)
        next_number, skipped_done = self._find_neighbor_episode(delta)
        next_frame = self._episode_frame(next_number)
        if not next_frame.exists():
            raise FileNotFoundError(f"Target first frame does not exist: {next_frame}")

        self.first_frame = next_frame
        self.target_jpeg = None
        self.latest_render_jpeg = None
        loaded_saved_state = self._load_saved_state(next_frame)
        if not loaded_saved_state:
            self.state = self._default_state(next_frame)
        if not self._scene_matches_state():
            self.scene = None
            self.initialize()
        else:
            self.state["rendering"]["shader_actual"] = self.scene.shader_mode
            self.render()

        result = self.payload()
        result.update(
            {
                "ok": True,
                "episode": _episode_name(next_frame),
                "episode_number": next_number,
                "loaded_saved_state": loaded_saved_state,
                "loaded_save_kind": self.state.get("loaded_save", {}).get("kind"),
                "skipped_done": skipped_done,
                "saved_previous_path": None if saved_previous_path is None else str(saved_previous_path),
            }
        )
        return result

    def reinitialize_current_episode(self) -> None:
        self.state = self._default_state(self.first_frame)
        if not self._scene_matches_state():
            self.scene = None
            self.initialize()
        else:
            self.state["rendering"]["shader_actual"] = self.scene.shader_mode
            self.render()

    def _progress_payload_for_dir(self, progress_dir: Path, *, kind: str) -> dict[str, Any]:
        tasks: dict[str, dict[str, Any]] = {}
        for task_name in _all_robotwin_tasks():
            tasks[task_name] = {
                "task": task_name,
                "total": 0,
                "done": 0,
                "episodes": [],
                "done_episodes": [],
                "next_unfinished": None,
            }
        for episode, task_name in sorted(self.task_matches.items()):
            item = tasks.setdefault(
                task_name,
                {
                    "task": task_name,
                    "total": 0,
                    "done": 0,
                    "episodes": [],
                    "done_episodes": [],
                    "next_unfinished": None,
                },
            )
            item["total"] += 1
            item["episodes"].append(episode)
            if _state_path_for_episode(progress_dir, episode, task_name).exists():
                item["done"] += 1
                item["done_episodes"].append(episode)
            elif item["next_unfinished"] is None:
                item["next_unfinished"] = episode

        rows = []
        for item in tasks.values():
            total = int(item["total"])
            done = int(item["done"])
            item["percent"] = (done / total * 100.0) if total else 0.0
            rows.append(item)
        rows.sort(key=lambda row: (row["done"] == row["total"], row["task"]))
        total = sum(int(row["total"]) for row in rows)
        done = sum(int(row["done"]) for row in rows)
        return {
            "kind": kind,
            "split": self.split,
            "task_count": len(rows),
            "total": total,
            "done": done,
            "percent": (done / total * 100.0) if total else 0.0,
            "tasks": rows,
            "progress_dir": str(progress_dir),
            "state_dir": str(self.state_dir),
            "auto_state_dir": str(self.auto_state_dir),
            "initializer_state_dir": str(self.initializer_state_dir),
        }

    def progress_payload(self) -> dict[str, Any]:
        return self._progress_payload_for_dir(self.state_dir, kind="manual")

    def initializer_progress_payload(self) -> dict[str, Any]:
        return self._progress_payload_for_dir(self.initializer_state_dir, kind="initializer")

    def _apply_camera(self) -> None:
        if self.scene is None:
            return
        cameras = self.scene.task.cameras
        camera_id = cameras.head_camera_id
        if camera_id is None:
            return
        camera = cameras.static_camera_list[camera_id]
        sapien = self.scene.sapien
        cam_state = self.state["camera"]
        position = np.asarray(cam_state["position"], dtype=np.float64)
        forward = np.asarray(BASE_CAMERA_FORWARD, dtype=np.float64)
        forward = forward / np.linalg.norm(forward)
        left = np.asarray(BASE_CAMERA_LEFT, dtype=np.float64)
        left = left / np.linalg.norm(left)
        pitch = np.deg2rad(float(cam_state.get("pitch_deg", 0.0)))
        if abs(pitch) > 1e-9:
            forward = self._rotate_around_axis(forward, left, pitch)
        up = np.cross(forward, left)
        up = up / np.linalg.norm(up)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = position
        camera.entity.set_pose(sapien.Pose(mat44))

    @staticmethod
    def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
        )

    def update_object(self, object_name: str, updates: dict[str, Any]) -> None:
        if object_name not in self.state["objects"]:
            raise ValueError(f"Unknown object: {object_name}")
        obj = self.state["objects"][object_name]
        if "x" in updates:
            obj["table_xy"][0] = float(updates["x"])
        if "y" in updates:
            obj["table_xy"][1] = float(updates["y"])
        if "dx" in updates:
            obj["table_xy"][0] = float(obj["table_xy"][0]) + float(updates["dx"])
        if "dy" in updates:
            obj["table_xy"][1] = float(obj["table_xy"][1]) + float(updates["dy"])
        if "yaw_deg" in updates:
            obj["yaw_deg"] = float(updates["yaw_deg"])
        if "dyaw_deg" in updates:
            obj["yaw_deg"] = float(obj.get("yaw_deg", 0.0)) + float(updates["dyaw_deg"])
        if "model_id" in updates:
            obj["model_id"] = int(updates["model_id"])
        if "dmodel_id" in updates:
            obj["model_id"] = _shift_model_id(
                obj.get("modelname"),
                int(obj.get("model_id", 0)),
                int(updates["dmodel_id"]),
            )
        obj["table_xy"][0] = max(-0.35, min(0.35, float(obj["table_xy"][0])))
        obj["table_xy"][1] = max(-0.25, min(0.12, float(obj["table_xy"][1])))
        obj["yaw_deg"] = _wrap_degrees(float(obj.get("yaw_deg", 0.0)))
        if "model_id" in obj:
            valid_ids = _model_ids(obj.get("modelname"))
            if valid_ids:
                current_id = int(obj.get("model_id", 0))
                obj["model_id"] = current_id if current_id in valid_ids else _shift_model_id(obj.get("modelname"), current_id, 0)
            else:
                obj["model_id"] = max(0, int(obj.get("model_id", 0)))

    def update_camera(self, updates: dict[str, Any]) -> None:
        camera = self.state["camera"]
        position = camera["position"]
        if "y" in updates:
            position[1] = float(updates["y"])
        if "z" in updates:
            position[2] = float(updates["z"])
        if "dy" in updates:
            position[1] = float(position[1]) + float(updates["dy"])
        if "dz" in updates:
            position[2] = float(position[2]) + float(updates["dz"])
        if "pitch_deg" in updates:
            camera["pitch_deg"] = float(updates["pitch_deg"])
        if "dpitch_deg" in updates:
            camera["pitch_deg"] = float(camera["pitch_deg"]) + float(updates["dpitch_deg"])
        position[1] = max(-0.8, min(-0.15, float(position[1])))
        position[2] = max(0.9, min(1.8, float(position[2])))
        camera["pitch_deg"] = max(-25.0, min(25.0, float(camera["pitch_deg"])))

    def payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "first_frame_url": f"/file?kind=target&t={_now_token()}",
            "render_url": f"/file?kind=render&t={_now_token()}",
            "error": self.error,
        }


class _StaticPlaceEmptyCupScene:
    """Editor-only renderer: static actors, no physics step between edits."""

    def __init__(self, initial_params: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from SceneRecon.rendering.sapien_first_frame import _default_kwargs, _table_xy
        _install_editor_curobo_stub()
        from envs.place_empty_cup import place_empty_cup
        from envs.utils import create_actor

        self.geometry_signature = _geometry_signature(initial_params)
        cup_xy = _table_xy(initial_params, "cup")
        coaster_xy = _table_xy(initial_params, "coaster")
        if cup_xy is None or coaster_xy is None:
            raise ValueError("place_empty_cup editor render needs cup.table_xy and coaster.table_xy")

        class FixedPlaceEmptyCup(place_empty_cup):
            def load_actors(self_inner):
                self_inner.cup = create_actor(
                    self_inner,
                    pose=sapien.Pose([cup_xy[0], cup_xy[1], 0.741], _object_qpos(initial_params, "cup")),
                    modelname="021_cup",
                    convex=True,
                    model_id=int(initial_params.get("cup", {}).get("model_id", 0)),
                    is_static=True,
                )
                self_inner.coaster = create_actor(
                    self_inner,
                    pose=sapien.Pose([coaster_xy[0], coaster_xy[1], 0.741], _object_qpos(initial_params, "coaster")),
                    modelname="019_coaster",
                    convex=True,
                    model_id=int(initial_params.get("coaster", {}).get("model_id", 0)),
                    is_static=True,
                )
                self_inner.add_prohibit_area(self_inner.cup, padding=0.05)
                self_inner.add_prohibit_area(self_inner.coaster, padding=0.05)

        self.sapien = sapien
        self._table_xy = _table_xy
        self.task = FixedPlaceEmptyCup()
        self.shader_mode = "unknown"
        kwargs = _default_kwargs("place_empty_cup", embodiment)
        kwargs["need_plan"] = False
        self.task.setup_demo(**kwargs)
        self.shader_mode = "rt"

    def render_array(self, params: dict[str, Any]) -> np.ndarray:
        cup_xy = self._table_xy(params, "cup")
        coaster_xy = self._table_xy(params, "coaster")
        if cup_xy is None or coaster_xy is None:
            raise ValueError("place_empty_cup editor render needs cup.table_xy and coaster.table_xy")

        self.task.cup.actor.set_pose(self.sapien.Pose([cup_xy[0], cup_xy[1], 0.741], _object_qpos(params, "cup")))
        self.task.coaster.actor.set_pose(
            self.sapien.Pose([coaster_xy[0], coaster_xy[1], 0.741], _object_qpos(params, "coaster"))
        )
        self.task._update_render()
        self.task.cameras.update_picture()
        return self.task.cameras.get_rgb()["head_camera"]["rgb"]

    def render(self, params: dict[str, Any], out_path: Path) -> Path:
        rgb = self.render_array(params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out_path)
        return out_path


class _StaticGenericObjectsScene:
    """Static renderer for editor states made of model actors and simple boxes."""

    def __init__(self, state: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from SceneRecon.rendering.sapien_first_frame import _default_kwargs
        _install_editor_curobo_stub()
        from envs._base_task import Base_Task
        from envs.utils import create_actor, create_box

        initial_state = copy.deepcopy(state)
        self.geometry_signature = _geometry_signature(initial_state.get("objects", {}))

        class FixedGenericTask(Base_Task):
            def setup_demo(self_inner, **kwargs):
                super()._init_task_env_(**kwargs)

            def load_actors(self_inner):
                self_inner.editor_actors = {}
                for name, obj in initial_state.get("objects", {}).items():
                    pose = _object_pose_from_state(sapien, obj)
                    if obj.get("kind") == "box":
                        color = obj.get("color")
                        if not isinstance(color, list):
                            color = [0.5, 0.5, 0.5]
                        actor = create_box(
                            scene=self_inner,
                            pose=pose,
                            half_size=obj.get("half_size", [0.025, 0.025, 0.025]),
                            color=color,
                            is_static=True,
                            name=name,
                        )
                    elif obj.get("modelname"):
                        actor = create_actor(
                            scene=self_inner,
                            pose=pose,
                            modelname=str(obj.get("modelname")),
                            convex=True,
                            model_id=int(obj.get("model_id", 0)),
                            is_static=True,
                        )
                    else:
                        continue
                    self_inner.editor_actors[name] = actor

        self.sapien = sapien
        self.task = FixedGenericTask()
        self.shader_mode = "unknown"
        kwargs = _default_kwargs("place_empty_cup", embodiment)
        kwargs["need_plan"] = False
        self.task.setup_demo(**kwargs)
        self.shader_mode = "rt"

    def _actor_entity(self, actor: Any) -> Any:
        return getattr(actor, "actor", actor)

    def render_array(self, state: dict[str, Any]) -> np.ndarray:
        for name, obj in state.get("objects", {}).items():
            actor = getattr(self.task, "editor_actors", {}).get(name)
            if actor is None:
                continue
            self._actor_entity(actor).set_pose(_object_pose_from_state(self.sapien, obj))
        self.task._update_render()
        self.task.cameras.update_picture()
        return self.task.cameras.get_rgb()["head_camera"]["rgb"]

    def render(self, state: dict[str, Any], out_path: Path) -> Path:
        rgb = self.render_array(state)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out_path)
        return out_path


def _object_pose_from_state(sapien: Any, obj: dict[str, Any]) -> Any:
    xy = obj.get("table_xy", [0.0, 0.0])
    z = float(obj.get("z", 0.741))
    return sapien.Pose([float(xy[0]), float(xy[1]), z], _yaw_quat(list(obj.get("qpos", BASE_TABLE_OBJECT_QPOS)), float(obj.get("yaw_deg", 0.0))))


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SceneRecon2 place_empty_cup</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #191b20;
      --panel-2: #20232a;
      --text: #f3f4f6;
      --muted: #a6adbb;
      --line: #343842;
      --accent: #58c4a7;
      --warn: #e7bc65;
      --bad: #ff7d7d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #14161a;
    }
    header h1 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .header-right {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .active-pill {
      min-width: 96px;
      border: 1px solid var(--accent);
      color: var(--text);
      background: rgba(88, 196, 167, 0.14);
      border-radius: 999px;
      padding: 3px 9px;
      text-align: center;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      height: calc(100vh - 52px);
      min-height: 0;
    }
    .stage {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 10px;
      min-width: 0;
      min-height: 0;
    }
    .view {
      min-width: 0;
      min-height: 0;
      background: #050607;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: grid;
      grid-template-rows: 34px minmax(0, 1fr);
      overflow: hidden;
    }
    .view-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .title-left {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .render-badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.04);
      font-size: 11px;
      white-space: nowrap;
    }
    .render-badge.manual_saved {
      border-color: rgba(88, 196, 167, 0.65);
      color: var(--accent);
      background: rgba(88, 196, 167, 0.12);
    }
    .render-badge.initializer {
      border-color: rgba(231, 188, 101, 0.65);
      color: var(--warn);
      background: rgba(231, 188, 101, 0.10);
    }
    .render-badge.editing {
      border-color: rgba(166, 173, 187, 0.55);
      color: var(--text);
      background: rgba(166, 173, 187, 0.10);
    }
    .view img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      min-height: 0;
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
    }
    .object {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 12px;
    }
    .object.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(88, 196, 167, 0.28) inset;
    }
    .object h2 {
      margin: 0 0 8px;
      font-size: 14px;
      display: flex;
      align-items: center;
      gap: 7px;
    }
    .object h2 span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
    }
    label {
      display: grid;
      grid-template-columns: 18px 1fr 64px;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      margin: 8px 0;
    }
    input[type="range"] { width: 100%; }
    input[type="number"] {
      width: 64px;
      background: #111318;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 4px 5px;
      font: inherit;
    }
    .nudges {
      display: grid;
      grid-template-columns: repeat(3, 34px);
      gap: 5px;
      width: max-content;
      margin-top: 8px;
    }
    button {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #252933;
      color: var(--text);
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: var(--accent); }
    .wide { width: 100%; margin: 6px 0; }
    .status {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      white-space: pre-wrap;
    }
    .error { color: var(--bad); }
    .reserved {
      opacity: 0.55;
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .toast {
      position: fixed;
      right: 14px;
      bottom: 14px;
      max-width: min(420px, calc(100vw - 28px));
      padding: 9px 11px;
      border: 1px solid rgba(88, 196, 167, 0.55);
      border-radius: 6px;
      background: rgba(20, 22, 26, 0.95);
      color: var(--text);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
      font-size: 12px;
      opacity: 0;
      pointer-events: none;
      transform: translateY(6px);
      transition: opacity 140ms ease, transform 140ms ease;
      z-index: 20;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
    .toast.error {
      border-color: rgba(255, 125, 125, 0.75);
      color: var(--bad);
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; height: auto; }
      .stage { grid-template-columns: 1fr; height: auto; }
      .view { height: 48vh; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="editorTitle">SceneRecon2</h1>
    <div class="header-right">
      <div class="active-pill" id="activeObjectBadge">Active: cup</div>
      <div id="topStatus">starting</div>
    </div>
  </header>
  <main>
    <section class="stage">
      <div class="view">
        <div class="view-title"><span>Target first frame</span></div>
        <img id="targetImage" alt="target">
      </div>
      <div class="view">
        <div class="view-title">
          <div class="title-left">
            <span id="rightViewTitle">Current render</span>
            <span class="render-badge initializer" id="renderStateBadge">initializer</span>
          </div>
        </div>
        <img id="renderImage" alt="render">
      </div>
    </section>
    <aside>
      <button class="wide" id="saveBtn">Save reconstruction</button>
      <button class="wide" id="progressBtn">Progress overview</button>
      <div id="controls"></div>
      <button class="wide" id="resetBtn">Reset rough initial pose</button>
      <button class="wide" id="initBtn">Re-run rough initializer</button>
      <button class="wide" id="autoBtn">Auto refine selected object</button>
      <div class="status" id="status"></div>
    </aside>
  </main>
  <div class="toast" id="toast"></div>
  <script>
    let objects = ["camera"];
    let state = null;
    let selected = "cup";
    let renderUrl = "";
    let targetUrl = "";
    let compareHeld = false;
    let updateInFlight = false;
    let queuedDelta = null;
    let nudgeMode = "normal";
    let toastTimer = null;

    function fmt(v) { return Number(v).toFixed(3); }
    function episodeLabel(value) {
      return String(value || "").replace(/^episode/i, "");
    }
    function renderStatusLabel(status) {
      if (status === "manual_saved") return "上一次人工保存的";
      if (status === "initializer") return "Initializer跑出来的";
      return "修改中的";
    }
    function applyRenderTitle() {
      const status = state && state.edit_status ? state.edit_status : "editing";
      const badge = document.getElementById("renderStateBadge");
      document.getElementById("rightViewTitle").textContent = compareHeld ? "Target first frame" : "Current render";
      badge.textContent = compareHeld ? "对比中" : renderStatusLabel(status);
      badge.className = `render-badge ${compareHeld ? "" : status}`;
    }

    async function api(path, options) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function showToast(message, kind = "ok") {
      const toast = document.getElementById("toast");
      toast.textContent = message;
      toast.className = `toast show ${kind === "error" ? "error" : ""}`;
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => {
        toast.className = `toast ${kind === "error" ? "error" : ""}`;
      }, 1800);
    }

    function objectControls(name, obj) {
      if (name === "camera") return cameraControls(state.camera);
      const [x, y] = obj.table_xy;
      const yaw = obj.yaw_deg || 0;
      const modelId = Number(obj.model_id || 0);
      const modelName = obj.modelname || obj.kind || "object";
      return `
        <div class="object ${selected === name ? "active" : ""}">
          <h2><input type="radio" name="selected" value="${name}" ${selected === name ? "checked" : ""}> ${name}<span>${objects.indexOf(name) + 1}</span></h2>
          <label>
            <span>model</span>
            <button data-model-delta="${name}:-1">‹</button>
            <input data-object="${name}" data-field="model_id" type="number" step="1" min="0" value="${modelId}">
            <button data-model-delta="${name}:1">›</button>
          </label>
          <div class="reserved">${modelName} #${modelId}; ,/. switch model</div>
          <label>
            <span>x</span>
            <input data-object="${name}" data-field="x" type="range" min="-0.35" max="0.35" step="0.002" value="${x}">
            <input data-object="${name}" data-field="x" type="number" step="0.002" value="${fmt(x)}">
          </label>
          <label>
            <span>y</span>
            <input data-object="${name}" data-field="y" type="range" min="-0.25" max="0.12" step="0.002" value="${y}">
            <input data-object="${name}" data-field="y" type="number" step="0.002" value="${fmt(y)}">
          </label>
          <label>
            <span>yaw</span>
            <input data-object="${name}" data-field="yaw_deg" type="range" min="-180" max="180" step="0.5" value="${yaw}">
            <input data-object="${name}" data-field="yaw_deg" type="number" step="0.5" value="${fmt(yaw)}">
          </label>
          <div class="nudges">
            <span></span><button data-nudge="${name}:0:0.005">↑</button><span></span>
            <button data-nudge="${name}:-0.005:0">←</button><button data-nudge="${name}:0:0">·</button><button data-nudge="${name}:0.005:0">→</button>
            <span></span><button data-nudge="${name}:0:-0.005">↓</button><span></span>
          </div>
          <div class="reserved">Arrows/WASD move; Z fine; X coarse; Q/E yaw</div>
        </div>`;
    }

    function cameraControls(camera) {
      const y = camera.position[1];
      const z = camera.position[2];
      const pitch = camera.pitch_deg;
      return `
        <div class="object ${selected === "camera" ? "active" : ""}">
          <h2><input type="radio" name="selected" value="camera" ${selected === "camera" ? "checked" : ""}> camera<span>C</span></h2>
          <label>
            <span>y</span>
            <input data-camera-field="y" type="range" min="-0.8" max="-0.15" step="0.002" value="${y}">
            <input data-camera-field="y" type="number" step="0.002" value="${fmt(y)}">
          </label>
          <label>
            <span>z</span>
            <input data-camera-field="z" type="range" min="0.9" max="1.8" step="0.002" value="${z}">
            <input data-camera-field="z" type="number" step="0.002" value="${fmt(z)}">
          </label>
          <label>
            <span>p</span>
            <input data-camera-field="pitch_deg" type="range" min="-25" max="25" step="0.1" value="${pitch}">
            <input data-camera-field="pitch_deg" type="number" step="0.1" value="${fmt(pitch)}">
          </label>
          <div class="reserved">camera arrows/WASD: left/right=y, up/down=z; Q/E=pitch</div>
        </div>`;
    }

    function setSelected(name) {
      if (!objects.includes(name)) return;
      selected = name;
      document.getElementById("activeObjectBadge").textContent = `Active: ${selected}`;
      if (state) renderControls();
    }

    function renderControls() {
      const controls = document.getElementById("controls");
      controls.innerHTML = objects.map(name => objectControls(name, state.objects[name])).join("");
      controls.querySelectorAll('input[name="selected"]').forEach(input => {
        input.addEventListener("change", () => setSelected(input.value));
      });
      controls.querySelectorAll('input[data-field]').forEach(input => {
        input.addEventListener("change", () => updateObject(input.dataset.object, {[input.dataset.field]: Number(input.value)}));
      });
      controls.querySelectorAll('input[data-camera-field]').forEach(input => {
        input.addEventListener("change", () => updateCamera({[input.dataset.cameraField]: Number(input.value)}));
      });
      controls.querySelectorAll('button[data-nudge]').forEach(button => {
        button.addEventListener("click", () => {
          const [name, dx, dy] = button.dataset.nudge.split(":");
          updateObject(name, {dx: Number(dx), dy: Number(dy)});
        });
      });
      controls.querySelectorAll('button[data-model-delta]').forEach(button => {
        button.addEventListener("click", () => {
          const [name, delta] = button.dataset.modelDelta.split(":");
          updateObject(name, {dmodel_id: Number(delta)});
        });
      });
    }

    function renderPayload(data) {
      state = data.state;
      objects = Object.keys(state.objects || {});
      objects.push("camera");
      if (!objects.includes(selected)) selected = objects[0] || "camera";
      targetUrl = data.first_frame_url;
      renderUrl = data.render_url;
      document.getElementById("editorTitle").textContent = `SceneRecon2 / ${state.task} / ${episodeLabel(state.episode)}`;
      document.getElementById("targetImage").src = targetUrl;
      document.getElementById("renderImage").src = compareHeld ? targetUrl : renderUrl;
      applyRenderTitle();
      document.getElementById("topStatus").textContent = `${episodeLabel(state.episode)} ready`;
      document.getElementById("activeObjectBadge").textContent = `Active: ${selected}`;
      document.getElementById("status").textContent = JSON.stringify({
        objects: state.objects,
        initialization: state.initialization || null
      }, null, 2);
      document.getElementById("status").className = data.error ? "status error" : "status";
      renderControls();
    }

    async function refresh() {
      document.getElementById("topStatus").textContent = "loading";
      renderPayload(await api("/api/state"));
    }

    async function updateObject(name, updates) {
      document.getElementById("topStatus").textContent = "rendering";
      const data = await api("/api/update", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({object: name, updates})
      });
      renderPayload(data);
    }

    async function updateCamera(updates) {
      document.getElementById("topStatus").textContent = "rendering";
      const data = await api("/api/camera", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({updates})
      });
      renderPayload(data);
    }

    function compareDown() {
      compareHeld = true;
      if (targetUrl) document.getElementById("renderImage").src = targetUrl;
      applyRenderTitle();
    }

    function compareUp() {
      compareHeld = false;
      if (renderUrl) document.getElementById("renderImage").src = renderUrl;
      applyRenderTitle();
    }

    function queueMove(dx, dy) {
      if (!queuedDelta) queuedDelta = {dx: 0, dy: 0, dyaw_deg: 0};
      queuedDelta.dx = (queuedDelta.dx || 0) + dx;
      queuedDelta.dy = (queuedDelta.dy || 0) + dy;
      drainMoveQueue();
    }

    function queueYaw(dyawDeg) {
      if (!queuedDelta) queuedDelta = {dx: 0, dy: 0, dyaw_deg: 0};
      queuedDelta.dyaw_deg = (queuedDelta.dyaw_deg || 0) + dyawDeg;
      drainMoveQueue();
    }

    function queueCameraMove(dy, dz, dpitchDeg = 0) {
      if (!queuedDelta) queuedDelta = {dy: 0, dz: 0, dpitch_deg: 0};
      queuedDelta.dy = (queuedDelta.dy || 0) + dy;
      queuedDelta.dz = (queuedDelta.dz || 0) + dz;
      queuedDelta.dpitch_deg = (queuedDelta.dpitch_deg || 0) + dpitchDeg;
      drainMoveQueue();
    }

    async function drainMoveQueue() {
      if (updateInFlight || !queuedDelta) return;
      const delta = queuedDelta;
      queuedDelta = null;
      updateInFlight = true;
      try {
        if (selected === "camera") {
          await updateCamera(delta);
        } else {
          await updateObject(selected, delta);
        }
      } finally {
        updateInFlight = false;
        if (queuedDelta) drainMoveQueue();
      }
    }

    function keyboardStep(event) {
      if (nudgeMode === "coarse") return 0.02;
      if (nudgeMode === "fine") return 0.001;
      return 0.005;
    }

    function pitchStep() {
      if (nudgeMode === "coarse") return 2.0;
      if (nudgeMode === "fine") return 0.1;
      return 0.5;
    }

    function yawStep() {
      if (nudgeMode === "coarse") return 10.0;
      if (nudgeMode === "fine") return 0.5;
      return 2.0;
    }

    function shouldIgnoreKeys(event) {
      const tag = event.target && event.target.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    }

    function movementDeltaForKey(code, step) {
      if (code === "ArrowLeft" || code === "KeyA") return [-step, 0];
      if (code === "ArrowRight" || code === "KeyD") return [step, 0];
      if (code === "ArrowUp" || code === "KeyW") return [0, step];
      if (code === "ArrowDown" || code === "KeyS") return [0, -step];
      return null;
    }

    document.getElementById("resetBtn").addEventListener("click", async () => {
      renderPayload(await api("/api/reset", {method: "POST"}));
    });
    document.getElementById("initBtn").addEventListener("click", async () => {
      try {
        document.getElementById("topStatus").textContent = "initializing";
        const data = await api("/api/reinitialize", {method: "POST"});
        renderPayload(data);
        showToast(data.state.initialization && data.state.initialization.ok ? "Initialized rough pose" : "Initializer fell back");
      } catch (err) {
        showToast(err.message, "error");
      }
    });
    document.getElementById("autoBtn").addEventListener("click", async () => {
      try {
        await api("/api/auto-refine", {method: "POST"});
      } catch (err) {
        document.getElementById("status").textContent = err.message;
        document.getElementById("status").className = "status error";
      }
    });
    document.getElementById("saveBtn").addEventListener("click", async () => {
      await saveReconstruction();
    });
    document.getElementById("progressBtn").addEventListener("click", () => {
      window.open("/progress", "_blank");
    });

    async function saveReconstruction() {
      try {
        const data = await api("/api/save", {method: "POST"});
        state = data.state;
        applyRenderTitle();
        showToast(`Saved ${episodeLabel(state.episode)}`);
        document.getElementById("topStatus").textContent = `${episodeLabel(state.episode)} saved`;
      } catch (err) {
        showToast(err.message, "error");
      }
    }

    function showUnimplemented(message) {
      document.getElementById("status").textContent = message;
      document.getElementById("status").className = "status error";
    }

    async function switchEpisode(direction) {
      try {
        document.getElementById("topStatus").textContent = direction === "next" ? "loading next" : "loading previous";
        const data = await api("/api/episode", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({direction})
        });
        renderPayload(data);
        const loaded = data.loaded_save_kind === "manual"
          ? "loaded manual save"
          : data.loaded_save_kind === "auto_initializer"
            ? "loaded initializer"
          : data.loaded_save_kind === "auto_draft"
            ? "loaded draft"
            : "new rough state";
        showToast(`${episodeLabel(data.episode)}: ${loaded}`);
      } catch (err) {
        document.getElementById("topStatus").textContent = "ready";
        showToast(err.message, "error");
      }
    }

    window.addEventListener("keydown", event => {
      if (shouldIgnoreKeys(event)) return;
      if (event.code === "Space") {
        event.preventDefault();
        compareDown();
        return;
      }
      if (/^Digit[1-9]$/.test(event.code)) {
        event.preventDefault();
        const index = Number(event.code.slice(5)) - 1;
        if (objects[index]) setSelected(objects[index]);
        return;
      }
      if (event.code === "KeyZ") {
        event.preventDefault();
        nudgeMode = "fine";
        document.getElementById("topStatus").textContent = "fine";
        return;
      }
      if (event.code === "KeyX") {
        event.preventDefault();
        nudgeMode = "coarse";
        document.getElementById("topStatus").textContent = "coarse";
        return;
      }
      if (event.code === "KeyC") {
        event.preventDefault();
        setSelected("camera");
        return;
      }
      if (event.code === "Comma") {
        event.preventDefault();
        if (selected !== "camera" && !event.repeat) updateObject(selected, {dmodel_id: -1});
        return;
      }
      if (event.code === "Period") {
        event.preventDefault();
        if (selected !== "camera" && !event.repeat) updateObject(selected, {dmodel_id: 1});
        return;
      }
      if (event.code === "Enter" || event.code === "NumpadEnter") {
        event.preventDefault();
        if (event.repeat) return;
        saveReconstruction();
        return;
      }
      if (event.code === "Equal" || event.code === "NumpadAdd" || event.key === "+" || event.key === "=") {
        event.preventDefault();
        if (event.repeat) return;
        switchEpisode("next");
        return;
      }
      if (event.code === "Minus" || event.code === "NumpadSubtract" || event.key === "-") {
        event.preventDefault();
        if (event.repeat) return;
        switchEpisode("previous");
        return;
      }
      if (event.code === "Tab") {
        event.preventDefault();
        const index = objects.indexOf(selected);
        setSelected(objects[(index + 1) % objects.length]);
        return;
      }
      const step = keyboardStep(event);
      const cameraPitchStep = pitchStep();
      const objectYawStep = yawStep();
      const movementDelta = movementDeltaForKey(event.code, step);
      if (selected === "camera" && movementDelta) {
        event.preventDefault();
        queueCameraMove(movementDelta[0], movementDelta[1]);
      } else if (selected === "camera" && event.code === "KeyQ") {
        event.preventDefault();
        queueCameraMove(0, 0, -cameraPitchStep);
      } else if (selected === "camera" && event.code === "KeyE") {
        event.preventDefault();
        queueCameraMove(0, 0, cameraPitchStep);
      } else if (event.code === "KeyQ") {
        event.preventDefault();
        queueYaw(-objectYawStep);
      } else if (event.code === "KeyE") {
        event.preventDefault();
        queueYaw(objectYawStep);
      } else if (movementDelta) {
        event.preventDefault();
        queueMove(movementDelta[0], movementDelta[1]);
      } else if (["KeyR", "KeyF"].includes(event.code)) {
        event.preventDefault();
        document.getElementById("status").textContent = "Roll, pitch, and scale controls are reserved but not implemented in this MVP.";
        document.getElementById("status").className = "status error";
      }
    });
    window.addEventListener("keyup", event => {
      if (event.code === "Space") {
        event.preventDefault();
        compareUp();
      } else if (event.code === "KeyZ" || event.code === "KeyX") {
        event.preventDefault();
        nudgeMode = "normal";
        document.getElementById("topStatus").textContent = "ready";
      }
    });
    refresh().catch(err => {
      document.getElementById("topStatus").textContent = "error";
      document.getElementById("status").textContent = err.stack || String(err);
      document.getElementById("status").className = "status error";
    });
  </script>
</body>
</html>
"""


PROGRESS_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SceneRecon2 progress</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #191b20;
      --panel-2: #20232a;
      --text: #f3f4f6;
      --muted: #a6adbb;
      --line: #343842;
      --accent: #58c4a7;
      --warn: #e7bc65;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #14161a;
    }
    h1 { margin: 0; font-size: 16px; letter-spacing: 0; }
    main { padding: 16px; max-width: 1180px; margin: 0 auto; }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }
    th { color: var(--muted); font-weight: 600; background: #15171c; }
    tr.done td { color: var(--muted); }
    .bar {
      height: 8px;
      width: 160px;
      background: #0d0f13;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
    }
    .fill { height: 100%; background: var(--accent); }
    .next { color: var(--warn); }
    .muted { color: var(--muted); }
    a { color: var(--text); text-decoration: none; }
    a:hover { color: var(--accent); text-decoration: underline; }
    button {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #252933;
      color: var(--text);
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: var(--accent); }
    @media (max-width: 800px) {
      .summary { grid-template-columns: 1fr 1fr; }
      table { font-size: 12px; }
      .bar { width: 90px; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="pageTitle">SceneRecon2 progress</h1>
    <button id="refreshBtn">Refresh</button>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><div class="label">Split</div><div class="value" id="split">-</div></div>
      <div class="metric"><div class="label">Tasks</div><div class="value" id="tasks">-</div></div>
      <div class="metric"><div class="label">Episodes</div><div class="value" id="episodes">-</div></div>
      <div class="metric"><div class="label">Done</div><div class="value" id="done">-</div></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>task</th>
          <th>progress</th>
          <th>done / total</th>
          <th>next unfinished</th>
          <th>done episodes</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    function pct(v) { return `${Number(v).toFixed(1)}%`; }
    async function refresh() {
      const initializerMode = window.location.pathname.includes("initializer");
      const res = await fetch(initializerMode ? "/api/progress_initializer" : "/api/progress");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      document.title = initializerMode ? "SceneRecon2 initializer progress" : "SceneRecon2 progress";
      document.getElementById("pageTitle").textContent =
        initializerMode ? "SceneRecon2 initializer progress" : "SceneRecon2 progress";
      document.getElementById("split").textContent = data.split;
      document.getElementById("tasks").textContent = data.task_count;
      document.getElementById("episodes").textContent = data.total;
      document.getElementById("done").textContent = `${data.done} / ${pct(data.percent)}`;
      document.getElementById("rows").innerHTML = data.tasks.map(row => {
        const done = row.done === row.total;
        const taskUrl = `/task/${encodeURIComponent(row.task)}`;
        const next = row.next_unfinished == null
          ? "complete"
          : `<a href="/task/${encodeURIComponent(row.task)}/${row.next_unfinished}">${row.next_unfinished}</a>`;
        const doneEpisodes = row.done_episodes.length
          ? row.done_episodes.map(ep => `<a href="/task/${encodeURIComponent(row.task)}/${ep}">${ep}</a>`).join(", ")
          : "-";
        return `
          <tr class="${done ? "done" : ""}">
            <td><a href="${taskUrl}">${row.task}</a></td>
            <td><div class="bar"><div class="fill" style="width:${row.percent}%"></div></div></td>
            <td>${row.done} / ${row.total} <span class="muted">${pct(row.percent)}</span></td>
            <td class="${done ? "muted" : "next"}">${next}</td>
            <td class="muted">${doneEpisodes}</td>
          </tr>`;
      }).join("");
    }
    document.getElementById("refreshBtn").addEventListener("click", refresh);
    refresh().catch(err => {
      document.getElementById("rows").innerHTML = `<tr><td colspan="5">${err.stack || String(err)}</td></tr>`;
    });
  </script>
</body>
</html>
"""


TASK_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SceneRecon2 task</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #191b20;
      --panel-2: #20232a;
      --text: #f3f4f6;
      --muted: #a6adbb;
      --line: #343842;
      --accent: #58c4a7;
      --warn: #e7bc65;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #14161a;
    }
    h1 { margin: 0; font-size: 16px; letter-spacing: 0; }
    main { padding: 16px; max-width: 1280px; margin: 0 auto; }
    a { color: var(--text); text-decoration: none; }
    a:hover { color: var(--accent); text-decoration: underline; }
    .summary {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      margin-bottom: 14px;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 6px 10px;
    }
    h2 {
      margin: 18px 0 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 650;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(156px, 1fr));
      gap: 10px;
    }
    .card {
      display: block;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .card:hover { border-color: var(--accent); text-decoration: none; }
    .card img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: #0d0f13;
    }
    .card .meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .card .number { color: var(--text); font-weight: 650; }
    details {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 18px;
      background: var(--panel);
    }
  </style>
</head>
<body>
  <header>
    <h1 id="title">SceneRecon2 task</h1>
    <a id="progressLink" href="/progress">Progress</a>
  </header>
  <main>
    <div class="summary">
      <span class="pill" id="split">split</span>
      <span class="pill" id="kind">kind</span>
      <span class="pill" id="count">count</span>
    </div>
    <h2>Unfinished</h2>
    <section class="grid" id="unfinished"></section>
    <details>
      <summary id="finishedSummary">Completed</summary>
      <section class="grid" id="finished"></section>
    </details>
  </main>
  <script>
    function card(item) {
      const badges = [];
      if (item.manual_done) badges.push("manual");
      if (item.initializer_done) badges.push("init");
      return `
        <a class="card" href="${item.entry_url}">
          <img loading="lazy" src="${item.image_url}" alt="episode ${item.episode}">
          <div class="meta"><span class="number">${item.episode}</span><span>${badges.join(" / ") || "new"}</span></div>
        </a>`;
    }
    async function load() {
      const parts = location.pathname.split("/").filter(Boolean);
      const task = decodeURIComponent(parts[1] || "");
      const params = new URLSearchParams(location.search);
      const kind = params.get("kind") || "manual";
      const res = await fetch(`/api/task?task=${encodeURIComponent(task)}&kind=${encodeURIComponent(kind)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      document.title = `SceneRecon2 ${data.task}`;
      document.getElementById("title").textContent = data.task;
      document.getElementById("progressLink").href = kind === "initializer" ? "/progress_initializer" : "/progress";
      document.getElementById("split").textContent = data.split;
      document.getElementById("kind").textContent = kind;
      document.getElementById("count").textContent = `${data.done} / ${data.total} (${data.percent.toFixed(1)}%)`;
      document.getElementById("unfinished").innerHTML = data.unfinished.length
        ? data.unfinished.map(card).join("")
        : `<div class="empty">No unfinished episodes.</div>`;
      document.getElementById("finishedSummary").textContent = `Completed (${data.finished.length})`;
      document.getElementById("finished").innerHTML = data.finished.map(card).join("");
    }
    load().catch(err => {
      document.getElementById("unfinished").innerHTML = `<div class="empty">${err.stack || String(err)}</div>`;
    });
  </script>
</body>
</html>
"""


class EditorHandler(BaseHTTPRequestHandler):
    session: PlaceEmptyCupSession

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[SceneRecon2] {self.address_string()} - {format % args}")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_progress_html(self) -> None:
        data = PROGRESS_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_task_html(self) -> None:
        data = TASK_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")
            return
        self._send_bytes(path.read_bytes(), "image/png")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            episode_only_match = re.fullmatch(r"/(\d+)", parsed.path)
            if episode_only_match:
                episode = int(episode_only_match.group(1))
                task_name = self.session.task_matches.get(episode)
                if task_name is None:
                    self.send_error(HTTPStatus.NOT_FOUND, f"Unknown episode: {episode}")
                    return
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/task/{quote(task_name)}/{episode}")
                self.end_headers()
                return
            task_entry_match = re.fullmatch(r"/task/([^/]+)/(\d+)", parsed.path)
            if task_entry_match:
                task_name = unquote(task_entry_match.group(1))
                episode = int(task_entry_match.group(2))
                self.session.load_episode(task_name, episode)
                self._send_html()
                return
            task_match = re.fullmatch(r"/task/([^/]+)", parsed.path)
            if task_match:
                self._send_task_html()
                return
            if parsed.path == "/":
                self._send_html()
                return
            if parsed.path == "/progress":
                self._send_progress_html()
                return
            if parsed.path == "/progress_initializer":
                self._send_progress_html()
                return
            if parsed.path == "/api/progress":
                self._send_json(self.session.progress_payload())
                return
            if parsed.path == "/api/progress_initializer":
                self._send_json(self.session.initializer_progress_payload())
                return
            if parsed.path == "/api/task":
                query = parse_qs(parsed.query)
                task_name = unquote(query.get("task", [""])[0])
                kind = str(query.get("kind", ["manual"])[0])
                if kind not in {"manual", "initializer"}:
                    raise ValueError(f"Unknown task progress kind: {kind}")
                self._send_json(self.session.task_payload(task_name, kind=kind))
                return
            if parsed.path == "/api/state":
                self.session.initialize()
                self._send_json(self.session.payload())
                return
            if parsed.path == "/file":
                kind = parse_qs(parsed.query).get("kind", [""])[0]
                if kind == "target":
                    self._send_bytes(self.session.target_image_bytes(), "image/jpeg")
                    return
                if kind == "render":
                    if self.session.latest_render_jpeg is None:
                        self.session.render()
                    self._send_bytes(self.session.latest_render_jpeg or b"", "image/jpeg")
                    return
                if kind == "episode":
                    query = parse_qs(parsed.query)
                    episode = int(query.get("episode", ["0"])[0])
                    self._send_bytes(self.session._image_to_jpeg(Image.open(self.session._episode_frame_for_number(episode))), "image/jpeg")
                    return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.session.error = traceback.format_exc()
            self._send_json({"error": repr(exc), "traceback": self.session.error}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            self.session.initialize()
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
            if parsed.path == "/api/update":
                self.session.update_object(str(payload.get("object")), dict(payload.get("updates", {})))
                self.session.mark_editing()
                if not self.session._scene_matches_state():
                    self.session.scene = None
                    self.session.initialize()
                else:
                    self.session.render()
                self._send_json(self.session.payload())
                return
            if parsed.path == "/api/camera":
                self.session.update_camera(dict(payload.get("updates", {})))
                self.session.mark_editing()
                self.session.render()
                self._send_json(self.session.payload())
                return
            if parsed.path == "/api/reset":
                self.session.state["objects"]["cup"]["table_xy"] = [0.22, -0.08]
                self.session.state["objects"]["coaster"]["table_xy"] = [0.02, -0.08]
                self.session.state["objects"]["cup"]["yaw_deg"] = 0.0
                self.session.state["objects"]["coaster"]["yaw_deg"] = 0.0
                self.session.state["camera"]["position"] = list(BASE_CAMERA_POSITION)
                self.session.state["camera"]["pitch_deg"] = 0.0
                self.session.mark_editing()
                self.session.render()
                self._send_json(self.session.payload())
                return
            if parsed.path == "/api/reinitialize":
                self.session.reinitialize_current_episode()
                self.session.mark_editing()
                self._send_json(self.session.payload())
                return
            if parsed.path == "/api/save":
                path = self.session.save_state(manual=True)
                self.session.state["loaded_save"] = {"kind": "manual", "path": str(path)}
                self.session.state["edit_status"] = "manual_saved"
                self._send_json({"ok": True, "path": str(path), "state": self.session.state})
                return
            if parsed.path == "/api/episode":
                direction = str(payload.get("direction", "next"))
                if direction not in {"next", "previous"}:
                    raise ValueError(f"Unknown episode direction: {direction}")
                self._send_json(self.session.switch_episode(1 if direction == "next" else -1))
                return
            if parsed.path == "/api/auto-refine":
                self._send_json({"error": "Unimplemented in the vertical MVP"}, HTTPStatus.NOT_IMPLEMENTED)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.session.error = traceback.format_exc()
            self._send_json({"error": repr(exc), "traceback": self.session.error}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(
    host: str,
    port: int,
    first_frame: Path,
    display_scale: float,
    jpeg_quality: int,
    auto_initialize: bool,
) -> None:
    session = PlaceEmptyCupSession(
        first_frame=first_frame,
        display_scale=display_scale,
        jpeg_quality=jpeg_quality,
        auto_initialize=auto_initialize,
    )
    EditorHandler.session = session
    server = ThreadingHTTPServer((host, port), EditorHandler)
    print(f"SceneRecon2 place_empty_cup editor: http://{host}:{port}")
    print(f"Target first frame: {session.first_frame}")
    print(f"Display transport: JPEG quality={jpeg_quality}, scale={display_scale}")
    print(f"Rough initializer: {'enabled' if auto_initialize else 'disabled'}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SceneRecon2 place_empty_cup vertical MVP editor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--first-frame", type=Path, default=_default_first_frame())
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--no-auto-initialize", action="store_true")
    args = parser.parse_args()
    serve(
        args.host,
        args.port,
        args.first_frame,
        display_scale=max(0.1, min(1.0, args.display_scale)),
        jpeg_quality=max(20, min(95, args.jpeg_quality)),
        auto_initialize=not args.no_auto_initialize,
    )


if __name__ == "__main__":
    main()
