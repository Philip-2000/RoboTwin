from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw


def _load_embodiment(embodiment: str) -> tuple[str, dict[str, Any]]:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_file = embodiment_types[embodiment]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return robot_file, embodiment_config


def _default_kwargs(task_name: str, embodiment: str) -> dict[str, Any]:
    robot_file, embodiment_config = _load_embodiment(embodiment)
    return {
        "seed": 0,
        "task_name": task_name,
        "now_ep_num": 0,
        "render_freq": 0,
        "save_path": "./data/scene_recon_sapien",
        "save_freq": 1,
        "data_type": {"rgb": True},
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": False,
        },
        "left_robot_file": robot_file,
        "right_robot_file": robot_file,
        "left_embodiment_config": embodiment_config,
        "right_embodiment_config": embodiment_config,
        "dual_arm_embodied": True,
        "pcd_crop": False,
        "pcd_down_sample_num": 0,
    }


def _table_xy(params: dict[str, Any], object_name: str) -> tuple[float, float] | None:
    item = params.get(object_name)
    if not isinstance(item, dict):
        return None
    value = item.get("table_xy")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _item_xy(item: dict[str, Any]) -> tuple[float, float] | None:
    value = item.get("table_xy")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _iter_target_objects(params: dict[str, Any]):
    for name, value in params.items():
        if isinstance(value, dict) and value.get("bbox_xyxy") is not None:
            yield str(value.get("name", name)), value
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and item.get("bbox_xyxy") is not None:
                    yield str(item.get("name", f"{name}_{index}")), item


def _bbox_center(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def score_rendered_image(task_name: str, target_params: dict[str, Any], rendered_path: str | Path) -> dict[str, Any]:
    from ..detectors.simple_cv import SimpleCVDetector

    detections = SimpleCVDetector(task_name).detect(Path(rendered_path))
    rendered_by_label = {det.label: det for det in detections.detections}
    per_object: list[dict[str, Any]] = []
    scores: list[float] = []

    for name, params in _iter_target_objects(target_params):
        target_bbox = tuple(float(v) for v in params["bbox_xyxy"])
        rendered_det = rendered_by_label.get(name)
        if rendered_det is None:
            # Some strategies use generic names on one side.
            aliases = {
                "cup": ("cup", "mug"),
                "coaster": ("coaster", "mat", "pad"),
                "block": ("red_block", "block"),
                "red_block": ("red_block", "block"),
                "green_block": ("green_block", "block"),
                "blue_block": ("blue_block", "block"),
            }.get(name, (name,))
            for label, det in rendered_by_label.items():
                if any(alias in label for alias in aliases):
                    rendered_det = det
                    break
        if rendered_det is None:
            per_object.append({"name": name, "score": 0.0, "missing_render_detection": True})
            scores.append(0.0)
            continue
        rendered_bbox = tuple(float(v) for v in rendered_det.bbox_xyxy)
        iou = _bbox_iou(target_bbox, rendered_bbox)
        center_error_px = float(np.linalg.norm(_bbox_center(target_bbox) - _bbox_center(rendered_bbox)))
        score = 0.65 * iou + 0.35 * float(np.exp(-center_error_px / 35.0))
        per_object.append(
            {
                "name": name,
                "score": score,
                "iou": iou,
                "center_error_px": center_error_px,
                "target_bbox_xyxy": list(target_bbox),
                "render_bbox_xyxy": list(rendered_bbox),
                "render_label": rendered_det.label,
            }
        )
        scores.append(score)

    return {
        "score": float(np.mean(scores)) if scores else 0.0,
        "renderer": "sapien_first_frame",
        "per_object": per_object,
        "render_detection_count": len(detections.detections),
    }


class SapienFirstFrameRenderer:
    """Task-specific real renderer for fixed candidate scenes.

    Only place_empty_cup is implemented for now. Other tasks should follow the
    same pattern: subclass the RoboTwin task in SceneRecon and override
    load_actors with candidate-controlled object poses.
    """

    def __init__(self, embodiment: str = "aloha-agilex"):
        self.embodiment = embodiment
        self._place_empty_cup_scene: _ReusablePlaceEmptyCupScene | None = None
        self._click_bell_scenes: dict[int, _ReusableClickBellScene] = {}
        self._beat_block_hammer_scene: _ReusableBeatBlockHammerScene | None = None
        self._stack_blocks_three_scene: _ReusableStackBlocksThreeScene | None = None

    def render(self, candidate, out_path: str | Path) -> Path | None:
        if candidate.task_name == "place_empty_cup":
            if self._place_empty_cup_scene is None:
                self._place_empty_cup_scene = _ReusablePlaceEmptyCupScene(candidate.parameters, self.embodiment)
            return self._place_empty_cup_scene.render(candidate.parameters, Path(out_path))
        if candidate.task_name == "click_bell":
            model_id = _model_id(candidate.parameters.get("bell", {}))
            if model_id not in self._click_bell_scenes:
                self._click_bell_scenes[model_id] = _ReusableClickBellScene(candidate.parameters, self.embodiment)
            return self._click_bell_scenes[model_id].render(candidate.parameters, Path(out_path))
        if candidate.task_name == "beat_block_hammer":
            if self._beat_block_hammer_scene is None:
                self._beat_block_hammer_scene = _ReusableBeatBlockHammerScene(candidate.parameters, self.embodiment)
            return self._beat_block_hammer_scene.render(candidate.parameters, Path(out_path))
        if candidate.task_name == "stack_blocks_three":
            if self._stack_blocks_three_scene is None:
                self._stack_blocks_three_scene = _ReusableStackBlocksThreeScene(candidate.parameters, self.embodiment)
            return self._stack_blocks_three_scene.render(candidate.parameters, Path(out_path))
        return None

    def _render_place_empty_cup(self, params: dict[str, Any], out_path: Path) -> Path:
        import sapien.core as sapien
        from envs.place_empty_cup import place_empty_cup
        from envs.utils import create_actor

        cup_xy = _table_xy(params, "cup")
        coaster_xy = _table_xy(params, "coaster")
        if cup_xy is None or coaster_xy is None:
            raise ValueError("place_empty_cup SAPIEN render needs cup.table_xy and coaster.table_xy")

        class FixedPlaceEmptyCup(place_empty_cup):
            def load_actors(self_inner):
                self_inner.cup = create_actor(
                    self_inner,
                    pose=sapien.Pose([cup_xy[0], cup_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
                    modelname="021_cup",
                    convex=True,
                    model_id=int(params.get("cup", {}).get("model_id", 0)),
                )
                self_inner.coaster = create_actor(
                    self_inner,
                    pose=sapien.Pose([coaster_xy[0], coaster_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
                    modelname="019_coaster",
                    convex=True,
                    model_id=int(params.get("coaster", {}).get("model_id", 0)),
                    is_static=True,
                )
                self_inner.add_prohibit_area(self_inner.cup, padding=0.05)
                self_inner.add_prohibit_area(self_inner.coaster, padding=0.05)

        old_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
        os.environ["ROBOTWIN_RT_DENOISER"] = ""
        task = FixedPlaceEmptyCup()
        try:
            task.setup_demo(**_default_kwargs("place_empty_cup", self.embodiment))
            task._update_render()
            task.cameras.update_picture()
            rgb = task.cameras.get_rgb()["head_camera"]["rgb"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out_path)
            return out_path
        finally:
            if old_denoiser is None:
                os.environ.pop("ROBOTWIN_RT_DENOISER", None)
            else:
                os.environ["ROBOTWIN_RT_DENOISER"] = old_denoiser


class _ReusablePlaceEmptyCupScene:
    def __init__(self, initial_params: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from envs.place_empty_cup import place_empty_cup
        from envs.utils import create_actor

        cup_xy = _table_xy(initial_params, "cup")
        coaster_xy = _table_xy(initial_params, "coaster")
        if cup_xy is None or coaster_xy is None:
            raise ValueError("place_empty_cup reusable render needs cup.table_xy and coaster.table_xy")

        class FixedPlaceEmptyCup(place_empty_cup):
            def load_actors(self_inner):
                self_inner.cup = create_actor(
                    self_inner,
                    pose=sapien.Pose([cup_xy[0], cup_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
                    modelname="021_cup",
                    convex=True,
                    model_id=int(initial_params.get("cup", {}).get("model_id", 0)),
                )
                self_inner.coaster = create_actor(
                    self_inner,
                    pose=sapien.Pose([coaster_xy[0], coaster_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
                    modelname="019_coaster",
                    convex=True,
                    model_id=int(initial_params.get("coaster", {}).get("model_id", 0)),
                    is_static=True,
                )
                self_inner.add_prohibit_area(self_inner.cup, padding=0.05)
                self_inner.add_prohibit_area(self_inner.coaster, padding=0.05)

        old_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
        os.environ["ROBOTWIN_RT_DENOISER"] = ""
        try:
            self.task = FixedPlaceEmptyCup()
            self.task.setup_demo(**_default_kwargs("place_empty_cup", embodiment))
        finally:
            if old_denoiser is None:
                os.environ.pop("ROBOTWIN_RT_DENOISER", None)
            else:
                os.environ["ROBOTWIN_RT_DENOISER"] = old_denoiser

        self.sapien = sapien

    def render(self, params: dict[str, Any], out_path: Path) -> Path:
        cup_xy = _table_xy(params, "cup")
        coaster_xy = _table_xy(params, "coaster")
        if cup_xy is None or coaster_xy is None:
            raise ValueError("place_empty_cup SAPIEN render needs cup.table_xy and coaster.table_xy")

        self.task.cup.actor.set_pose(self.sapien.Pose([cup_xy[0], cup_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]))
        self.task.coaster.actor.set_pose(
            self.sapien.Pose([coaster_xy[0], coaster_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5])
        )
        self.task.scene.step()
        self.task._update_render()
        self.task.cameras.update_picture()
        rgb = self.task.cameras.get_rgb()["head_camera"]["rgb"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out_path)
        return out_path


class _ReusableClickBellScene:
    def __init__(self, initial_params: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from envs.click_bell import click_bell
        from envs.utils import create_actor

        bell_xy = _table_xy(initial_params, "bell")
        if bell_xy is None:
            raise ValueError("click_bell reusable render needs bell.table_xy")
        bell_params = initial_params.get("bell", {})
        model_id = _model_id(bell_params)

        class FixedClickBell(click_bell):
            def load_actors(self_inner):
                self_inner.bell_id = model_id
                self_inner.bell = create_actor(
                    scene=self_inner,
                    pose=sapien.Pose([bell_xy[0], bell_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
                    modelname="050_bell",
                    convex=True,
                    model_id=model_id,
                    is_static=True,
                )
                self_inner.add_prohibit_area(self_inner.bell, padding=0.07)
                self_inner.check_arm_function = (
                    self_inner.is_left_gripper_close if self_inner.bell.get_pose().p[0] < 0 else self_inner.is_right_gripper_close
                )

        self.sapien = sapien
        self.task = self._setup(FixedClickBell, "click_bell", embodiment)

    def _setup(self, task_cls, task_name: str, embodiment: str):
        old_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
        os.environ["ROBOTWIN_RT_DENOISER"] = ""
        try:
            task = task_cls()
            task.setup_demo(**_default_kwargs(task_name, embodiment))
            return task
        finally:
            if old_denoiser is None:
                os.environ.pop("ROBOTWIN_RT_DENOISER", None)
            else:
                os.environ["ROBOTWIN_RT_DENOISER"] = old_denoiser

    def render(self, params: dict[str, Any], out_path: Path) -> Path:
        bell_xy = _table_xy(params, "bell")
        if bell_xy is None:
            raise ValueError("click_bell SAPIEN render needs bell.table_xy")
        self.task.bell.actor.set_pose(self.sapien.Pose([bell_xy[0], bell_xy[1], 0.741], [0.5, 0.5, 0.5, 0.5]))
        return _capture_task_rgb(self.task, out_path)


class _ReusableBeatBlockHammerScene:
    def __init__(self, initial_params: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from envs.beat_block_hammer import beat_block_hammer
        from envs.utils import create_actor, create_box

        block_xy = _table_xy(initial_params, "block")
        if block_xy is None:
            raise ValueError("beat_block_hammer reusable render needs block.table_xy")

        class FixedBeatBlockHammer(beat_block_hammer):
            def load_actors(self_inner):
                self_inner.hammer = create_actor(
                    scene=self_inner,
                    pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
                    modelname="020_hammer",
                    convex=True,
                    model_id=0,
                )
                self_inner.block = create_box(
                    scene=self_inner,
                    pose=sapien.Pose([block_xy[0], block_xy[1], 0.76], [1, 0, 0, 0]),
                    half_size=(0.025, 0.025, 0.025),
                    color=(1, 0, 0),
                    name="box",
                    is_static=True,
                )
                self_inner.hammer.set_mass(0.001)
                self_inner.add_prohibit_area(self_inner.hammer, padding=0.10)

        self.sapien = sapien
        self.task = _setup_task(FixedBeatBlockHammer, "beat_block_hammer", embodiment)

    def render(self, params: dict[str, Any], out_path: Path) -> Path:
        block_xy = _table_xy(params, "block")
        if block_xy is None:
            raise ValueError("beat_block_hammer SAPIEN render needs block.table_xy")
        self.task.block.actor.set_pose(self.sapien.Pose([block_xy[0], block_xy[1], 0.76], [1, 0, 0, 0]))
        return _capture_task_rgb(self.task, out_path)


class _ReusableStackBlocksThreeScene:
    def __init__(self, initial_params: dict[str, Any], embodiment: str):
        import sapien.core as sapien
        from envs.stack_blocks_three import stack_blocks_three
        from envs.utils import create_box

        blocks = initial_params.get("blocks", [])
        if not isinstance(blocks, list) or len(blocks) < 3:
            raise ValueError("stack_blocks_three reusable render needs three blocks")
        color_by_name = {
            "red_block": (1, 0, 0),
            "green_block": (0, 1, 0),
            "blue_block": (0, 0, 1),
        }

        class FixedStackBlocksThree(stack_blocks_three):
            def load_actors(self_inner):
                actors = []
                for block in blocks:
                    xy = _item_xy(block)
                    if xy is None:
                        continue
                    name = block.get("name", "block")
                    actor = create_box(
                        scene=self_inner,
                        pose=sapien.Pose([xy[0], xy[1], 0.766], [1, 0, 0, 0]),
                        half_size=(0.025, 0.025, 0.025),
                        color=color_by_name.get(name, (1, 0, 0)),
                        name="box",
                    )
                    actors.append(actor)
                    self_inner.add_prohibit_area(actor, padding=0.05)
                self_inner.block1, self_inner.block2, self_inner.block3 = actors[:3]
                self_inner.block1_target_pose = [0, -0.13, 0.75 + self_inner.table_z_bias, 0, 1, 0, 0]

        self.sapien = sapien
        self.task = _setup_task(FixedStackBlocksThree, "stack_blocks_three", embodiment)

    def render(self, params: dict[str, Any], out_path: Path) -> Path:
        blocks = params.get("blocks", [])
        actors = [self.task.block1, self.task.block2, self.task.block3]
        for block, actor in zip(blocks, actors):
            xy = _item_xy(block)
            if xy is None:
                continue
            actor.actor.set_pose(self.sapien.Pose([xy[0], xy[1], 0.766], [1, 0, 0, 0]))
        return _capture_task_rgb(self.task, out_path)


def _setup_task(task_cls, task_name: str, embodiment: str):
    old_denoiser = os.environ.get("ROBOTWIN_RT_DENOISER")
    os.environ["ROBOTWIN_RT_DENOISER"] = ""
    try:
        task = task_cls()
        task.setup_demo(**_default_kwargs(task_name, embodiment))
        return task
    finally:
        if old_denoiser is None:
            os.environ.pop("ROBOTWIN_RT_DENOISER", None)
        else:
            os.environ["ROBOTWIN_RT_DENOISER"] = old_denoiser


def _capture_task_rgb(task, out_path: Path) -> Path:
    task.scene.step()
    task._update_render()
    task.cameras.update_picture()
    rgb = task.cameras.get_rgb()["head_camera"]["rgb"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out_path)
    return out_path


def _model_id(params: dict[str, Any]) -> int:
    if "model_id" in params:
        return int(params["model_id"])
    candidates = params.get("model_id_candidates")
    if isinstance(candidates, list) and candidates:
        return int(candidates[0])
    return 0


def save_sapien_comparison(target_path: str | Path, render_path: str | Path, out_path: str | Path) -> Path:
    target = Image.open(target_path).convert("RGB")
    render = Image.open(render_path).convert("RGB")
    if render.size != target.size:
        render = render.resize(target.size)
    w, h = target.size
    pad = 18
    title_h = 30
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + title_h + pad * 2), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(target, (pad, title_h + pad))
    canvas.paste(render, (w + pad * 2, title_h + pad))
    draw.text((pad, pad), "target first frame", fill=(20, 20, 20))
    draw.text((w + pad * 2, pad), "SAPIEN fixed candidate", fill=(20, 20, 20))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path
