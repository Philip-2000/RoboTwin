from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import WorldArenaDataset
from .detectors import JsonDetector, SimpleCVDetector
from .detectors.simple_cv import draw_detections, write_detection_json
from .fitting import add_local_search_ranges, fill_projected_bbox_3d, fill_projected_table_xy
from .geometry import load_table_projector
from .optimizers.render_search import RenderSearchOptimizer
from .registry import get_reconstructor
from .task_mapping import TaskTop1Mapping, default_search_gt_path


def _mapping(split: str, path: str | None) -> TaskTop1Mapping:
    return TaskTop1Mapping.from_search_gt(path or default_search_gt_path(split))


def _detections(args: argparse.Namespace, image_path: Path):
    if getattr(args, "detections_json", None) is None:
        return None
    return JsonDetector(args.detections_json).detect(image_path)


def inspect(args: argparse.Namespace) -> None:
    dataset = WorldArenaDataset(args.worldarena_root, args.worldarena_split)
    episode = dataset.episode(args.episode)
    match = _mapping(args.worldarena_split, args.search_gt).get(args.episode)
    reconstructor = get_reconstructor(match.task_name)
    detections = _detections(args, episode.first_frame_path)
    report = reconstructor.reconstruct(
        episode,
        task_match=match,
        detections=detections,
    )
    report = _postprocess_report(report, args)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def reconstruct(args: argparse.Namespace) -> None:
    dataset = WorldArenaDataset(args.worldarena_root, args.worldarena_split)
    match = _mapping(args.worldarena_split, args.search_gt).get(args.episode)
    episode = dataset.episode(args.episode)
    reconstructor = get_reconstructor(match.task_name)
    detections = _detections(args, episode.first_frame_path)
    report = reconstructor.reconstruct(
        episode,
        task_match=match,
        detections=detections,
    )
    report = _postprocess_report(report, args)

    out_dir = Path(args.output_root) / args.worldarena_split / "fixed_scene_task"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode{args.episode}.scene_recon.json"
    out_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


def _postprocess_report(report, args: argparse.Namespace):
    projector = load_table_projector(
        getattr(args, "calibration_json", None),
        use_default=getattr(args, "use_default_calibration", False),
    )
    candidates = fill_projected_table_xy(report.candidates, projector)
    candidates = fill_projected_bbox_3d(candidates, projector)
    if getattr(args, "add_search_ranges", False):
        candidates = add_local_search_ranges(candidates)
    if getattr(args, "run_render_search", False):
        render_search_output_root = getattr(args, "render_search_output_root", None)
        if render_search_output_root is None and getattr(args, "output_root", None) is not None:
            render_search_output_root = str(Path(args.output_root).parent / "render_search")
        candidates = RenderSearchOptimizer(
            projector=projector,
            output_root=render_search_output_root,
            max_attempts=getattr(args, "render_search_max_attempts", 48),
            sapien_check=getattr(args, "render_search_sapien_check", False),
            sapien_optimize=getattr(args, "render_search_sapien_optimize", False),
            sapien_max_attempts=getattr(args, "render_search_sapien_max_attempts", 5),
            sapien_strategy=getattr(args, "render_search_sapien_strategy", "coordinate"),
            coordinate_initial_step=getattr(args, "render_search_coordinate_step", 0.02),
            coordinate_min_step=getattr(args, "render_search_coordinate_min_step", 0.005),
        ).optimize(report.episode, candidates)

    from dataclasses import replace

    metadata = dict(report.metadata)
    metadata["table_projector"] = (
        None
        if projector is None
        else {
            "type": projector.__class__.__name__,
            "calibration_json": getattr(args, "calibration_json", None),
            "use_default_calibration": getattr(args, "use_default_calibration", False),
        }
    )
    return replace(report, candidates=candidates, metadata=metadata)


def make_detection_template(args: argparse.Namespace) -> None:
    dataset = WorldArenaDataset(args.worldarena_root, args.worldarena_split)
    episode = dataset.episode(args.episode)
    template = {
        "image_path": str(episode.first_frame_path),
        "source": "manual",
        "metadata": {
            "split": args.worldarena_split,
            "episode": args.episode,
            "instruction": episode.instruction,
        },
        "detections": [
            {
                "label": "object_name",
                "bbox_xyxy": [0, 0, 10, 10],
                "score": 1.0,
                "attributes": {},
            }
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


def detect(args: argparse.Namespace) -> None:
    dataset = WorldArenaDataset(args.worldarena_root, args.worldarena_split)
    episode = dataset.episode(args.episode)
    match = _mapping(args.worldarena_split, args.search_gt).get(args.episode)
    if args.detector != "simple-cv":
        raise ValueError(f"Unsupported detector for this CLI path: {args.detector}")

    detections = SimpleCVDetector(match.task_name).detect(episode.first_frame_path)
    out_dir = Path(args.output_root) / args.worldarena_split / "fixed_scene_task"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"episode{args.episode}.detections.json"
    debug_path = out_dir / f"episode{args.episode}.detections.jpg"
    write_detection_json(detections, json_path)
    draw_detections(episode.first_frame_path, detections, debug_path)
    print(json_path)
    print(debug_path)


def serve_viewer(args: argparse.Namespace) -> None:
    from .web_viewer import serve

    serve(Path(args.worldarena_root), args.host, args.port)


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldArena/RoboTwin scene reconstruction utilities.")
    sub = parser.add_subparsers(required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--worldarena-root", default="/home/users/liang01.yue/D/WorldArena_Robotwin2.0")
    common.add_argument("--worldarena-split", default="val_dataset")
    common.add_argument("--episode", type=int, required=True)
    common.add_argument("--search-gt", default=None)

    p_inspect = sub.add_parser("inspect", parents=[common])
    p_inspect.add_argument("--detections-json", default=None)
    p_inspect.add_argument("--calibration-json", default=None)
    p_inspect.add_argument("--use-default-calibration", action="store_true")
    p_inspect.add_argument("--add-search-ranges", action="store_true")
    p_inspect.add_argument("--run-render-search", action="store_true")
    p_inspect.add_argument("--render-search-output-root", default=None)
    p_inspect.add_argument("--render-search-max-attempts", type=int, default=48)
    p_inspect.add_argument("--render-search-sapien-check", action="store_true")
    p_inspect.add_argument("--render-search-sapien-optimize", action="store_true")
    p_inspect.add_argument("--render-search-sapien-max-attempts", type=int, default=17)
    p_inspect.add_argument("--render-search-sapien-strategy", choices=["coordinate", "objectwise", "topk"], default="objectwise")
    p_inspect.add_argument("--render-search-coordinate-step", type=float, default=0.02)
    p_inspect.add_argument("--render-search-coordinate-min-step", type=float, default=0.0025)
    p_inspect.set_defaults(func=inspect)

    p_reconstruct = sub.add_parser("reconstruct", parents=[common])
    p_reconstruct.add_argument("--output-root", required=True)
    p_reconstruct.add_argument("--detections-json", default=None)
    p_reconstruct.add_argument("--calibration-json", default=None)
    p_reconstruct.add_argument("--use-default-calibration", action="store_true")
    p_reconstruct.add_argument("--add-search-ranges", action="store_true")
    p_reconstruct.add_argument("--run-render-search", action="store_true")
    p_reconstruct.add_argument("--render-search-output-root", default=None)
    p_reconstruct.add_argument("--render-search-max-attempts", type=int, default=48)
    p_reconstruct.add_argument("--render-search-sapien-check", action="store_true")
    p_reconstruct.add_argument("--render-search-sapien-optimize", action="store_true")
    p_reconstruct.add_argument("--render-search-sapien-max-attempts", type=int, default=17)
    p_reconstruct.add_argument("--render-search-sapien-strategy", choices=["coordinate", "objectwise", "topk"], default="objectwise")
    p_reconstruct.add_argument("--render-search-coordinate-step", type=float, default=0.02)
    p_reconstruct.add_argument("--render-search-coordinate-min-step", type=float, default=0.0025)
    p_reconstruct.set_defaults(func=reconstruct)

    p_template = sub.add_parser("make-detection-template", parents=[common])
    p_template.add_argument("--out", required=True)
    p_template.set_defaults(func=make_detection_template)

    p_detect = sub.add_parser("detect", parents=[common])
    p_detect.add_argument("--detector", default="simple-cv", choices=["simple-cv"])
    p_detect.add_argument("--output-root", required=True)
    p_detect.set_defaults(func=detect)

    p_viewer = sub.add_parser("serve-viewer")
    p_viewer.add_argument("--worldarena-root", default="/home/users/liang01.yue/D/WorldArena_Robotwin2.0")
    p_viewer.add_argument("--host", default="127.0.0.1")
    p_viewer.add_argument("--port", type=int, default=8765)
    p_viewer.set_defaults(func=serve_viewer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
