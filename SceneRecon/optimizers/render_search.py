from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core import CandidateScene, EpisodeRef
from ..geometry import TableProjector
from ..rendering import CandidateOverlayRenderer
from .base import BaseSceneOptimizer


class RenderSearchOptimizer(BaseSceneOptimizer):
    """Run a first-frame render-and-compare search.

    The current renderer is a fast projection overlay. It deliberately keeps the
    same optimizer/report interface that a slower SAPIEN renderer can implement
    later.
    """

    def __init__(
        self,
        projector: TableProjector | None = None,
        output_root: str | Path | None = None,
        max_attempts: int = 48,
        sapien_check: bool = False,
        sapien_optimize: bool = False,
        sapien_max_attempts: int = 5,
        sapien_strategy: str = "coordinate",
        coordinate_initial_step: float = 0.02,
        coordinate_min_step: float = 0.005,
    ):
        self.renderer = CandidateOverlayRenderer(projector=projector)
        self.output_root = None if output_root is None else Path(output_root)
        self.max_attempts = max_attempts
        self.sapien_check = sapien_check
        self.sapien_optimize = sapien_optimize
        self.sapien_max_attempts = sapien_max_attempts
        self.sapien_strategy = sapien_strategy
        self.coordinate_initial_step = coordinate_initial_step
        self.coordinate_min_step = coordinate_min_step

    def optimize(self, episode: EpisodeRef, candidates: tuple[CandidateScene, ...]) -> tuple[CandidateScene, ...]:
        optimized: list[CandidateScene] = []
        for candidate_index, candidate in enumerate(candidates):
            attempts = self._candidate_attempts(candidate)
            scored_attempts = []
            for attempt_index, attempt in enumerate(attempts[: self.max_attempts]):
                score = self.renderer.score(attempt)
                scored_attempts.append((score["score"], attempt_index, attempt, score))
            if not scored_attempts:
                optimized.append(candidate)
                continue

            scored_attempts.sort(key=lambda item: item[0], reverse=True)
            best_score, _, best_candidate, best_detail = scored_attempts[0]
            sapien_search = None
            if self.output_root is not None and self.sapien_optimize:
                out_dir = self.output_root / episode.split / "fixed_scene_task"
                if self.sapien_strategy == "coordinate":
                    sapien_search = self._coordinate_sapien_search(
                        episode=episode,
                        candidate_index=candidate_index,
                        start_candidate=best_candidate,
                        out_dir=out_dir,
                    )
                elif self.sapien_strategy == "objectwise":
                    sapien_search = self._objectwise_sapien_search(
                        episode=episode,
                        candidate_index=candidate_index,
                        start_candidate=best_candidate,
                        out_dir=out_dir,
                    )
                else:
                    sapien_search = self._score_sapien_attempts(
                        episode=episode,
                        candidate_index=candidate_index,
                        scored_attempts=scored_attempts[: self.sapien_max_attempts],
                        out_dir=out_dir,
                    )
                valid_sapien_attempts = [
                    item for item in (sapien_search or {}).get("attempts", []) if "sapien_score" in item
                ]
                if valid_sapien_attempts:
                    best_sapien = max(valid_sapien_attempts, key=lambda item: item["sapien_score"]["score"])
                    best_score = best_sapien["sapien_score"]["score"]
                    best_candidate = best_sapien["candidate"]
                    best_detail = best_sapien["sapien_score"]
            metadata = {
                "episode": episode.episode,
                "split": episode.split,
                "task_name": candidate.task_name,
                "candidate_index": candidate_index,
                "best_score": best_score,
                "best_detail": best_detail,
                "sapien_search": self._serialize_sapien_search(sapien_search),
                "attempts": [
                    {
                        "attempt_index": attempt_index,
                        "score": score,
                        "parameters": attempt.parameters,
                    }
                    for attempt_score, attempt_index, attempt, score in scored_attempts
                ],
            }
            visualization_path = None
            if self.output_root is not None:
                out_dir = self.output_root / episode.split / "fixed_scene_task"
                visualization_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.render_search.jpg"
                self.renderer.save_visualization(
                    episode,
                    best_candidate,
                    visualization_path,
                    title=f"{candidate.task_name} score={best_score:.4f}",
                    metadata=metadata,
                )
                sapien_paths = self._save_sapien_check(episode, best_candidate, out_dir, candidate_index)
                if sapien_paths:
                    metadata["sapien_check"] = sapien_paths
                    visualization_path.with_suffix(".json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            render_search = {
                "score": best_score,
                "renderer": best_detail["renderer"],
                "best_detail": best_detail,
                "attempt_count": len(scored_attempts),
                "visualization_path": None if visualization_path is None else str(visualization_path),
                "sidecar_path": None if visualization_path is None else str(visualization_path.with_suffix(".json")),
            }
            if self.output_root is not None:
                sapien_paths = metadata.get("sapien_check")
                if sapien_paths:
                    render_search["sapien_check"] = sapien_paths
                if sapien_search is not None:
                    render_search["sapien_search"] = self._serialize_sapien_search(sapien_search)
            parameters = copy.deepcopy(best_candidate.parameters)
            parameters["render_search"] = render_search
            optimized.append(
                replace(
                    best_candidate,
                    parameters=parameters,
                    confidence=max(best_candidate.confidence, min(1.0, best_score)),
                    notes=tuple(best_candidate.notes)
                    + (
                        (
                            "Render-search optimized with SAPIEN first-frame scoring."
                            if sapien_search is not None
                            else "Render-search optimized with candidate_overlay renderer."
                        ),
                        "Optimization is black-box/non-differentiable.",
                    ),
                )
            )

        return tuple(sorted(optimized, key=lambda item: item.confidence, reverse=True))

    def _objectwise_sapien_search(
        self,
        episode: EpisodeRef,
        candidate_index: int,
        start_candidate: CandidateScene,
        out_dir: Path,
    ) -> dict[str, Any] | None:
        try:
            from ..rendering.sapien_first_frame import (
                SapienFirstFrameRenderer,
                save_sapien_comparison,
                score_rendered_image,
            )
        except Exception as exc:
            return {"error": repr(exc), "attempts": []}

        renderer = SapienFirstFrameRenderer()
        attempts = []
        cache: dict[str, dict[str, Any]] = {}

        def object_name_for_path(path: list[Any]) -> str:
            obj = self._get_path(current.parameters, path)
            if isinstance(obj, dict):
                return str(obj.get("name", path[-1] if path else ""))
            return str(path[-1]) if path else ""

        def object_score(score: dict[str, Any], object_name: str) -> float:
            for item in score.get("per_object", []):
                if item.get("name") == object_name:
                    return float(item.get("score", 0.0))
            return 0.0

        def score_candidate(candidate: CandidateScene, tag: str) -> dict[str, Any]:
            key = self._candidate_key(candidate)
            if key in cache:
                return cache[key]
            attempt_index = len(attempts)
            render_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.obj{attempt_index:03d}.sapien.png"
            record: dict[str, Any] = {"attempt_index": attempt_index, "tag": tag, "candidate": candidate}
            try:
                rendered = renderer.render(candidate, render_path)
                if rendered is None:
                    record["error"] = "SAPIEN renderer does not support this task yet."
                else:
                    sapien_score = score_rendered_image(candidate.task_name, candidate.parameters, rendered)
                    comparison_path = out_dir / (
                        f"episode{episode.episode}.candidate{candidate_index}.obj{attempt_index:03d}.sapien_compare.jpg"
                    )
                    save_sapien_comparison(episode.first_frame_path, rendered, comparison_path)
                    record.update(
                        {
                            "sapien_score": sapien_score,
                            "render_path": str(rendered),
                            "comparison_path": str(comparison_path),
                        }
                    )
            except Exception as exc:
                record["error"] = repr(exc)
            attempts.append(record)
            cache[key] = record
            return record

        model_variants = self._model_variants(start_candidate)
        model_records = []
        for variant in model_variants:
            if len(attempts) >= self.sapien_max_attempts:
                break
            model_records.append(score_candidate(variant, self._model_variant_tag(variant)))
        valid_model_records = [record for record in model_records if "sapien_score" in record]
        if valid_model_records:
            start_record = max(valid_model_records, key=lambda item: item["sapien_score"]["score"])
            current = start_record["candidate"]
        else:
            current = start_candidate
            start_record = score_candidate(current, "start")
        if "sapien_score" not in start_record:
            return {
                "renderer": "sapien_first_frame",
                "strategy": "objectwise",
                "attempt_count": len(attempts),
                "attempts": attempts,
            }

        movable_paths = self._movable_table_xy_paths(current.parameters)
        per_object_budget = max(1, (self.sapien_max_attempts - 1) // max(1, len(movable_paths)))
        current_record = start_record
        for path in movable_paths:
            if len(attempts) >= self.sapien_max_attempts:
                break
            object_name = object_name_for_path(path)
            base_score = object_score(current_record["sapien_score"], object_name)
            best_record = current_record
            best_candidate = current
            step = self.coordinate_initial_step
            object_start_attempt_count = len(attempts)
            while (
                len(attempts) < self.sapien_max_attempts
                and len(attempts) - object_start_attempt_count < per_object_budget
                and step >= self.coordinate_min_step
            ):
                improved = False
                trials = []
                for axis, direction in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)):
                    if (
                        len(attempts) >= self.sapien_max_attempts
                        or len(attempts) - object_start_attempt_count >= per_object_budget
                    ):
                        break
                    trial = self._move_table_xy(best_candidate, path, axis, direction * step)
                    record = score_candidate(trial, f"{object_name}.axis{axis}.{direction:+.0f}.step{step:.4f}")
                    if "sapien_score" not in record:
                        continue
                    trial_score = object_score(record["sapien_score"], object_name)
                    objectwise_scores = record.setdefault("objectwise_scores", {})
                    objectwise_scores[object_name] = trial_score
                    record.setdefault("objectwise_target", object_name)
                    record.setdefault("objectwise_score", trial_score)
                    trials.append((trial_score, record))
                if trials:
                    trial_score, record = max(trials, key=lambda item: item[0])
                    if trial_score > base_score + 1e-6:
                        base_score = trial_score
                        best_record = record
                        best_candidate = record["candidate"]
                        improved = True
                if improved:
                    continue
                step *= 0.5
            current = best_candidate
            current_record = best_record

        final_record = score_candidate(current, "merged_objectwise_best")
        return {
            "renderer": "sapien_first_frame",
            "strategy": "objectwise",
            "model_variant_count": len(model_variants),
            "attempt_count": len(attempts),
            "best_attempt_index": final_record.get("attempt_index"),
            "attempts": attempts,
        }

    def _coordinate_sapien_search(
        self,
        episode: EpisodeRef,
        candidate_index: int,
        start_candidate: CandidateScene,
        out_dir: Path,
    ) -> dict[str, Any] | None:
        try:
            from ..rendering.sapien_first_frame import (
                SapienFirstFrameRenderer,
                save_sapien_comparison,
                score_rendered_image,
            )
        except Exception as exc:
            return {"error": repr(exc), "attempts": []}

        renderer = SapienFirstFrameRenderer()
        attempts = []
        cache: dict[str, dict[str, Any]] = {}

        def score_candidate(candidate: CandidateScene, tag: str) -> dict[str, Any]:
            key = self._candidate_key(candidate)
            if key in cache:
                return cache[key]
            attempt_index = len(attempts)
            render_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.coord{attempt_index:03d}.sapien.png"
            record: dict[str, Any] = {
                "attempt_index": attempt_index,
                "tag": tag,
                "candidate": candidate,
            }
            try:
                rendered = renderer.render(candidate, render_path)
                if rendered is None:
                    record["error"] = "SAPIEN renderer does not support this task yet."
                else:
                    sapien_score = score_rendered_image(candidate.task_name, candidate.parameters, rendered)
                    comparison_path = out_dir / (
                        f"episode{episode.episode}.candidate{candidate_index}.coord{attempt_index:03d}.sapien_compare.jpg"
                    )
                    save_sapien_comparison(episode.first_frame_path, rendered, comparison_path)
                    record.update(
                        {
                            "sapien_score": sapien_score,
                            "render_path": str(rendered),
                            "comparison_path": str(comparison_path),
                        }
                    )
            except Exception as exc:
                record["error"] = repr(exc)
            attempts.append(record)
            cache[key] = record
            return record

        current = start_candidate
        current_record = score_candidate(current, "start")
        current_score = current_record.get("sapien_score", {}).get("score", float("-inf"))
        step = self.coordinate_initial_step
        direction_scores: dict[tuple[str, int, float], float] = {}

        while len(attempts) < self.sapien_max_attempts and step >= self.coordinate_min_step:
            trial_records = []
            improved = False
            trial_specs = []
            for path in self._movable_table_xy_paths(current.parameters):
                path_key = ".".join(map(str, path))
                for axis, direction in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)):
                    history = direction_scores.get((path_key, axis, direction), float("-inf"))
                    trial_specs.append((history, path, path_key, axis, direction))
            trial_specs.sort(key=lambda item: item[0], reverse=True)

            for _, path, path_key, axis, direction in trial_specs:
                if len(attempts) >= self.sapien_max_attempts:
                    break
                move_key = (path_key, axis, direction)
                trial = self._move_table_xy(current, path, axis, direction * step)
                record = score_candidate(trial, f"{path_key}.axis{axis}.{direction:+.0f}.step{step:.4f}")
                if "sapien_score" in record:
                    score = record["sapien_score"]["score"]
                    direction_scores[move_key] = max(direction_scores.get(move_key, float("-inf")), score)
                    trial_records.append(record)
                    if score > current_score + 1e-6:
                        current = record["candidate"]
                        current_record = record
                        current_score = score
                        improved = True
                        break
            if improved:
                continue
            valid = [item for item in trial_records if "sapien_score" in item]
            if not valid:
                step *= 0.5
                continue
            step *= 0.5

        return {
            "renderer": "sapien_first_frame",
            "strategy": "coordinate",
            "attempt_count": len(attempts),
            "best_attempt_index": current_record.get("attempt_index"),
            "attempts": attempts,
        }

    def _score_sapien_attempts(
        self,
        episode: EpisodeRef,
        candidate_index: int,
        scored_attempts: list[tuple[float, int, CandidateScene, dict[str, Any]]],
        out_dir: Path,
    ) -> dict[str, Any] | None:
        try:
            from ..rendering.sapien_first_frame import (
                SapienFirstFrameRenderer,
                save_sapien_comparison,
                score_rendered_image,
            )
        except Exception as exc:
            return {"error": repr(exc), "attempts": []}

        renderer = SapienFirstFrameRenderer()
        results = []
        for rank, (overlay_score, attempt_index, attempt, overlay_detail) in enumerate(scored_attempts):
            render_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.attempt{attempt_index}.sapien.png"
            try:
                rendered = renderer.render(attempt, render_path)
                if rendered is None:
                    continue
                sapien_score = score_rendered_image(attempt.task_name, attempt.parameters, rendered)
                comparison_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.attempt{attempt_index}.sapien_compare.jpg"
                save_sapien_comparison(episode.first_frame_path, rendered, comparison_path)
                results.append(
                    {
                        "rank": rank,
                        "attempt_index": attempt_index,
                        "overlay_score": overlay_score,
                        "overlay_detail": overlay_detail,
                        "sapien_score": sapien_score,
                        "render_path": str(rendered),
                        "comparison_path": str(comparison_path),
                        "candidate": attempt,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "rank": rank,
                        "attempt_index": attempt_index,
                        "overlay_score": overlay_score,
                        "error": repr(exc),
                        "candidate": attempt,
                    }
                )
        return {
            "renderer": "sapien_first_frame",
            "attempt_count": len(results),
            "attempts": results,
        }

    @staticmethod
    def _serialize_sapien_search(search: dict[str, Any] | None) -> dict[str, Any] | None:
        if search is None:
            return None
        serialized = dict(search)
        attempts = []
        for item in search.get("attempts", []):
            out = dict(item)
            candidate = out.pop("candidate", None)
            if candidate is not None:
                out["parameters"] = candidate.parameters
            attempts.append(out)
        serialized["attempts"] = attempts
        return serialized

    def _save_sapien_check(
        self,
        episode: EpisodeRef,
        candidate: CandidateScene,
        out_dir: Path,
        candidate_index: int,
    ) -> dict[str, str] | None:
        if not self.sapien_check:
            return None
        try:
            from ..rendering.sapien_first_frame import SapienFirstFrameRenderer, save_sapien_comparison

            render_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.sapien.png"
            rendered = SapienFirstFrameRenderer().render(candidate, render_path)
            if rendered is None:
                return None
            comparison_path = out_dir / f"episode{episode.episode}.candidate{candidate_index}.sapien_compare.jpg"
            save_sapien_comparison(episode.first_frame_path, rendered, comparison_path)
            return {"render_path": str(rendered), "comparison_path": str(comparison_path)}
        except Exception as exc:
            return {"error": repr(exc)}

    def _candidate_attempts(self, candidate: CandidateScene) -> list[CandidateScene]:
        attempts = [candidate]
        edits = list(self._search_edits(candidate.parameters))
        for path, key, value in edits:
            params = copy.deepcopy(candidate.parameters)
            target = params
            for part in path:
                target = target[part]
            target[key] = value
            if key == "table_xy" and isinstance(value, list) and len(value) == 2:
                pose_guess = target.get("table_pose_guess")
                if isinstance(pose_guess, dict):
                    p = list(pose_guess.get("p", [value[0], value[1], 0.741]))
                    if len(p) >= 2:
                        p[0] = value[0]
                        p[1] = value[1]
                        pose_guess["p"] = p
            attempts.append(
                replace(
                    candidate,
                    parameters=params,
                    notes=tuple(candidate.notes) + (f"Render-search perturbation {'.'.join(map(str, path + [key]))}={value}.",),
                )
            )
        return attempts

    def _model_variants(self, candidate: CandidateScene) -> list[CandidateScene]:
        model_paths: list[tuple[list[Any], list[Any]]] = []

        def walk(obj: Any, path: list[Any]) -> None:
            if isinstance(obj, dict):
                candidates = obj.get("model_id_candidates")
                if isinstance(candidates, list) and candidates:
                    model_paths.append((path, candidates))
                for key, value in obj.items():
                    walk(value, path + [key])
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    walk(value, path + [index])

        walk(candidate.parameters, [])
        if not model_paths:
            return [candidate]

        variants = [copy.deepcopy(candidate.parameters)]
        for path, ids in model_paths:
            next_variants = []
            for params in variants:
                for model_id in ids:
                    copied = copy.deepcopy(params)
                    target = self._get_path(copied, path)
                    target["model_id"] = int(model_id)
                    next_variants.append(copied)
            variants = next_variants

        return [
            replace(
                candidate,
                parameters=params,
                notes=tuple(candidate.notes) + ("Discrete model_id variant for SAPIEN search.",),
            )
            for params in variants
        ]

    @staticmethod
    def _model_variant_tag(candidate: CandidateScene) -> str:
        tags: list[str] = []

        def walk(obj: Any, path: list[Any]) -> None:
            if isinstance(obj, dict):
                if "model_id" in obj:
                    tags.append(f"{'.'.join(map(str, path))}.model_id={obj['model_id']}")
                for key, value in obj.items():
                    walk(value, path + [key])
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    walk(value, path + [index])

        walk(candidate.parameters, [])
        return "model_select:" + ",".join(tags) if tags else "start"

    @staticmethod
    def _candidate_key(candidate: CandidateScene) -> str:
        def relevant(obj: Any):
            if isinstance(obj, dict):
                out = {}
                for key, value in obj.items():
                    if key in {"table_xy", "yaw", "model_id", "model_id_candidates"}:
                        out[key] = value
                    elif isinstance(value, (dict, list)):
                        child = relevant(value)
                        if child not in ({}, []):
                            out[key] = child
                return out
            if isinstance(obj, list):
                return [relevant(item) for item in obj]
            return {}

        return json.dumps(relevant(candidate.parameters), sort_keys=True)

    def _movable_table_xy_paths(self, parameters: dict[str, Any]) -> list[list[Any]]:
        paths: list[list[Any]] = []

        def walk(obj: Any, path: list[Any]) -> None:
            if isinstance(obj, dict):
                if isinstance(obj.get("table_xy"), list) and len(obj.get("table_xy")) == 2 and isinstance(obj.get("search"), dict):
                    paths.append(path)
                for key, value in obj.items():
                    walk(value, path + [key])
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    walk(value, path + [index])

        walk(parameters, [])
        return paths

    def _move_table_xy(self, candidate: CandidateScene, path: list[Any], axis: int, delta: float) -> CandidateScene:
        params = copy.deepcopy(candidate.parameters)
        target = self._get_path(params, path)
        xy = list(target["table_xy"])
        xy[axis] = self._clamp_axis(xy[axis] + delta, target.get("search", {}).get("x" if axis == 0 else "y"))
        target["table_xy"] = xy
        pose_guess = target.get("table_pose_guess")
        if isinstance(pose_guess, dict):
            p = list(pose_guess.get("p", [xy[0], xy[1], 0.741]))
            if len(p) >= 2:
                p[0] = xy[0]
                p[1] = xy[1]
                pose_guess["p"] = p
        return replace(
            candidate,
            parameters=params,
            notes=tuple(candidate.notes) + (f"Coordinate search moved {'.'.join(map(str, path))}[{axis}] by {delta:.4f}.",),
        )

    @staticmethod
    def _get_path(root: Any, path: list[Any]) -> Any:
        target = root
        for part in path:
            target = target[part]
        return target

    @staticmethod
    def _clamp_axis(value: float, bounds: Any) -> float:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return value
        return min(max(value, float(bounds[0])), float(bounds[1]))

    def _search_edits(self, parameters: dict[str, Any]) -> list[tuple[list[Any], str, Any]]:
        edits: list[tuple[list[Any], str, Any]] = []

        def walk(obj: Any, path: list[Any]) -> None:
            if isinstance(obj, dict):
                search = obj.get("search")
                table_xy = obj.get("table_xy")
                if isinstance(search, dict) and isinstance(table_xy, list) and len(table_xy) == 2:
                    x_values = self._values_around(search.get("x"), table_xy[0])
                    y_values = self._values_around(search.get("y"), table_xy[1])
                    for value in x_values:
                        edits.append((path, "table_xy", [value, table_xy[1]]))
                    for value in y_values:
                        edits.append((path, "table_xy", [table_xy[0], value]))
                    for x_value in x_values:
                        for y_value in y_values:
                            edits.append((path, "table_xy", [x_value, y_value]))
                    if "yaw" in search:
                        for yaw in self._values_around(search.get("yaw"), obj.get("yaw", 0.0)):
                            edits.append((path, "yaw", yaw))
                for key, value in obj.items():
                    walk(value, path + [key])
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    walk(value, path + [index])

        walk(parameters, [])
        return edits

    @staticmethod
    def _values_around(bounds: Any, center: Any) -> list[float]:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return []
        lo, hi = float(bounds[0]), float(bounds[1])
        center = float(center)
        return [lo, (lo + center) * 0.5, (center + hi) * 0.5, hi]
