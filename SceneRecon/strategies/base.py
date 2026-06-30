from __future__ import annotations

from abc import ABC, abstractmethod

from ..core import CandidateScene, DetectionSet, EpisodeRef, ObjectPrior, ReconstructionReport, ScenePrior
from ..task_mapping import TaskMatch


class BaseReconstructor(ABC):
    task_name: str

    @abstractmethod
    def prior(self) -> ScenePrior:
        raise NotImplementedError

    def initial_candidates(
        self,
        episode: EpisodeRef,
        detections: DetectionSet | None = None,
    ) -> tuple[CandidateScene, ...]:
        return (
            CandidateScene(
                task_name=self.task_name,
                parameters={"mode": "prior_only", "instruction": episode.instruction},
                confidence=0.1,
                notes=("No vision fitting has been run yet.",),
            ),
        )

    def reconstruct(
        self,
        episode: EpisodeRef,
        task_match: TaskMatch | None = None,
        detections: DetectionSet | None = None,
    ) -> ReconstructionReport:
        warnings = []
        if task_match is not None and task_match.task_name != self.task_name:
            warnings.append(f"Task match {task_match.task_name} differs from strategy {self.task_name}")
        return ReconstructionReport(
            episode=episode,
            task_name=self.task_name,
            strategy=self.__class__.__name__,
            prior=self.prior(),
            candidates=self.initial_candidates(episode, detections=detections),
            detections=detections,
            warnings=tuple(warnings),
            metadata={
                "task_match_score": None if task_match is None else task_match.score,
                "task_match_reason": None if task_match is None else task_match.reason,
            },
        )


class PriorOnlyReconstructor(BaseReconstructor):
    def __init__(self, task_name: str):
        self.task_name = task_name

    def prior(self) -> ScenePrior:
        return ScenePrior(
            task_name=self.task_name,
            difficulty="unknown",
            objects=(),
            strategy_notes=("No task-specific strategy registered yet.",),
        )
