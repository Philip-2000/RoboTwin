from __future__ import annotations

from abc import ABC, abstractmethod

from ..core import CandidateScene, EpisodeRef


class BaseSceneOptimizer(ABC):
    @abstractmethod
    def optimize(self, episode: EpisodeRef, candidates: tuple[CandidateScene, ...]) -> tuple[CandidateScene, ...]:
        raise NotImplementedError

