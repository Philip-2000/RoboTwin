from __future__ import annotations

from .strategies.base import BaseReconstructor, PriorOnlyReconstructor
from .strategies.beat_block_hammer import BeatBlockHammerReconstructor
from .strategies.click_bell import ClickBellReconstructor
from .strategies.place_empty_cup import PlaceEmptyCupReconstructor
from .strategies.place_object_basket import PlaceObjectBasketReconstructor
from .strategies.stack_blocks_three import StackBlocksThreeReconstructor


def build_registry() -> dict[str, type[BaseReconstructor]]:
    strategies: list[type[BaseReconstructor]] = [
        ClickBellReconstructor,
        PlaceEmptyCupReconstructor,
        StackBlocksThreeReconstructor,
        BeatBlockHammerReconstructor,
        PlaceObjectBasketReconstructor,
    ]
    return {strategy.task_name: strategy for strategy in strategies}


def get_reconstructor(task_name: str) -> BaseReconstructor:
    registry = build_registry()
    cls = registry.get(task_name, PriorOnlyReconstructor)
    return cls(task_name=task_name) if cls is PriorOnlyReconstructor else cls()

