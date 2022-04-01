from abc import ABC, abstractmethod
from dataclasses import Field
from typing import Callable, Dict, List

import numpy as np

from movement import MoveAction, Position
from world import GridWorld


class Policy(ABC):
    def __init__(self, world: GridWorld) -> None:
        super().__init__()
        self._world = world

    @abstractmethod
    def func(self, field: Field) -> Callable[[MoveAction], float]:
        pass

    def sample(self, field: Field) -> MoveAction:
        func = self.func(field)
        actions = self._world.possible_actions(field)
        probs = [func(act) for act in actions]
        return np.random.choice(actions, p=probs)


class UniformRandomPolicy(Policy):
    def __init__(self, world: GridWorld) -> None:
        super().__init__(world)

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        possible_actions = self._world.possible_actions(field)
        return lambda x: 1.0 / len(possible_actions)


class ManuelPolicy(Policy):
    def __init__(
        self, world: GridWorld, fieldToAction: Dict[Field, List[MoveAction]]
    ) -> None:
        super().__init__(world)
        self._fieldToAction = fieldToAction

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        chosen_actions = self._fieldToAction[field]
        return (
            lambda action: 1.0 / len(chosen_actions)
            if action in chosen_actions
            else 0.0
        )
