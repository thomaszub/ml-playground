from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np

from movement import MoveAction, Position
from world import GridWorld


class Policy(ABC):
    def __init__(self, world: GridWorld) -> None:
        super().__init__()
        self._world = world

    @abstractmethod
    def func(self, position: Position) -> Callable[[MoveAction], float]:
        pass

    def sample(self, position: Position) -> MoveAction:
        func = self.func(position)
        actions = self._world.possible_actions(position)
        probs = [func(act) for act in actions]
        return np.random.choice(actions, p=probs)


class UniformRandomPolicy(Policy):
    def __init__(self, world: GridWorld) -> None:
        super().__init__(world)

    def func(self, position: Position) -> Callable[[MoveAction], float]:
        possible_actions = self._world.possible_actions(position)
        return lambda x: 1.0 / len(possible_actions)


class ManuelPolicy(Policy):
    def __init__(
        self, world: GridWorld, positionToAction: Dict[Position, List[MoveAction]]
    ) -> None:
        super().__init__(world)
        self._positionToAction = positionToAction

    def func(self, position: Position) -> Callable[[MoveAction], float]:
        chosen_actions = self._positionToAction[position]
        return (
            lambda action: 1.0 / len(chosen_actions)
            if action in chosen_actions
            else 0.0
        )
