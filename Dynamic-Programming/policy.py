from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from movement import MoveAction, Position
from world import GridWorld


class Policy(ABC):
    @abstractmethod
    def func(self, position: Position) -> Callable[[MoveAction], float]:
        pass

    @abstractmethod
    def sample(self, position: Position) -> MoveAction:
        pass


class UniformRandomPolicy(Policy):
    def __init__(self, world: GridWorld) -> None:
        self._world = world

    def func(self, position: Position) -> Callable[[MoveAction], float]:
        possible_actions = self._world.possible_actions(position)
        return lambda x: 1.0 / len(possible_actions)

    def sample(self, position: Position) -> MoveAction:
        func = self.func(position)
        action_probs = [
            (act, func(act)) for act in self._world.possible_actions(position)
        ]
        val = np.random.uniform()
        for act_prob in action_probs:
            val -= act_prob[1]
            if val < 0.0:
                return act_prob[0]
