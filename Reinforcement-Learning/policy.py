from abc import ABC, abstractmethod
from dataclasses import Field
from typing import Callable, Dict, Tuple

import numpy as np

from environment.movement import MoveAction
from environment.world import GridWorld


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


class ArgMaxPolicy(Policy):
    def __init__(
        self,
        world: GridWorld,
        action_values: Dict[Tuple[Field, MoveAction], float] = {},
    ) -> None:
        super().__init__(world)
        self._action_values = action_values

    def update(self, action_values: Dict[Tuple[Field, MoveAction], float]) -> None:
        self._action_values = action_values

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        action_values_for_state = [
            (it[0][1], it[1]) for it in self._action_values.items() if it[0][0] == field
        ]
        best_action = max(action_values_for_state, key=lambda a: a[1])[0]
        return lambda action: 1.0 if action == best_action else 0.0
