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


class UpdateablePolicy(Policy):
    def __init__(self, world: GridWorld, discount_factor: float) -> None:
        super().__init__(world)
        self._state_to_action = {}
        self._discount_factor = discount_factor
        self.update({})

    def update(self, state_values: Dict[Field, float]) -> None:
        non_terminal_states = [
            field for field in self._world.get_fields() if not field.type.is_terminal()
        ]
        for field in non_terminal_states:
            possible_actions = self._world.possible_actions(field)
            values = []
            for action in possible_actions:
                new_state, reward = self._world.move(field, action)
                value = reward + self._discount_factor * state_values.get(
                    new_state, 0.0
                )
                values.append(value)
            self._state_to_action.update({field: possible_actions[np.argmax(values)]})

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        best_action = self._state_to_action[field]
        return lambda action: 1.0 if action == best_action else 0.0
