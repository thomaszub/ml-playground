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


class EpsilonGreedyPolicy(Policy):
    def __init__(self, world: GridWorld, epsilon: float, policy: Policy = None) -> None:
        super().__init__(world)
        if epsilon < 1.0 and policy is None:
            raise ValueError("policy can not be None if epsilon is smaller 1.0")
        self._epsilon = epsilon
        self._policy = policy

    def _random_func(self, field: Field) -> Callable[[MoveAction], float]:
        possible_actions = self._world.possible_actions(field)
        return lambda _: 1.0 / len(possible_actions)

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        eps = self._epsilon
        if self._policy is None:
            return lambda a: self._random_func(field)(a)
        else:
            return lambda a: eps * self._random_func(field)(a) + (
                1.0 - eps
            ) * self._policy.func(field)(a)

    def _sample(self, field: Field, prob: Callable[[MoveAction], float]) -> MoveAction:
        actions = self._world.possible_actions(field)
        probs = [prob(act) for act in actions]
        return np.random.choice(actions, p=probs)

    def sample(
        self,
        field: Field,
    ) -> MoveAction:
        if np.random.random() < self._epsilon:
            return self._sample(field, self._random_func(field))
        else:
            return self._sample(field, self._policy.func(field))


class ArgMaxPolicy(Policy):
    def __init__(
        self,
        world: GridWorld,
        action_values: Dict[Tuple[Field, MoveAction], float],
    ) -> None:
        super().__init__(world)
        self._action_values = action_values

    def func(self, field: Field) -> Callable[[MoveAction], float]:
        action_values_for_state = [
            (it[0][1], it[1]) for it in self._action_values.items() if it[0][0] == field
        ]
        best_action = max(action_values_for_state, key=lambda a: a[1])[0]
        return lambda action: 1.0 if action == best_action else 0.0
