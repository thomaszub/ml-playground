from abc import ABC, abstractmethod
from typing import Callable, Tuple

from typing import Dict

from environment.world import Field
from environment.movement import MoveAction
from policy import Policy


class Agent(ABC):
    @abstractmethod
    def reset(self, start_field: Field):
        pass

    @abstractmethod
    def action_callback(self) -> Callable[[Field], MoveAction]:
        pass

    @abstractmethod
    def learn_callback(self) -> Callable[[Field, float, Field], None]:
        pass


class SarsaAgent(Agent):
    def __init__(
        self,
        policy: Policy,
        action_values: Dict[Tuple[Field, MoveAction], float],
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        super().__init__()
        self._policy = policy
        self._action_values = action_values
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    def reset(self, start_field: Field):
        self._field = start_field
        self._action = self._policy.sample(start_field)

    def action_callback(self) -> Callable[[Field], MoveAction]:
        return lambda _: self._action

    def learn_callback(self) -> Callable[[Field, float, Field], None]:
        def learn_callback(field: Field, reward: float, new_field: Field) -> None:
            next_action = self._policy.sample(new_field)
            self._action_values[(field, self._action)] += self._learning_rate * (
                reward
                + self._discount_factor * self._action_values[(new_field, next_action)]
                - self._action_values[(field, self._action)]
            )
            self._action = next_action

        return learn_callback
