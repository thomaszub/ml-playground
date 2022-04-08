from abc import ABC, abstractmethod
from typing import Callable, Tuple

from typing import Dict

from environment.world import Field, GridWorld
from environment.movement import MoveAction
from policy import Policy, EpsilonGreedyPolicy, ArgMaxPolicy


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
        world: GridWorld,
        epsilon: float,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        super().__init__()
        self._action_values = {
            (field, action): 0.0
            for field in world.get_fields()
            for action in world.possible_actions(field)
        }
        self._policy = EpsilonGreedyPolicy(
            world, epsilon, ArgMaxPolicy(world, self._action_values)
        )
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    def reset(self, start_field: Field):
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

    def get_action_values(self) -> Dict[Tuple[Field, MoveAction], float]:
        return self._action_values

    def get_policy(self) -> Policy:
        return self._policy
