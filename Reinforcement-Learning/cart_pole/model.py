from abc import ABC, abstractmethod

import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class ActionModel(ABC):
    @abstractmethod
    def predict(self, observation: np.ndarray, action: int) -> float:
        pass

    @abstractmethod
    def update(self, observation: np.ndarray, action: int, target: float) -> None:
        pass


class LinearRBFActionModel(ActionModel):
    def __init__(
        self,
        observation_space: gym.Space[np.ndarray],
        action_space: gym.Space[int],
        learning_rate: float,
    ) -> None:
        self._sampler = RBFSampler(n_components=4)
        self._sampler.fit(
            self._input(observation_space.sample(), action_space.sample())
        )
        self._model = SGDRegressor(learning_rate="constant", eta0=learning_rate)
        self._model.partial_fit(
            self._sampled_input(observation_space.sample(), action_space.sample()),
            self._target(1.0),
        )

    def predict(self, observation: np.ndarray, action: int) -> float:
        return self._model.predict(self._sampled_input(observation, action))

    def update(self, observation: np.ndarray, action: int, target: float) -> None:
        self._model.partial_fit(
            self._sampled_input(observation, action), self._target(target)
        )

    def _input(self, observation: np.ndarray, action: int) -> np.ndarray:
        return [np.concatenate((observation, [action]))]

    def _sampled_input(self, observation: np.ndarray, action: int) -> np.ndarray:
        return self._sampler.transform(self._input(observation, action))

    def _target(self, value: float):
        return np.array([value])
