from abc import ABC, abstractmethod

import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import OneHotEncoder


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
        self._action_encoder = OneHotEncoder(categories=[[0, 1]])
        self._action_encoder.fit(np.array([[action_space.sample()]]))
        self._sampler = RBFSampler()
        self._sampler.fit([observation_space.sample()])
        self._model = SGDRegressor()
        self._model.partial_fit(
            self._input(observation_space.sample(), action_space.sample()),
            self._target(0.0),
        )

    def predict(self, observation: np.ndarray, action: int) -> float:
        return self._model.predict(self._input(observation, action))

    def update(self, observation: np.ndarray, action: int, target: float) -> None:
        self._model.partial_fit(self._input(observation, action), self._target(target))

    def _input(self, observation: np.ndarray, action: int) -> np.ndarray:
        return [
            np.concatenate(
                (
                    observation,
                    self._action_encoder.transform([[action]]).toarray().reshape(-1),
                )
            )
        ]

    def _target(self, value: float):
        return np.array([value]).ravel()
