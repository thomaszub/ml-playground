import numpy as np

from model import ActionModel


class Agent:
    def __init__(
        self,
        model: ActionModel,
        discount_rate: float,
        epsilon: float,
    ):
        self._discount_rate = discount_rate
        self._espsilon = epsilon
        self._model = model

    def sample(self, observation: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._model.predict(observation, 0)
            Q_1 = self._model.predict(observation, 1)
            return np.argmax([Q_0, Q_1])

    def train(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
    ):
        if done:
            target = reward
        else:
            Q_0 = self._model.predict(new_observation, 0)
            Q_1 = self._model.predict(new_observation, 1)
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._model.update(observation, action, target)
