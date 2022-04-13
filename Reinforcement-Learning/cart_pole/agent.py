from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader, TensorDataset


class Agent(ABC):
    @abstractmethod
    def sample(self, observation: np.ndarray, eps_greedy: bool) -> int:
        pass

    @abstractmethod
    def train(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
    ) -> None:
        pass


class LinearRBFAgent(Agent):
    def __init__(
        self,
        observation_space: gym.Space[np.ndarray],
        action_space: gym.Space[int],
        n_components: int,
        learning_rate: float,
        discount_rate: float,
        epsilon: float,
    ):
        self._sampler = RBFSampler(n_components=n_components)
        self._sampler.fit(
            self._input(observation_space.sample(), action_space.sample())
        )
        self._model = SGDRegressor(learning_rate="constant", eta0=learning_rate)
        self._model.partial_fit(
            self._sampled_input(observation_space.sample(), action_space.sample()),
            self._target(1.0),
        )
        self._discount_rate = discount_rate
        self._espsilon = epsilon

    def sample(self, observation: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._model.predict(self._sampled_input(observation, 0))
            Q_1 = self._model.predict(self._sampled_input(observation, 1))
            return np.argmax([Q_0, Q_1])

    def train(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
    ) -> None:
        if done:
            target = reward
        else:
            Q_0 = self._model.predict(self._sampled_input(new_observation, 0))
            Q_1 = self._model.predict(self._sampled_input(new_observation, 1))
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._model.partial_fit(
            self._sampled_input(observation, action), self._target(target)
        )

    def _input(self, observation: np.ndarray, action: int) -> np.ndarray:
        return [np.concatenate((observation, [action]))]

    def _sampled_input(self, observation: np.ndarray, action: int) -> np.ndarray:
        return self._sampler.transform(self._input(observation, action))

    def _target(self, value: float):
        return np.array([value])


class DeepNNLinearRBFAgent(Agent):
    def __init__(
        self,
        hidden_nodes: Tuple[int, int],
        batch_size: int,
        replay_buffer_size_in_batches: int,
        discount_rate: float,
        epsilon: float,
    ):
        self._model = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_nodes[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_nodes[0], hidden_nodes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes[1], 1),
        )
        self._loss = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(params=self._model.parameters())
        self._replay_buffer_size = batch_size * replay_buffer_size_in_batches
        self._batch_size = batch_size
        self._replay_buffer_input = []
        self._replay_buffer_target = []
        self._discount_rate = discount_rate
        self._espsilon = epsilon

    def sample(self, observation: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._predict(observation, 0)
            Q_1 = self._predict(observation, 1)
            return np.argmax([Q_0, Q_1])

    def train(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
    ) -> None:
        if done:
            target = reward
        else:
            Q_0 = self._predict(new_observation, 0)
            Q_1 = self._predict(new_observation, 1)
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._update(observation, action, target)

    @torch.no_grad()
    def _predict(self, observation: np.ndarray, action: int) -> float:
        return self._model(self._input(observation, action)).detach().numpy()

    def _update(self, observation: np.ndarray, action: int, target: float) -> None:
        if len(self._replay_buffer_input) < self._replay_buffer_size:
            self._replay_buffer_input.append(np.concatenate((observation, [action])))
            self._replay_buffer_target.append([target])
        else:
            self._train()
            self._replay_buffer_input = []
            self._replay_buffer_target = []

    def _train(self) -> None:
        input = torch.tensor(np.array(self._replay_buffer_input)).float()
        target = torch.tensor(np.array(self._replay_buffer_target)).float()
        dataset = TensorDataset(input, target)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        for X, y in loader:
            self._optimizer.zero_grad()
            self._loss(self._model(X), y).backward()
            self._optimizer.step()

    def _input(self, observation: np.ndarray, action: int) -> torch.Tensor:
        return torch.tensor(np.concatenate((observation, [action]))).float().view(1, -1)
