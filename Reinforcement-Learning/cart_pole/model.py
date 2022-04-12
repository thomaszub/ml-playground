from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader, TensorDataset


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
        n_components: int,
        learning_rate: float,
    ) -> None:
        self._sampler = RBFSampler(n_components=n_components)
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


class DeepNNActionModel(ActionModel):
    def __init__(
        self,
        hidden_nodes: Tuple[int, int],
        batch_size: int,
        replay_buffer_size_in_batches: int,
    ) -> None:
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

    @torch.no_grad()
    def predict(self, observation: np.ndarray, action: int) -> float:
        return self._model(self._input(observation, action)).detach().numpy()

    def update(self, observation: np.ndarray, action: int, target: float) -> None:
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
