from abc import ABC, abstractmethod
from copy import deepcopy
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


class DeepQAgent(Agent):
    def __init__(
        self,
        hidden_nodes: Tuple[int, int],
        batch_size: int,
        train_after_num_episodes: int,
        update_target_after_trainings: int,
        discount_rate: float,
        epsilon: float,
    ):
        self._model = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_nodes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes[0], hidden_nodes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes[1], 1),
            torch.nn.ReLU(),
        )
        self._target_model = deepcopy(self._model)
        self._loss = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(params=self._model.parameters())
        self._batch_size = batch_size
        self._train_after_num_episodes = train_after_num_episodes
        self._update_target_after_trainings = update_target_after_trainings
        self._episodes = 0
        self._trainings = 0
        self._replay_buffer_input = []
        self._replay_buffer_target = []
        self._discount_rate = discount_rate
        self._espsilon = epsilon

    def sample(self, observation: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._predict(self._model, observation, 0)
            Q_1 = self._predict(self._model, observation, 1)
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
            Q_0 = self._predict(self._target_model, new_observation, 0)
            Q_1 = self._predict(self._target_model, new_observation, 1)
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._replay_buffer_input.append(np.concatenate((observation, [action])))
        self._replay_buffer_target.append([target])
        if done:
            self._episodes += 1
        if self._episodes >= self._train_after_num_episodes:
            self._episodes = 0
            self._trainings += 1
            self._train()
            self._replay_buffer_input = []
            self._replay_buffer_target = []
        if self._trainings >= self._update_target_after_trainings:
            self._trainings = 0
            self._target_model = deepcopy(self._model)

    @torch.no_grad()
    def _predict(
        self, model: torch.nn.Module, observation: np.ndarray, action: int
    ) -> float:
        input = (
            torch.tensor(np.concatenate((observation, [action]))).float().view(1, -1)
        )
        return model(input).detach().numpy()

    def _train(self) -> None:
        input = torch.tensor(np.array(self._replay_buffer_input)).float()
        target = torch.tensor(np.array(self._replay_buffer_target)).float()
        dataset = TensorDataset(input, target)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        for X, y in loader:
            self._optimizer.zero_grad()
            self._loss(self._model(X), y).backward()
            self._optimizer.step()
