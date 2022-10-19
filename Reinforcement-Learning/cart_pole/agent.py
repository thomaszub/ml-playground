from copy import deepcopy
from typing import Protocol, Tuple

import gym
import numpy as np
import torch
from torch.nn import Module, Sequential, MSELoss, Linear, ReLU
from torch.optim import Adam
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader, TensorDataset


class Agent(Protocol):
    def sample(self, state: np.ndarray, eps_greedy: bool) -> int:
        ...

    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        ...


class LinearRBFAgent(Agent):
    def __init__(
        self,
        state_space: gym.Space[np.ndarray],
        action_space: gym.Space[int],
        n_components: int,
        learning_rate: float,
        discount_rate: float,
        epsilon: float,
    ):
        self._sampler = RBFSampler(n_components=n_components)
        self._sampler.fit(self._input(state_space.sample(), action_space.sample()))
        self._model = SGDRegressor(learning_rate="constant", eta0=learning_rate)
        self._model.partial_fit(
            self._sampled_input(state_space.sample(), action_space.sample()),
            self._target(1.0),
        )
        self._discount_rate = discount_rate
        self._espsilon = epsilon

    def sample(self, state: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._model.predict(self._sampled_input(state, 0))
            Q_1 = self._model.predict(self._sampled_input(state, 1))
            return np.argmax([Q_0, Q_1])

    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        if done:
            target = reward
        else:
            Q_0 = self._model.predict(self._sampled_input(new_state, 0))
            Q_1 = self._model.predict(self._sampled_input(new_state, 1))
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._model.partial_fit(
            self._sampled_input(state, action), self._target(target)
        )

    def _input(self, state: np.ndarray, action: int) -> np.ndarray:
        return [np.concatenate((state, [action]))]

    def _sampled_input(self, state: np.ndarray, action: int) -> np.ndarray:
        return self._sampler.transform(self._input(state, action))

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
        self._model = Sequential(
            Linear(5, hidden_nodes[0]),
            ReLU(),
            Linear(hidden_nodes[0], hidden_nodes[1]),
            ReLU(),
            Linear(hidden_nodes[1], 1),
            ReLU(),
        )
        self._target_model = deepcopy(self._model)
        self._loss = MSELoss()
        self._optimizer = Adam(params=self._model.parameters())
        self._batch_size = batch_size
        self._train_after_num_episodes = train_after_num_episodes
        self._update_target_after_trainings = update_target_after_trainings
        self._episodes = 0
        self._trainings = 0
        self._replay_buffer_input = []
        self._replay_buffer_target = []
        self._discount_rate = discount_rate
        self._espsilon = epsilon

    def sample(self, state: np.ndarray, eps_greedy: bool) -> int:
        if eps_greedy and np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._predict(self._model, state, 0)
            Q_1 = self._predict(self._model, state, 1)
            return np.argmax([Q_0, Q_1])

    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        if done:
            target = reward
        else:
            Q_0 = self._predict(self._target_model, new_state, 0)
            Q_1 = self._predict(self._target_model, new_state, 1)
            maxQ = np.max([Q_0, Q_1])
            target = reward + self._discount_rate * maxQ
        self._replay_buffer_input.append(np.concatenate((state, [action])))
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
    def _predict(self, model: Module, state: np.ndarray, action: int) -> float:
        state_action = np.concatenate((state, [action]))
        input = torch.tensor(state_action).float().view(1, -1)
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
