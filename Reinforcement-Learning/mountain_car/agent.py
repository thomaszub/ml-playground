from copy import deepcopy
from typing import Protocol, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class Agent(Protocol):
    def sample(self, state: np.ndarray, eps_greedy: bool) -> int:
        ...

    def train_mode(self, active: bool) -> None:
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


class DeepQAgent(Agent):
    def __init__(
        self,
        discount_rate: float,
        epsilon: float,
        batch_size: int,
        train_after_steps: int,
        update_target_after_num_trainigs: int,
    ):
        self._discount_rate = discount_rate
        self._epsilon = epsilon
        self._train_mode = False
        self._replay_buffer_input = []
        self._replay_buffer_target = []
        self._model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self._target_model = None
        self._loss = torch.nn.MSELoss()
        self._optimizer = Adam(params=self._model.parameters())
        self._batch_size = batch_size
        self._steps = 0
        self._trainings = 0
        self._height_reached = 0.0
        self._train_after_steps = train_after_steps
        self._update_target_after_num_trainigs = update_target_after_num_trainigs

    def sample(self, state: np.ndarray) -> int:
        if self._train_mode and np.random.random() < self._epsilon:
            return np.random.choice([0, 1, 2])

        Q_0 = self._predict(self._model, state, self._action_int_to_array(0))
        Q_1 = self._predict(self._model, state, self._action_int_to_array(1))
        Q_2 = self._predict(self._model, state, self._action_int_to_array(2))
        return np.argmax([Q_0, Q_1, Q_2])

    def _action_int_to_array(self, action: int) -> np.ndarray:
        if action == 1:
            return np.array([0, 0])
        if action == 0:
            return np.array([1, 0])
        return np.array([0, 1])

    def train_mode(self, active: bool) -> None:
        self._train_mode = active
        self._model.requires_grad_(active)
        if active:
            self._target_model = deepcopy(self._model)
        else:
            self._target_model = None
            self._replay_buffer_input = []
            self._replay_buffer_target = []

    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> None:
        if not self._train_mode:
            return

        height = np.sin(3 * state[0]) * 0.45 + 0.55
        if height > self._height_reached:
            self._height_reached = height
        if done:
            target = reward + 100.0 * self._height_reached
            self._height_reached = 0.0
        else:
            Q_0 = self._predict(
                self._target_model, new_state, self._action_int_to_array(0)
            )
            Q_1 = self._predict(
                self._target_model, new_state, self._action_int_to_array(1)
            )
            Q_2 = self._predict(
                self._target_model, new_state, self._action_int_to_array(2)
            )
            max_Q = np.max([Q_0, Q_1, Q_2])
            target = reward + self._discount_rate * max_Q
        action_array = self._action_int_to_array(action)
        self._replay_buffer_input.append(np.concatenate((state, action_array)))
        self._replay_buffer_target.append([target])

        self._steps += 1
        if self._steps >= self._train_after_steps:
            self._steps = 0
            self._trainings += 1
            self._epsilon *= self._epsilon_decay
            self._train()
            self._replay_buffer_input = []
            self._replay_buffer_target = []
        if self._trainings >= self._update_target_after_num_trainigs:
            self._trainings = 0
            self._target_model = deepcopy(self._model)

    def _train(self) -> None:
        input = torch.tensor(np.array(self._replay_buffer_input)).float()
        target = torch.tensor(np.array(self._replay_buffer_target)).float()
        dataset = TensorDataset(input, target)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        for X, y in loader:
            self._optimizer.zero_grad()
            self._loss(self._model(X), y).backward()
            self._optimizer.step()

    @torch.no_grad()
    def _predict(self, model: torch.nn.Module, state: np.ndarray, action: List[int]):
        state_action = np.concatenate((state, action))
        input = torch.tensor(state_action).float().view(1, -1)
        return model(input).detach().numpy()
