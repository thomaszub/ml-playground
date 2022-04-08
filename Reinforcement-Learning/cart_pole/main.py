import gym
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class Agent:
    def __init__(
        self,
        observation_space: gym.Space[np.ndarray],
        action_space: gym.Space[int],
        discount_rate: float,
        epsilon: float,
    ):
        self._discount_rate = discount_rate
        self._espsilon = epsilon
        self._action_encoder = OneHotEncoder(categories=[[0, 1]])
        self._action_encoder.fit(np.array([[action_space.sample()]]))
        self._sampler = RBFSampler()
        self._sampler.fit([observation_space.sample()])
        self._model = SGDRegressor()
        self._model.partial_fit(
            self._input(observation_space.sample(), action_space.sample()),
            self._target(0.0),
        )

    def sample(self, observation: np.ndarray):
        if np.random.random() < self._espsilon:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            Q_0 = self._model.predict(self._input(observation, 0))
            Q_1 = self._model.predict(self._input(observation, 1))
            return 0 if Q_0 > Q_1 else 1

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
            Q_0 = self._model.predict(self._input(new_observation, 0))
            Q_1 = self._model.predict(self._input(new_observation, 1))
            maxQ = Q_0 if Q_0 > Q_1 else Q_1
            target = reward + self._discount_rate * maxQ
        self._model.partial_fit(self._input(observation, action), self._target(target))


def main() -> None:
    env = gym.make("CartPole-v1")

    agent = Agent(env.observation_space, env.action_space, 0.9, 0.1)
    for it in range(0, 100):
        steps = 0
        done = False
        observation = env.reset()
        while not done:
            steps += 1
            env.render()
            action = agent.sample(observation)
            new_observation, reward, done, info = env.step(action)
            agent.train(observation, action, reward, new_observation, done)
            observation = new_observation
        print(f"Try: {it}, steps: {steps}")

    env.close()


if __name__ == "__main__":
    main()
