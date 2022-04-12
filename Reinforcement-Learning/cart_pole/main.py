import gym
import numpy as np
from tqdm import trange

from agent import Agent
from model import DeepNNActionModel, LinearRBFActionModel


def play(env: gym.Env[np.ndarray, int], agent: Agent, render: bool, train: bool) -> int:
    sum_reward = 0
    done = False
    observation = env.reset()
    while not done:
        if render:
            env.render()
        action = agent.sample(observation, train)
        new_observation, reward, done, info = env.step(action)
        sum_reward += reward
        if train:
            agent.train(observation, action, reward, new_observation, done)
        observation = new_observation
    return sum_reward


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = Agent(
        model=DeepNNActionModel(hidden_nodes=(32, 16), replay_buffer_size=1),
        discount_rate=0.9,
        epsilon=0.1,
    )

    with trange(0, 1000, desc="Iteration") as titer:
        for _ in titer:
            reward = play(env, agent, False, True)
            titer.set_postfix(reward=reward)

    reward = play(env, agent, True, False)
    print(f"Reward: {reward}")

    env.close()


if __name__ == "__main__":
    main()
