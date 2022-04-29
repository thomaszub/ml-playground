import gym
import numpy as np
from tqdm import trange

from agent import Agent, DeepQAgent

Env = gym.Env[np.ndarray, int]


def play(env: Env, agent: Agent, render: bool, train: bool) -> int:
    sum_reward = 0
    done = False
    state = env.reset()
    while not done:
        if render:
            env.render()
        action = agent.sample(state, train)
        new_state, reward, done, _ = env.step(action)
        sum_reward += reward
        if train:
            agent.train(state, action, reward, new_state, done)
        state = new_state
    return sum_reward


def main() -> None:
    env = gym.make("CartPole-v1")
    discount_rate = 0.99
    agent = DeepQAgent(
        hidden_nodes=(32, 32),
        batch_size=32,
        train_after_num_episodes=8,
        update_target_after_trainings=8,
        discount_rate=discount_rate,
        epsilon=0.1,
    )

    with trange(0, 5000, desc="Iteration") as titer:
        for _ in titer:
            reward = play(env, agent, False, True)
            titer.set_postfix(reward=reward)

    reward = play(env, agent, True, False)
    print(f"Reward: {reward}")

    env.close()


if __name__ == "__main__":
    main()
