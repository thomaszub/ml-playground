import gym
import numpy as np
from tqdm import trange

from agent import Agent, DeepQAgent

Env = gym.Env[np.ndarray, int]

env_id = "CartPole-v1"

max_reward = 500

max_train_iter = 5000


def play(agent: Agent, render: bool, train: bool) -> int:
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id, render_mode=None)
    sum_reward = 0
    terminated = False
    truncated = False
    state, _ = env.reset()
    while not (terminated or truncated):
        action = agent.sample(state, train)
        new_state, reward, terminated, truncated, _ = env.step(action)
        sum_reward += reward
        if train:
            agent.train(state, action, reward, new_state, terminated or truncated)
        state = new_state
    env.close()
    return sum_reward


def main() -> None:
    discount_rate = 0.99
    agent = DeepQAgent(
        hidden_nodes=(32, 32),
        batch_size=32,
        train_after_num_episodes=8,
        update_target_after_trainings=8,
        discount_rate=discount_rate,
        epsilon=0.1,
    )

    with trange(0, max_train_iter, desc="Iteration") as titer:
        for _ in titer:
            reward = play(agent, False, True)
            titer.set_postfix(reward=reward)
            if reward >= max_reward:
                break

    reward = play(agent, True, False)
    print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
