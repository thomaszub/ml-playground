import gymnasium as gym
import numpy as np
from tqdm import trange

from agent import Agent, DeepQAgent

Env = gym.Env[np.ndarray, int]

env_id = "MountainCar-v0"

max_train_iter = 5000


def play(agent: Agent, render: bool, train: bool) -> int:
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id, render_mode=None)
    agent.train_mode(train)
    sum_reward = 0
    terminated = False
    truncated = False
    state, _ = env.reset()
    while not (terminated or truncated):
        action = agent.sample(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        sum_reward += reward
        agent.train(state, action, reward, new_state, terminated or truncated)
        state = new_state
    env.close()
    return sum_reward


def main() -> None:
    agent = DeepQAgent(
        discount_rate=0.99,
        epsilon_decay=0.995,
        batch_size=64,
        train_after_steps=4096,
        update_target_after_num_trainigs=16,
    )

    with trange(0, max_train_iter, desc="Iteration") as titer:
        for _ in titer:
            reward = play(agent, False, True)
            titer.set_postfix(reward=reward)

    reward = play(agent, True, False)
    print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
