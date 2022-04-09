import gym

from agent import Agent
from model import LinearRBFActionModel


def main() -> None:
    env = gym.make("CartPole-v1")

    agent = Agent(
        LinearRBFActionModel(env.observation_space, env.action_space, 0.1), 0.9, 0.1
    )
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
