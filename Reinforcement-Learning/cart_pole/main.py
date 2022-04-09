import gym
from tqdm import trange

from agent import Agent
from model import LinearRBFActionModel


def main() -> None:
    env = gym.make("CartPole-v1")
    render = False
    agent = Agent(
        LinearRBFActionModel(env.observation_space, env.action_space, 0.1), 0.9, 0.1
    )

    with trange(0, 200, desc="Iteration") as titer:
        for _ in titer:
            steps = 0
            done = False
            observation = env.reset()
            while not done:
                steps += 1
                if render:
                    env.render()
                action = agent.sample(observation)
                new_observation, reward, done, info = env.step(action)
                agent.train(observation, action, reward, new_observation, done)
                observation = new_observation

            titer.set_postfix(steps=steps)
            titer.update()

    env.close()


if __name__ == "__main__":
    main()
