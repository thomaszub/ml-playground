import gym


def main():
    env = gym.make("BipedalWalker-v3")
    env.reset()
    done = False
    sum_reward = 0
    while not done:
        env.render()
        new_state, reward, done, _ = env.step(env.action_space.sample())
        sum_reward += reward

    print(sum_reward)


if __name__ == "__main__":
    main()
