import numpy as np
from typing import Callable


def es_adv_fit(
    reward: Callable,
    num_params: int,
    generations: int,
    num_children: int,
    learning_rate: float,
    std_dev: float,
    seed: int = None,
):
    if seed != None:
        np.random.seed(seed)
    params = np.zeros(num_params)
    for generation in range(1, generations + 1):
        new_params = np.random.normal(params, std_dev, (num_children, num_params))
        rewards = np.array([reward(new_param) for new_param in new_params])
        advantages = (rewards - np.mean(rewards)) / np.std(rewards)
        pot_params = params + learning_rate / (num_children * std_dev ** 2) * np.dot(
            new_params.T, advantages
        )
        if reward(pot_params) > reward(params):
            params = pot_params
    return params


def es_sel_succ_fit(
    reward: Callable,
    num_params: int,
    generations: int,
    prob_success: float,
    seed: int = None,
):
    if seed != None:
        np.random.seed(seed)
    params = np.zeros(num_params)
    std_dev = 1.0
    reward_parent = reward(params)
    for generation in range(1, generations + 1):
        params_child = np.random.normal(params, std_dev, num_params)
        reward_child = reward(params_child)
        if reward_child > reward_parent:
            params = params_child
            reward_parent = reward_child
            std_dev *= np.exp(1.0 / 3.0)
        else:
            std_dev *= np.exp(-prob_success / (3.0 * (1.0 - prob_success)))
    return params
