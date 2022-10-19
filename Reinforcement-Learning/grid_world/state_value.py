from typing import Callable
from tqdm import trange
from util import state_values_dict_to_function

import game
from environment.world import Field, GridWorld
from policy import Policy


def calc_state_values_by_dict(
    world: GridWorld,
    policy: Policy,
    start_field: Field,
    learning_rate: float,
    discount_factor: float = 0.9,
    iterations: float = 2000,
) -> Callable[[Field], float]:
    state_values = {state: 0.0 for state in world.get_fields()}

    def learn_callback(state: Field, reward: float, new_state: Field) -> None:
        state_values[state] += learning_rate * (
            reward + discount_factor * state_values[new_state] - state_values[state]
        )

    for _ in trange(0, iterations, desc="State values -> Playing game"):
        game.play(world, policy.sample, start_field, learn_callback)

    return state_values_dict_to_function(state_values)
