from typing import Dict

from policy import Policy
from world import Field, GridWorld


def calc_state_values(
    world: GridWorld, policy: Policy, discount_factor: float, eps: float
) -> Dict[Field, float]:
    fields = world.get_fields()
    non_terminal_states = [field for field in fields if not field.type.is_terminal()]
    state_values = {state: 0.0 for state in fields}

    delta = eps + 0.1
    while delta >= eps:
        delta = 0.0
        for state in non_terminal_states:
            possible_actions = world.possible_actions(state)
            value = 0.0
            for action in possible_actions:
                new_state, reward = world.move(state, action)
                value += policy.func(state)(action) * (
                    reward + discount_factor * state_values[new_state]
                )
            delta = max(state_values[state] - value, delta)
            state_values[state] = value

    return state_values
