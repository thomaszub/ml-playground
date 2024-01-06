from typing import Dict, Tuple

from environment.movement import MoveAction
from environment.world import Field, GridWorld
from policy import Policy


def calc_state_values_for_policy(
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
            delta = max(abs(state_values[state] - value), delta)
            state_values[state] = value

    return state_values


def calc_state_values_optimal(
    world: GridWorld, discount_factor: float, eps: float
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
                value = max(value, (reward + discount_factor * state_values[new_state]))
            delta = max(abs(state_values[state] - value), delta)
            state_values[state] = value

    return state_values


def calc_action_values(
    world: GridWorld, state_values: Dict[Field, float], discount_factor: float
) -> Dict[Tuple[Field, MoveAction], float]:
    non_terminal_states = [
        field for field in world.get_fields() if not field.type.is_terminal()
    ]
    action_values = {}
    for field in non_terminal_states:
        possible_actions = world.possible_actions(field)
        for action in possible_actions:
            new_state, reward = world.move(field, action)
            action_values[(field, action)] = (
                reward + discount_factor * state_values.get(new_state, 0.0)
            )

    return action_values
