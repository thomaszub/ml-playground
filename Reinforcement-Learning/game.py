from copy import copy
from typing import Callable

from environment.world import Field, GridWorld
from policy import Policy


def play(
    world: GridWorld,
    policy: Policy,
    start_field: Field,
    new_state_callback: Callable[[Field, float, Field], None] = None,
):
    trajectorie = []

    field = start_field
    game_ended = False
    while not game_ended:
        action = policy.sample(field)
        new_field, reward = world.move(field, action)
        if new_state_callback != None:
            new_state_callback(field, reward, new_field)
        trajectorie.append((copy(field), action, reward, new_field))
        field = new_field
        game_ended = new_field.type.is_terminal()

    return trajectorie
