from copy import copy
from typing import Callable

from environment.world import Field, GridWorld
from environment.movement import MoveAction


def play(
    world: GridWorld,
    action_callback: Callable[[Field], MoveAction],
    start_field: Field,
    new_state_callback: Callable[[Field, float, Field], None] = None,
):
    trajectorie = []

    field = start_field
    game_ended = False
    while not game_ended:
        action = action_callback(field)
        new_field, reward = world.move(field, action)
        if new_state_callback != None:
            new_state_callback(field, reward, new_field)
        trajectorie.append((copy(field), action, reward, new_field))
        field = new_field
        game_ended = new_field.type.is_terminal()

    return trajectorie
