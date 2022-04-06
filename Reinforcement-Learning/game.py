from copy import copy

from policy import Policy
from environment.world import Field, GridWorld


def play(world: GridWorld, policy: Policy, start_field: Field):
    trajectorie = []

    field = start_field
    game_ended = False
    while not game_ended:
        action = policy.sample(field)
        new_field, reward = world.move(field, action)
        trajectorie.append((copy(field), action, reward, new_field))
        field = new_field
        game_ended = new_field.type.is_terminal()

    return trajectorie
