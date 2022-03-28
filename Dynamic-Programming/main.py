from copy import copy
from typing import Callable, List, Tuple
from world import Field, FieldType, GridWorld, MoveAction, Position
import numpy as np


def random_policy(
    world: GridWorld, position: Position
) -> Callable[[MoveAction], float]:
    possible_actions = world.possible_actions(position)
    return lambda x: 1.0 / len(possible_actions)


def sample_action(action_probs: List[Tuple[MoveAction, float]]) -> MoveAction:
    val = np.random.uniform()
    for act_prob in action_probs:
        val -= act_prob[1]
        if val < 0.0:
            return act_prob[0]


def play_game(world: GridWorld, policy: Callable[[MoveAction], float]):
    trajectorie = []

    position = Position(0, 0)
    game_ended = False
    while not game_ended:
        policy_func = policy(world, position)
        action_prob = [
            (act, policy_func(act)) for act in world.possible_actions(position)
        ]
        action = sample_action(action_prob)
        new_field, reward = world.move(position, action)

        trajectorie.append((copy(position), action, reward, new_field.position))
        position = new_field.position
        if new_field.type == FieldType.WIN or new_field.type == FieldType.LOSE:
            game_ended = True

    return trajectorie


def main():
    fields = [
        Field(Position(0, 0), FieldType.FREE),
        Field(Position(1, 0), FieldType.FREE),
        Field(Position(2, 0), FieldType.FREE),
        Field(Position(3, 0), FieldType.FREE),
        Field(Position(0, 1), FieldType.FREE),
        Field(Position(0, 2), FieldType.FREE),
        Field(Position(1, 2), FieldType.FREE),
        Field(Position(2, 1), FieldType.FREE),
        Field(Position(2, 2), FieldType.FREE),
        Field(Position(3, 1), FieldType.LOSE),
        Field(Position(3, 2), FieldType.WIN),
    ]

    world = GridWorld(fields)
    print("Start")

    policy = random_policy

    for it in range(0, 1):
        trajectorie = play_game(world, policy)
        for state in trajectorie:
            print(state)

        # TODO Update V


if __name__ == "__main__":
    main()
