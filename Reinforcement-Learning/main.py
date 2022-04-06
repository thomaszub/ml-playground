import numpy as np
from tqdm import trange

import game
from environment.world import Field, FieldType, GridWorld, Position
from policy import UniformRandomPolicy
from util import print_state_values


def main():
    start_field = Field(Position(0, 0), FieldType.FREE)
    fields = [
        start_field,
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

    discount_factor = 0.9
    learning_rate = 0.1

    policy = UniformRandomPolicy(world)

    state_values = {state: 0.0 for state in world.get_fields()}

    def learn_callback(state: Field, reward: float, new_state: Field) -> None:
        state_values[state] += learning_rate * (
            reward + discount_factor * state_values[new_state] - state_values[state]
        )

    for it in trange(0, 5000):
        game.play(world, policy, start_field, learn_callback)

    print_state_values(state_values)


if __name__ == "__main__":
    main()
