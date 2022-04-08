from typing import Dict
from tqdm import trange

import game
from environment.world import Field, FieldType, GridWorld, Position
from policy import Policy
from util import print_state_values, print_action_values
from agent import QAgent


def calc_state_values(
    world: GridWorld,
    policy: Policy,
    start_field: Field,
    learning_rate: float,
    discount_factor: float = 0.9,
    iterations: float = 2000,
) -> Dict[Field, float]:
    state_values = {state: 0.0 for state in world.get_fields()}

    def learn_callback(state: Field, reward: float, new_state: Field) -> None:
        state_values[state] += learning_rate * (
            reward + discount_factor * state_values[new_state] - state_values[state]
        )

    for _ in trange(0, iterations, desc="State values -> Playing game"):
        game.play(world, policy.sample, start_field, learn_callback)

    return state_values


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

    learning_rate = 0.1
    discount_factor = 0.9

    agent = QAgent(world, 0.1, learning_rate, discount_factor)

    for _ in trange(0, 2000, desc="SARSA -> Playing game"):
        agent.reset(start_field)
        game.play(world, agent.action_callback(), start_field, agent.learn_callback())

    state_values = calc_state_values(
        world, agent.get_policy(), start_field, learning_rate, discount_factor
    )
    print_state_values(state_values)
    print_action_values(agent.get_action_values())


if __name__ == "__main__":
    main()
