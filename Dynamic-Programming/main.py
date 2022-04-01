from copy import copy

from policy import Policy, UniformRandomPolicy
from world import Field, FieldType, GridWorld, Position


def play_game(world: GridWorld, policy: Policy, start_field: Field):
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
    policy = UniformRandomPolicy(world)

    discount_factor = 0.9

    trajectorie = play_game(world, policy, start_field)
    for entry in trajectorie:
        print(entry)

    non_terminal_states = [field for field in fields if not field.type.is_terminal()]
    state_values = {state: 0.0 for state in fields}

    for it in range(0, 1000):
        for state in non_terminal_states:
            possible_actions = world.possible_actions(state)
            value = 0.0
            for action in possible_actions:
                new_state, reward = world.move(state, action)
                value += policy.func(state)(action) * (
                    reward + discount_factor * state_values[new_state]
                )
            state_values[state] = value
    print(state_values)


if __name__ == "__main__":
    main()
