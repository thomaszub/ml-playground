from copy import copy

from policy import Policy, UniformRandomPolicy
from world import Field, FieldType, GridWorld, Position


def play_game(world: GridWorld, policy: Policy):
    trajectorie = []

    position = Position(0, 0)
    game_ended = False
    while not game_ended:
        action = policy.sample(position)
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
    policy = UniformRandomPolicy(world)

    discount_factor = 0.9

    trajectorie = play_game(world, policy)
    for entry in trajectorie:
        print(entry)

    possible_states = [field.position for field in fields]
    non_terminal_states = [
        field.position
        for field in fields
        if field.type != FieldType.WIN and field.type != FieldType.LOSE
    ]
    state_values = {state: 0.0 for state in possible_states}

    for it in range(0, 1000):
        for state in non_terminal_states:
            possible_actions = world.possible_actions(state)
            value = 0.0
            for action in possible_actions:
                new_state, reward = world.move(state, action)
                value += policy.func(state)(action) * (
                    reward + discount_factor * state_values[new_state.position]
                )
            state_values[state] = value
    print(state_values)


if __name__ == "__main__":
    main()
