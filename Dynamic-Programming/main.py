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

    for it in range(0, 1):
        trajectorie = play_game(world, policy)
        for state in trajectorie:
            print(state)

        # TODO Update V


if __name__ == "__main__":
    main()
