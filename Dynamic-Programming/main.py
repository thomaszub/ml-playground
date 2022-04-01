from copy import copy

from evaluation import calc_state_values
from policy import Policy, UniformRandomPolicy, UpdateablePolicy
from world import Field, FieldType, GridWorld, Position
from tqdm import trange


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
    random_policy = UniformRandomPolicy(world)
    policy = UpdateablePolicy(world, 0.9)

    discount_factor = 0.9
    eps = pow(10, -6)

    # initialize state values with random policy to avoid endless loop
    play_game(world, random_policy, start_field)
    state_values = calc_state_values(world, random_policy, discount_factor, eps)
    policy.update(state_values)

    # Optimize policy by iteration
    for run in trange(0, 100, desc="Run: "):
        play_game(world, policy, start_field)
        state_values = calc_state_values(world, policy, discount_factor, eps)
        policy.update(state_values)

    trajectorie = play_game(world, policy, start_field)
    for entry in trajectorie:
        print(entry)
    state_values = calc_state_values(world, policy, discount_factor, eps)

    print("\n".join([str(it[0]) + " -> " + str(it[1]) for it in state_values.items()]))


if __name__ == "__main__":
    main()
