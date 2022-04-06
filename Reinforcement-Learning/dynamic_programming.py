from tqdm import trange

import game
from environment.world import Field, FieldType, GridWorld, Position
from evaluation import state_values_for_policy, state_values_optimal
from policy import ArgMaxPolicy, Policy, UniformRandomPolicy
from util import print_state_values


def policy_by_policy_iteration(
    world: GridWorld, start_field: Field, discount_factor: float, eps: float
) -> Policy:
    random_policy = UniformRandomPolicy(world)
    policy = ArgMaxPolicy(world, 0.9)

    # initialize state values with random policy to avoid endless loop
    game.play(world, random_policy, start_field)
    state_values = state_values_for_policy(world, random_policy, discount_factor, eps)
    policy.update(state_values)

    # Optimize policy by iteration
    for run in trange(0, 100, desc="Run: "):
        state_values = state_values_for_policy(world, policy, discount_factor, eps)
        policy.update(state_values)

    return policy


def policy_by_value_iteration(
    world: GridWorld, discount_factor: float, eps: float
) -> Policy:
    state_values = state_values_optimal(world, discount_factor, eps)
    return ArgMaxPolicy(world, 0.9, state_values)


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
    eps = pow(10, -6)

    # policy = policy_by_policy_iteration(world, start_field, discount_factor, eps)
    policy = policy_by_value_iteration(world, discount_factor, eps)

    trajectorie = game.play(world, policy, start_field)
    for entry in trajectorie:
        print(entry)
    state_values = state_values_for_policy(world, policy, discount_factor, eps)

    print_state_values(state_values)


if __name__ == "__main__":
    main()
