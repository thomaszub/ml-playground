from argparse import Action
from typing import Callable, Dict, Tuple

from environment.world import Field


def print_state_values(state_values: Dict[Field, float]) -> None:
    print("\n".join([str(it[0]) + " -> " + str(it[1]) for it in state_values.items()]))


def print_action_values(action_values: Dict[Tuple[Field, Action], float]) -> None:
    print("\n".join([str(it[0]) + " -> " + str(it[1]) for it in action_values.items()]))


def state_values_dict_to_function(
    state_values: Dict[Field, float]
) -> Callable[[Field], float]:
    return lambda s: state_values[s]
