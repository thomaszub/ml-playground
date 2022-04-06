from typing import Dict

from environment.world import Field


def print_state_values(state_values: Dict[Field, float]):
    print("\n".join([str(it[0]) + " -> " + str(it[1]) for it in state_values.items()]))
