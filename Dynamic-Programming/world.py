from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from movement import Position, MoveAction


class FieldType(Enum):
    FREE = 1
    BLOCKED = 2
    WIN = 3
    LOSE = 4


@dataclass(frozen=True)
class Field:
    position: Position
    type: FieldType


class GridWorld:
    def __init__(self, fields: List[Field]) -> None:
        self._fields = dict()
        for field in fields:
            self._fields[field.position] = field.type

    def possible_actions(self, position: Position) -> List[MoveAction]:
        action_position_pairs = [
            (act, act.apply(position)) for act in MoveAction.list()
        ]
        return [act for act, pos in action_position_pairs if not self._is_blocked(pos)]

    def move(self, position: Position, action: MoveAction) -> Tuple[Field, float]:
        new_position = action.apply(position)
        new_field_type = self._fields.get(new_position)
        if new_field_type != None:
            new_field = Field(new_position, new_field_type)
            if new_field_type == FieldType.FREE:
                return new_field, -0.1
            if new_field_type == FieldType.WIN:
                return new_field, 1.0
            if new_field_type == FieldType.LOSE:
                return new_field, -1.0
        return None, 0.0

    def _is_blocked(self, position: Position) -> bool:
        field = self._fields.get(position)
        if field != None:
            return field == FieldType.BLOCKED
        else:
            return True
