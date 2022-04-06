from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from .movement import MoveAction, Position


class FieldType(Enum):
    FREE = auto()
    BLOCKED = auto()
    WIN = auto()
    LOSE = auto()

    def is_terminal(self) -> bool:
        return True if self == FieldType.WIN or self == FieldType.LOSE else False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Field:
    position: Position
    type: FieldType


class GridWorld:
    def __init__(self, fields: List[Field]) -> None:
        self._fields = dict()
        for field in fields:
            self._fields[field.position] = field.type

    def get_fields(self) -> List[Field]:
        return [Field(it[0], it[1]) for it in self._fields.items()]

    def possible_actions(self, field: Field) -> List[MoveAction]:
        action_position_pairs = [
            (act, act.apply(field.position)) for act in MoveAction.list()
        ]
        return [act for act, pos in action_position_pairs if not self._is_blocked(pos)]

    def move(self, field: Field, action: MoveAction) -> Tuple[Field, float]:
        new_position = action.apply(field.position)
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
