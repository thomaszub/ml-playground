from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


@dataclass(frozen=True)
class Position:
    X: int
    Y: int


class FieldType(Enum):
    FREE = 1
    BLOCKED = 2
    WIN = 3
    LOSE = 4


@dataclass(frozen=True)
class Field:
    position: Position
    type: FieldType


class MoveAction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class GridWorld:
    def __init__(self, fields: List[Field]) -> None:
        self._fields = dict()
        for field in fields:
            self._fields[field.position] = field.type

    def possible_actions(self, position: Position) -> List[MoveAction]:
        upper = Position(position.X, position.Y + 1)
        bottom = Position(position.X, position.Y - 1)
        left = Position(position.X - 1, position.Y)
        right = Position(position.X + 1, position.Y)

        actions = []
        if not self._is_blocked(upper):
            actions.append(MoveAction.UP)
        if not self._is_blocked(bottom):
            actions.append(MoveAction.DOWN)
        if not self._is_blocked(left):
            actions.append(MoveAction.LEFT)
        if not self._is_blocked(right):
            actions.append(MoveAction.RIGHT)
        return actions

    def move(self, position: Position, action: MoveAction) -> Tuple[Field, float]:
        new_position = GridWorld._apply_action(position, action)
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

    @staticmethod
    def _apply_action(position: Position, action: MoveAction) -> Position:
        if action == MoveAction.UP:
            return Position(position.X, position.Y + 1)
        if action == MoveAction.DOWN:
            return Position(position.X, position.Y - 1)
        if action == MoveAction.RIGHT:
            return Position(position.X + 1, position.Y)
        return Position(position.X - 1, position.Y)
