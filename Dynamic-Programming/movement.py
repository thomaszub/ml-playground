from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Position:
    X: int
    Y: int


class MoveAction(ABC):
    @abstractmethod
    def apply(self, position: Position) -> Position:
        pass

    def list() -> List["MoveAction"]:
        return [UpMoveAction(), DownMoveAction(), LeftMoveAction(), RightMoveAction()]


class UpMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X, position.Y + 1)

    def __str__(self) -> str:
        return "UP"

    def __repr__(self) -> str:
        return "UP"


class DownMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X, position.Y - 1)

    def __str__(self) -> str:
        return "DOWN"

    def __repr__(self) -> str:
        return "DOWN"


class LeftMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X - 1, position.Y)

    def __str__(self) -> str:
        return "LEFT"

    def __repr__(self) -> str:
        return "LEFT"


class RightMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X + 1, position.Y)

    def __str__(self) -> str:
        return "RIGHT"

    def __repr__(self) -> str:
        return "RIGHT"
