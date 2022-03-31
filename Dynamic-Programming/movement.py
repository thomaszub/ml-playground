from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Position:
    X: int
    Y: int


@dataclass(frozen=True)
class MoveAction(ABC):
    @abstractmethod
    def apply(self, position: Position) -> Position:
        pass

    def list() -> List["MoveAction"]:
        return [UpMoveAction(), DownMoveAction(), LeftMoveAction(), RightMoveAction()]


class UpMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X, position.Y + 1)


class DownMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X, position.Y - 1)


class LeftMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X - 1, position.Y)


class RightMoveAction(MoveAction):
    def apply(self, position: Position) -> Position:
        return Position(position.X + 1, position.Y)
