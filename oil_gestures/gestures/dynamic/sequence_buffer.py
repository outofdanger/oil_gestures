from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SequenceBufferConfig:
    max_length: int = 40


class SequenceBuffer(Generic[T]):
    def __init__(self, config: SequenceBufferConfig) -> None:
        if config.max_length <= 0:
            raise ValueError("SequenceBuffer max_length must be positive.")
        self.config = config
        self._items: Deque[T] = deque(maxlen=config.max_length)

    def append(self, item: T) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def clear(self) -> None:
        self._items.clear()

    def is_full(self) -> bool:
        return len(self._items) == self.config.max_length

    def __len__(self) -> int:
        return len(self._items)

    def as_list(self) -> list[T]:
        return list(self._items)
