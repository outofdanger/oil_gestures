from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from oil_gestures.core.types import CommandResult


@dataclass(frozen=True)
class CommandHistoryEntry:
    command_result: CommandResult
    executed: bool


class CommandHistory:
    """Ring buffer of recently dispatched commands.

    Producer: commands.command_dispatcher.py. Consumer: a future
    ui.debug_panel.py (currently empty) - this only holds the data, no Qt.
    """

    def __init__(self, max_length: int = 50) -> None:
        self._entries: Deque[CommandHistoryEntry] = deque(maxlen=max_length)

    def record(self, command_result: CommandResult, *, executed: bool) -> None:
        self._entries.append(CommandHistoryEntry(command_result, executed))

    def recent(self, count: int | None = None) -> list[CommandHistoryEntry]:
        items = list(self._entries)
        return items if count is None else items[-count:]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
