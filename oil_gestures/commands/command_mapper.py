from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from oil_gestures.core.enums import CommandName, GestureName
from oil_gestures.core.types import CommandResult, GestureResult


@dataclass
class CommandMapper:
    mapping: Mapping[GestureName, CommandName] = field(default_factory=dict)
    default_command: CommandName = CommandName.NONE

    @classmethod
    def from_strings(cls, mapping: Mapping[str, str]) -> "CommandMapper":
        parsed: dict[GestureName, CommandName] = {}
        for gesture_name, command_name in mapping.items():
            parsed[GestureName(gesture_name)] = CommandName(command_name)
        return cls(parsed)

    def map(self, result: GestureResult) -> CommandResult:
        command = self.mapping.get(result.name, self.default_command)
        return CommandResult(
            command=command,
            source_gesture=result.name,
            confidence=result.confidence,
            timestamp=result.timestamp,
        )
