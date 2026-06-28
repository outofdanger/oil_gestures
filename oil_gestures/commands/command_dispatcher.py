from __future__ import annotations

from oil_gestures.commands.command_history import CommandHistory
from oil_gestures.core.enums import CommandName
from oil_gestures.core.types import CommandResult


class CommandDispatcher:
    """
    Executes a mapped CommandResult against the scene model and records it.

    Only handles commands that don't need extra targeting context here
    (EMERGENCY_STOP). Commands that act on the currently selected detail
    (OPEN_VALVE, which toggles open/close depending on current valve state)
    need that detail passed in explicitly - SimulatorController owns the
    "which detail, and is a menu armed for it" decision; this class only owns
    "how do I actually carry out a command once decided".
    """

    def __init__(self, history: CommandHistory | None = None) -> None:
        self.history = history or CommandHistory()

    def dispatch(self, command_result: CommandResult, model, detail=None) -> str | None:
        """Returns the concrete action taken (e.g. "open"/"close"/"emergency_stop"),
        or None if nothing happened."""
        command = command_result.command
        action: str | None = None

        if command == CommandName.EMERGENCY_STOP:
            model.emergency_stop()
            action = "emergency_stop"
        elif command == CommandName.OPEN_VALVE and detail is not None:
            action = "close" if detail.is_open else "open"
            model.execute_action(detail, action)

        self.history.record(command_result, executed=action is not None)
        return action
