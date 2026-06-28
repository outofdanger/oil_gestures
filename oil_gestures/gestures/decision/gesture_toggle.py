from __future__ import annotations

from dataclasses import dataclass

from oil_gestures.core.enums import GestureName


@dataclass(frozen=True)
class GestureToggleConfig:
    """
    Configuration for a debounced, hold-to-confirm gesture toggle.

    enabled:
        When False the toggle never fires.
    target:
        Gesture that drives the toggle (e.g. GestureName.VICTORY).
    hold_seconds:
        The target gesture must be detected continuously for this long before a
        toggle fires (dwell). Any frame without the target resets the dwell.
    cooldown_seconds:
        Hard minimum delay between two consecutive toggles.
    """

    enabled: bool = True
    target: GestureName = GestureName.VICTORY
    hold_seconds: float = 0.5
    cooldown_seconds: float = 1.0


class GestureToggle:
    """
    Edge-triggered, false-positive-resistant gesture toggle.

    Feed it the currently recognized gesture (or None) once per frame together
    with the frame timestamp. ``update`` returns True exactly once when a toggle
    should occur. Protection layers, in order:

      1. Dwell        - the target gesture must be held for ``hold_seconds``;
                        a single non-target frame resets the dwell timer.
      2. Release latch - after firing, the gesture must be released (a non-target
                        frame) before another toggle can be armed, so one
                        continuous show produces exactly one toggle.
      3. Cooldown     - a hard floor of ``cooldown_seconds`` between fires.

    ``progress`` exposes the current dwell completion in the range [0, 1] for UI
    feedback.
    """

    def __init__(self, config: GestureToggleConfig | None = None) -> None:
        self.config = config or GestureToggleConfig()
        self._hold_start: float | None = None
        self._armed: bool = True
        self._last_fire_time: float | None = None
        self.progress: float = 0.0

    def reset(self) -> None:
        self._hold_start = None
        self._armed = True
        self._last_fire_time = None
        self.progress = 0.0

    def update(self, gesture: GestureName | None, timestamp: float) -> bool:
        if not self.config.enabled:
            self.progress = 0.0
            return False

        # Released (or different gesture / no hand): reset dwell and re-arm.
        if gesture != self.config.target:
            self._hold_start = None
            self._armed = True
            self.progress = 0.0
            return False

        # Target gesture is present.
        if not self._armed:
            # Waiting for a release before another toggle may be armed.
            self.progress = 0.0
            return False

        if self._hold_start is None:
            self._hold_start = timestamp

        held = max(0.0, timestamp - self._hold_start)
        hold_needed = max(0.0, self.config.hold_seconds)
        self.progress = 1.0 if hold_needed == 0.0 else min(1.0, held / hold_needed)

        if held < hold_needed:
            return False

        # Dwell satisfied; enforce the cooldown floor between fires.
        if (
            self._last_fire_time is not None
            and timestamp - self._last_fire_time < self.config.cooldown_seconds
        ):
            return False

        self._last_fire_time = timestamp
        self._armed = False
        self._hold_start = None
        self.progress = 0.0
        return True
