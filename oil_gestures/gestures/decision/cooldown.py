from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CooldownGate:
    cooldown_seconds: float
    _last_fired_by_key: dict[str, float] = field(default_factory=dict)

    def ready(self, key: str, timestamp: float) -> bool:
        previous = self._last_fired_by_key.get(key)
        if previous is None:
            return True
        return timestamp - previous >= self.cooldown_seconds

    def mark(self, key: str, timestamp: float) -> None:
        self._last_fired_by_key[key] = timestamp

    def try_fire(self, key: str, timestamp: float) -> bool:
        if not self.ready(key, timestamp):
            return False
        self.mark(key, timestamp)
        return True

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._last_fired_by_key.clear()
        else:
            self._last_fired_by_key.pop(key, None)
