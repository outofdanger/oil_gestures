from oil_gestures.gestures.decision.cooldown import CooldownGate


def test_cooldown_allows_first_fire() -> None:
    gate = CooldownGate(cooldown_seconds=0.5)

    assert gate.try_fire("INDEX_SQUEEZE", 1.0)


def test_cooldown_blocks_until_window_passes() -> None:
    gate = CooldownGate(cooldown_seconds=0.5)

    assert gate.try_fire("INDEX_SQUEEZE", 1.0)
    assert not gate.try_fire("INDEX_SQUEEZE", 1.25)
    assert gate.try_fire("INDEX_SQUEEZE", 1.50)


def test_cooldown_is_per_key() -> None:
    gate = CooldownGate(cooldown_seconds=0.5)

    assert gate.try_fire("INDEX_SQUEEZE", 1.0)
    assert gate.try_fire("INDEX_RELEASE", 1.1)
