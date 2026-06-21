from __future__ import annotations

from oil_gestures.core.enums import GestureName
from oil_gestures.gestures.decision.gesture_toggle import GestureToggle, GestureToggleConfig


def _toggle(**overrides) -> GestureToggle:
    defaults = dict(target=GestureName.VICTORY, hold_seconds=0.5, cooldown_seconds=1.0)
    defaults.update(overrides)
    return GestureToggle(GestureToggleConfig(**defaults))


def test_fires_after_hold_duration() -> None:
    toggle = _toggle()

    assert toggle.update(GestureName.VICTORY, 0.0) is False  # dwell starts
    assert toggle.update(GestureName.VICTORY, 0.3) is False  # still holding
    assert toggle.update(GestureName.VICTORY, 0.5) is True   # dwell satisfied


def test_does_not_fire_before_hold_duration() -> None:
    toggle = _toggle()

    assert toggle.update(GestureName.VICTORY, 0.0) is False
    assert toggle.update(GestureName.VICTORY, 0.49) is False


def test_interrupted_hold_resets_dwell() -> None:
    toggle = _toggle()

    toggle.update(GestureName.VICTORY, 0.0)
    toggle.update(GestureName.VICTORY, 0.4)
    # A single non-target frame (flicker / wrong gesture / no hand) resets dwell.
    assert toggle.update(GestureName.FIST, 0.45) is False
    assert toggle.update(GestureName.VICTORY, 0.5) is False   # dwell restarts here
    assert toggle.update(GestureName.VICTORY, 0.9) is False   # only 0.4s held
    assert toggle.update(GestureName.VICTORY, 1.0) is True    # 0.5s held


def test_no_hand_resets_dwell() -> None:
    toggle = _toggle()

    toggle.update(GestureName.VICTORY, 0.0)
    assert toggle.update(None, 0.3) is False
    assert toggle.update(GestureName.VICTORY, 0.4) is False
    assert toggle.update(GestureName.VICTORY, 0.9) is True


def test_continuous_hold_fires_only_once() -> None:
    toggle = _toggle()

    assert toggle.update(GestureName.VICTORY, 0.0) is False
    assert toggle.update(GestureName.VICTORY, 0.5) is True
    # Keeps holding well past hold + cooldown: must NOT fire again without release.
    assert toggle.update(GestureName.VICTORY, 1.0) is False
    assert toggle.update(GestureName.VICTORY, 2.0) is False
    assert toggle.update(GestureName.VICTORY, 5.0) is False


def test_release_then_show_again_toggles_back() -> None:
    toggle = _toggle()

    toggle.update(GestureName.VICTORY, 0.0)
    assert toggle.update(GestureName.VICTORY, 0.5) is True  # first toggle (e.g. ON)

    # Release.
    assert toggle.update(None, 0.6) is False

    # Show again after cooldown -> second toggle (e.g. OFF).
    toggle.update(GestureName.VICTORY, 1.7)
    assert toggle.update(GestureName.VICTORY, 2.2) is True


def test_cooldown_blocks_rapid_retrigger() -> None:
    toggle = _toggle(cooldown_seconds=2.0)

    toggle.update(GestureName.VICTORY, 0.0)
    assert toggle.update(GestureName.VICTORY, 0.5) is True  # fires at t=0.5

    # Release and immediately re-show; dwell completes at t=1.3 but cooldown
    # (until t=2.5) must block it.
    toggle.update(None, 0.6)
    toggle.update(GestureName.VICTORY, 0.8)
    assert toggle.update(GestureName.VICTORY, 1.3) is False  # blocked by cooldown
    # Still holding once cooldown expires -> fires.
    assert toggle.update(GestureName.VICTORY, 2.5) is True


def test_disabled_toggle_never_fires() -> None:
    toggle = _toggle(enabled=False)

    assert toggle.update(GestureName.VICTORY, 0.0) is False
    assert toggle.update(GestureName.VICTORY, 0.5) is False
    assert toggle.update(GestureName.VICTORY, 5.0) is False


def test_progress_tracks_dwell() -> None:
    toggle = _toggle()

    toggle.update(GestureName.VICTORY, 0.0)
    assert toggle.progress == 0.0
    toggle.update(GestureName.VICTORY, 0.25)
    assert abs(toggle.progress - 0.5) < 1e-9
    toggle.update(None, 0.3)
    assert toggle.progress == 0.0


def test_reset_clears_state() -> None:
    toggle = _toggle()

    toggle.update(GestureName.VICTORY, 0.0)
    toggle.update(GestureName.VICTORY, 0.5)  # fires, now latched (not armed)
    toggle.reset()
    # After reset it behaves fresh: a new hold fires again.
    toggle.update(GestureName.VICTORY, 1.0)
    assert toggle.update(GestureName.VICTORY, 1.5) is True
