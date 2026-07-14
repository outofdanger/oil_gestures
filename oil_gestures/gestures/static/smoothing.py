from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from oil_gestures.core.enums import GestureName

# NOTE: not wired into StaticGestureRecognizer.update() by default - that
# method is contractually single-frame, immediate (see
# tests/test_static_recognizer.py: one update() call, one result). Smoothing
# here is opt-in for a caller that wants to trade a frame or two of latency
# for fewer single-frame flickers from MediaPipe's canned classifier (e.g.
# app/main.py could wrap static_recognizer.update() with this if flicker
# becomes a problem in practice).


@dataclass(frozen=True)
class StaticSmoothingConfig:
    window: int = 3
    min_agreement: int = 2


class StaticGestureSmoother:
    """
    Majority-vote smoothing over a short window of recognized static gestures.

    MediaPipe's canned classifier occasionally reports a different category
    (or none) for a single frame even while the hand pose is stable. Feed it
    one channel's gesture name (or None) per frame; it only reports a gesture
    as stable once it has appeared at least ``min_agreement`` times in the
    last ``window`` frames, otherwise returns None.
    """

    def __init__(self, config: StaticSmoothingConfig | None = None) -> None:
        self.config = config or StaticSmoothingConfig()
        self._history: Deque[GestureName | None] = deque(maxlen=self.config.window)

    def reset(self) -> None:
        self._history.clear()

    def smooth(self, gesture: GestureName | None) -> GestureName | None:
        self._history.append(gesture)
        if gesture is None:
            return None
        agreement = sum(1 for seen in self._history if seen == gesture)
        return gesture if agreement >= self.config.min_agreement else None
