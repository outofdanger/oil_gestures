from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbabilitySmoothingConfig:
    # EMA weight for the newest frame; 1.0 disables smoothing entirely.
    alpha: float = 0.5


class ProbabilitySmoother:
    """
    Exponential moving average over a model's per-class probability vector.

    The dynamic-gesture classifier re-runs every frame on a sliding window
    that overlaps almost entirely with the previous frame's window, so
    consecutive predictions should largely agree - but softmax probabilities
    still jitter near class boundaries, occasionally flipping the argmax for
    a single frame. Smoothing the probability vector itself (not just the
    final label, the way gestures/static/smoothing.py does for the canned
    static classifier, which has no probability vector to work with) damps
    that without adding the extra latency a majority-vote over labels would.
    """

    def __init__(self, config: ProbabilitySmoothingConfig | None = None) -> None:
        self.config = config or ProbabilitySmoothingConfig()
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def smooth(self, probabilities: np.ndarray) -> np.ndarray:
        if self._state is None or self._state.shape != probabilities.shape:
            self._state = probabilities.copy()
            return self._state
        alpha = self.config.alpha
        self._state = alpha * probabilities + (1.0 - alpha) * self._state
        return self._state
