from __future__ import annotations

import math
from dataclasses import dataclass

from oil_gestures.core.constants import DEFAULT_CURSOR_SMOOTHING_ALPHA
from oil_gestures.core.types import ScreenPosition


@dataclass(frozen=True)
class CursorSmoothingConfig:
    alpha: float = DEFAULT_CURSOR_SMOOTHING_ALPHA
    min_cutoff: float = 7.0
    beta: float = 0.080
    derivative_cutoff: float = 1.0
    dead_zone_points: float = 0.0
    max_speed_points_per_second: float = 50000.0


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.previous_raw = 0.0
        self.previous_filtered = 0.0

    def reset(self, value: float | None = None) -> None:
        self.initialized = value is not None
        self.previous_raw = 0.0 if value is None else value
        self.previous_filtered = 0.0 if value is None else value

    def apply(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.previous_raw = value
            self.previous_filtered = value
            return value

        filtered = alpha * value + (1.0 - alpha) * self.previous_filtered
        self.previous_raw = value
        self.previous_filtered = filtered
        return filtered


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.value_filter = LowPassFilter()
        self.derivative_filter = LowPassFilter()
        self.last_time: float | None = None

    @staticmethod
    def smoothing_alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, value: float | None = None) -> None:
        self.value_filter.reset(value)
        self.derivative_filter.reset(0.0 if value is not None else None)
        self.last_time = None

    def apply(self, value: float, now: float) -> float:
        if self.last_time is None:
            self.last_time = now
            return self.value_filter.apply(value, 1.0)

        dt = max(1.0 / 240.0, now - self.last_time)
        self.last_time = now
        previous = self.value_filter.previous_raw
        derivative = (value - previous) / dt if self.value_filter.initialized else 0.0
        derivative_alpha = self.smoothing_alpha(self.derivative_cutoff, dt)
        filtered_derivative = self.derivative_filter.apply(derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        alpha = self.smoothing_alpha(cutoff, dt)
        return self.value_filter.apply(value, alpha)


class CursorSmoother:
    def __init__(self, config: CursorSmoothingConfig) -> None:
        self.config = config
        self.alpha = max(0.0, min(1.0, config.alpha))
        self.x_filter = OneEuroFilter(config.min_cutoff, config.beta, config.derivative_cutoff)
        self.y_filter = OneEuroFilter(config.min_cutoff, config.beta, config.derivative_cutoff)
        self.current: tuple[float, float] | None = None
        self.last_time: float | None = None

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reset(self, current_position: tuple[float, float] | None = None) -> None:
        self.current = current_position
        self.last_time = None
        self.x_filter.reset(None)
        self.y_filter.reset(None)

    def apply(self, target: ScreenPosition, current_position: tuple[float, float] | None = None) -> ScreenPosition:
        now = target.timestamp
        if self.current is None:
            self.current = current_position or (float(target.x), float(target.y))
            self.last_time = now

        filtered_target = (
            self.x_filter.apply(float(target.x), now),
            self.y_filter.apply(float(target.y), now),
        )

        dt = 1.0 / 60.0 if self.last_time is None else max(1.0 / 240.0, now - self.last_time)
        self.last_time = now

        current = self.current
        delta = self._distance(current, filtered_target)
        if delta <= self.config.dead_zone_points:
            next_point = current
        else:
            max_step = self.config.max_speed_points_per_second * dt
            if delta > max_step:
                scale = max_step / delta
                next_point = (
                    current[0] + (filtered_target[0] - current[0]) * scale,
                    current[1] + (filtered_target[1] - current[1]) * scale,
                )
            else:
                next_point = filtered_target

        if self.alpha < 1.0:
            next_point = (
                current[0] + (next_point[0] - current[0]) * self.alpha,
                current[1] + (next_point[1] - current[1]) * self.alpha,
            )

        self.current = next_point
        return ScreenPosition(int(round(next_point[0])), int(round(next_point[1])), target.timestamp)
