from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

Point = Tuple[float, float]


LANDMARK_INDEX = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_MCP": 5,
    "INDEX_PIP": 6,
    "INDEX_DIP": 7,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_PIP": 10,
    "MIDDLE_DIP": 11,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_PIP": 14,
    "RING_DIP": 15,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}


def landmark_index(name_or_index: str | int) -> int:
    if isinstance(name_or_index, int):
        return name_or_index

    key = name_or_index.strip().upper()
    if key not in LANDMARK_INDEX:
        known = ", ".join(sorted(LANDMARK_INDEX))
        raise ValueError(f"Unknown hand landmark '{name_or_index}'. Known landmarks: {known}")
    return LANDMARK_INDEX[key]


def as_landmark_list(hand_landmarks: Any) -> Sequence:
    if hand_landmarks is None:
        return []
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    return hand_landmarks


def xy(landmark: Any) -> Point:
    return (float(landmark.x), float(landmark.y))


def normalized_distance(a: Any, b: Any) -> float:
    ax, ay = xy(a)
    bx, by = xy(b)
    return math.hypot(ax - bx, ay - by)


def hand_scale(landmarks: Sequence) -> float:
    if not landmarks:
        return 0.001
    xs = [float(landmark.x) for landmark in landmarks]
    ys = [float(landmark.y) for landmark in landmarks]
    return max(0.001, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def pinch_ratio(landmarks: Sequence, first: int, second: int) -> float:
    if len(landmarks) <= max(first, second):
        return float("inf")
    return normalized_distance(landmarks[first], landmarks[second]) / hand_scale(landmarks)


def landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[int, int]:
    x_norm = max(0.0, min(1.0, float(landmark.x)))
    y_norm = max(0.0, min(1.0, float(landmark.y)))
    return (int(x_norm * width), int(y_norm * height))
