from __future__ import annotations

from typing import Sequence

import cv2

from oil_gestures.core.types import PointerPosition, ScreenPosition
from oil_gestures.vision.landmark_utils import as_landmark_list, landmark_to_pixel


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw_landmarks(frame, hand_landmarks: Sequence, pointer_index: int = 5) -> None:
    landmarks = as_landmark_list(hand_landmarks)
    height, width = frame.shape[:2]
    points = [landmark_to_pixel(landmark, width, height) for landmark in landmarks]

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (70, 220, 70), 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        color = (0, 170, 255) if index == pointer_index else (0, 255, 255)
        radius = 7 if index == pointer_index else 4
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def draw_pointer_cursor(frame, pointer: PointerPosition | None, pressed: bool = False) -> None:
    if pointer is None or not pointer.visible:
        return

    height, width = frame.shape[:2]
    x = int(max(0.0, min(1.0, pointer.x)) * width)
    y = int(max(0.0, min(1.0, pointer.y)) * height)
    color = (40, 40, 255) if pressed else (40, 220, 80)
    cv2.circle(frame, (x, y), 11, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 9, color, 2, cv2.LINE_AA)
    cv2.line(frame, (x - 18, y), (x + 18, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 18), (x, y + 18), color, 2, cv2.LINE_AA)


def draw_overlay(
    frame,
    status: str,
    pointer: PointerPosition | None = None,
    screen_position: ScreenPosition | None = None,
    action_status: str | None = None,
    click_status: str | None = None,
    gesture_status: str | None = None,
    feature_status: str | None = None,
    pressed: bool | None = None,
    mouse_status: str | None = None,
    fps: float = 0.0,
    paused: bool = False,
) -> None:
    color = (
        (0, 200, 255)
        if paused
        else ((80, 220, 80) if status in {"ACTIVE", "CURSOR ON", "GESTURE RECOGNITION"} else (80, 80, 255))
    )
    lines = [f"{status} | FPS {fps:4.1f}", "Space/p pause | q/Esc quit"]

    if gesture_status:
        lines.append(f"gesture {gesture_status}")
    if feature_status:
        lines.append(f"feature {feature_status}")
    if pointer is not None and pointer.visible:
        lines.append(f"pointer {pointer.x:.2f}, {pointer.y:.2f}")
    if screen_position is not None:
        lines.append(f"cursor target {screen_position.x}, {screen_position.y}")
    if pressed is not None:
        lines.append(f"pressed {'YES' if pressed else 'NO'}")
    if mouse_status:
        lines.append(f"mouse {mouse_status}")
    if action_status:
        lines.append(f"action {action_status}")
    elif click_status:
        lines.append(f"click {click_status}")

    y = 28
    for index, line in enumerate(lines):
        line_color = color if index == 0 else (235, 235, 235)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
        y += 28
