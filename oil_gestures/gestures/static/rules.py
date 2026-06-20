from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from oil_gestures.core.enums import GestureName
from oil_gestures.vision.landmark_utils import normalized_distance, pinch_ratio


@dataclass(frozen=True)
class StaticRuleConfig:
    ok_pinch_ratio: float = 0.16
    finger_extension_margin: float = 0.035
    fist_curl_margin: float = 0.015
    open_palm_min_extended: int = 4


FINGER_JOINTS = (
    (8, 6, 5),
    (12, 10, 9),
    (16, 14, 13),
    (20, 18, 17),
)


def _has_indices(landmarks: Sequence, *indices: int) -> bool:
    return len(landmarks) > max(indices)


def finger_extended(landmarks: Sequence, tip: int, pip: int, mcp: int, margin: float) -> bool:
    if not _has_indices(landmarks, tip, pip, mcp):
        return False
    return float(landmarks[tip].y) + margin < float(landmarks[pip].y) < float(landmarks[mcp].y) + margin


def finger_curled(landmarks: Sequence, tip: int, pip: int, margin: float) -> bool:
    if not _has_indices(landmarks, tip, pip):
        return False
    return float(landmarks[tip].y) > float(landmarks[pip].y) - margin


def extended_fingers(landmarks: Sequence, config: StaticRuleConfig) -> int:
    return sum(
        1
        for tip, pip, mcp in FINGER_JOINTS
        if finger_extended(landmarks, tip, pip, mcp, config.finger_extension_margin)
    )


def is_ok_sign(landmarks: Sequence, config: StaticRuleConfig) -> bool:
    if not _has_indices(landmarks, 4, 8):
        return False
    if pinch_ratio(landmarks, 4, 8) > config.ok_pinch_ratio:
        return False
    return extended_fingers(landmarks, config) >= 2


def is_fist(landmarks: Sequence, config: StaticRuleConfig) -> bool:
    if not _has_indices(landmarks, 8, 12, 16, 20):
        return False
    curled = sum(1 for tip, pip, _mcp in FINGER_JOINTS if finger_curled(landmarks, tip, pip, config.fist_curl_margin))
    if curled < 3:
        return False
    return normalized_distance(landmarks[4], landmarks[8]) <= normalized_distance(landmarks[0], landmarks[9])


def is_open_palm(landmarks: Sequence, config: StaticRuleConfig) -> bool:
    return extended_fingers(landmarks, config) >= config.open_palm_min_extended


def classify_static_gesture(landmarks: Sequence, config: StaticRuleConfig | None = None) -> GestureName:
    rule_config = config or StaticRuleConfig()
    if is_ok_sign(landmarks, rule_config):
        return GestureName.OK_SIGN
    if is_fist(landmarks, rule_config):
        return GestureName.FIST
    if is_open_palm(landmarks, rule_config):
        return GestureName.OPEN_PALM
    return GestureName.UNKNOWN
