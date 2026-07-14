from __future__ import annotations

from typing import Any

import numpy as np

from oil_gestures.vision.landmark_utils import LANDMARK_INDEX

# Mirrors dynamic_gestures/scripts/process_dynamic_dataset.py exactly - the
# live recognizer must use the identical normalization/packing the checkpoint
# was trained on, or its predictions are meaningless.
WRIST_INDEX = LANDMARK_INDEX["WRIST"]
MIDDLE_MCP_INDEX = LANDMARK_INDEX["MIDDLE_MCP"]
MIN_SCALE = 1e-4
EXPECTED_LANDMARK_COUNT = 21
EXPECTED_LANDMARK_DIMS = 3


def landmarks_to_array(landmarks: Any) -> np.ndarray | None:
    """Convert one frame's MediaPipe hand landmarks to a (21, 3) array.

    Same conversion dynamic_gestures/scripts/collect_dynamic_dataset.py uses
    while recording (``[[lm.x, lm.y, lm.z] for lm in points]``), so a live
    frame and a recorded training frame end up in the same representation.
    """
    points = list(landmarks) if landmarks is not None else []
    if len(points) != EXPECTED_LANDMARK_COUNT:
        return None
    return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32)


def resample_sequence(window: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, 21, 3) window to exactly target_len frames -
    a no-op when window already has target_len frames. Mirrors
    process_dynamic_dataset.py:resample_sequence, so a single shared rolling
    buffer (sized for the larger of two models' windows) can still feed each
    model its own window length exactly as it saw during training."""
    source_len = window.shape[0]
    if source_len == target_len:
        return window.astype(np.float32)
    if source_len == 1:
        return np.repeat(window, target_len, axis=0).astype(np.float32)

    source_t = np.linspace(0.0, 1.0, source_len)
    target_t = np.linspace(0.0, 1.0, target_len)
    flat = window.reshape(source_len, -1)
    resampled = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        resampled[:, col] = np.interp(target_t, source_t, flat[:, col])
    return resampled.reshape(target_len, window.shape[1], window.shape[2])


def normalize_sequence(window: np.ndarray) -> np.ndarray | None:
    """Center on the wrist and scale by the mean wrist->middle_mcp distance
    across the window. See process_dynamic_dataset.py:normalize_sequence -
    same formula, so the live window matches a resampled training clip."""
    wrist = window[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    centered = window - wrist
    distances = np.linalg.norm(centered[:, MIDDLE_MCP_INDEX, :], axis=-1)
    scale = float(np.mean(distances))
    if scale < MIN_SCALE:
        return None
    return (centered / scale).astype(np.float32)


def add_velocity_features(features: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]
    return np.concatenate([features, velocity], axis=-1)


def pack_bilstm_features(norm_image: np.ndarray, add_velocity: bool) -> np.ndarray:
    """(T, 21, 3) normalized landmarks -> (T, 63) or (T, 126) with velocity -
    the exact input layout BiLSTMGestureClassifier was trained on."""
    flattened = norm_image.reshape(norm_image.shape[0], -1)
    return add_velocity_features(flattened) if add_velocity else flattened


def pack_stgcn_features(norm_world: np.ndarray) -> np.ndarray:
    """(T, 21, 3) normalized WORLD landmarks -> (3, T, 21), the (C, T, V)
    layout STGCN.forward expects (caller adds the batch/person dims to make
    (1, C, T, V, 1)). ST-GCN was trained on world_landmarks, not image
    landmarks - see train_stgcn_model.py checkpoint metadata "node_features"."""
    return np.transpose(norm_world, (2, 0, 1))
