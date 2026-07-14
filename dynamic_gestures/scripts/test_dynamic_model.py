from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# dynamic_gestures/scripts/<file>.py -> parents[2] is the repository root (home of
# the ``oil_gestures`` runtime package); SCRIPTS_DIR enables the sibling-script
# imports below (collect/process/check/train modules live next to this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra_path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from oil_gestures.core.constants import (  # noqa: E402
    DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD,
    DEFAULT_LANDMARK_COUNT,
    DEFAULT_LANDMARK_DIMENSIONS,
    DEFAULT_MEDIAPIPE_MODEL_PATH,
    DEFAULT_SAFE_EXIT_KEY,
)
from oil_gestures.vision.drawing import draw_landmarks  # noqa: E402
from oil_gestures.vision.frame_processor import bgr_to_rgb, mirror_frame  # noqa: E402

# Reuse the recording-time MediaPipe session + landmark extraction so the live
# landmarks are produced exactly like the training recordings, and the same
# preprocessing the dataset builder applied before training.
from collect_dynamic_dataset import (  # noqa: E402
    CameraSettings,
    HandLandmarkerSession,
    LoopFpsMeter,
    landmarks_to_array,
    mirrored_landmarks_for_display,
    open_camera,
)
from process_dynamic_dataset import (  # noqa: E402
    add_velocity_features,
    normalize_sequence,
    resample_sequence,
)
from check_dynamic_model import (  # noqa: E402
    GESTURE_DESCRIPTIONS,
    load_checkpoint,
    predict_logits,
    resolve_device,
)
from train_dynamic_model import TORCH_AVAILABLE, resolve_path  # noqa: E402
from train_stgcn_model import STGCN  # noqa: E402

if TORCH_AVAILABLE:
    import torch

WINDOW_NAME = "Dynamic gesture LIVE test"
QUIT_KEY = ord((DEFAULT_SAFE_EXIT_KEY or "q")[:1])
RESET_KEY = ord("r")
BASELINE_LABEL = "IDLE"
# Drop the rolling window if the hand disappears for this many consecutive frames,
# so two separate motions are never stitched into one sequence.
HAND_LOST_RESET_FRAMES = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera test for a trained dynamic-gesture model (BiLSTM or ST-GCN, "
        "auto-detected from the checkpoint). Perform a gesture in front of the camera and see "
        "the recognized class in real time."
    )
    parser.add_argument("--checkpoint", type=str, default="dynamic_gestures/models/dynamic_bilstm.pt")
    parser.add_argument(
        "--stgcn-checkpoint",
        type=str,
        default=None,
        help=(
            "ENSEMBLE mode: ST-GCN checkpoint used as the fast TRIGGER. When both "
            "--stgcn-checkpoint and --bilstm-checkpoint are given, the live recognizer runs "
            "the dual-model 'ST-GCN leads + BiLSTM confirms' logic instead of --checkpoint."
        ),
    )
    parser.add_argument(
        "--bilstm-checkpoint",
        type=str,
        default=None,
        help="ENSEMBLE mode: BiLSTM checkpoint used to CONFIRM/veto the ST-GCN trigger.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MEDIAPIPE_MODEL_PATH, help="MediaPipe HandLandmarker model.")
    parser.add_argument("--camera", type=int, default=0, help="Preferred camera index (tried first).")
    parser.add_argument(
        "--no-probe",
        dest="probe_fallback",
        action="store_false",
        default=True,
        help="Do not fall back to other camera indices when the preferred one fails to open.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        default=False,
        help="List camera indices that deliver frames, then exit (diagnostic).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Number of recent frames classified as one gesture (default: checkpoint target_len).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_DYNAMIC_CONFIDENCE_THRESHOLD,
        help="Confidence required to highlight a non-IDLE gesture as detected.",
    )
    parser.add_argument(
        "--veto-floor",
        type=float,
        default=0.20,
        help=(
            "ENSEMBLE only: minimum BiLSTM probability for the ST-GCN-proposed class to count "
            "as CONFIRMED; below it the trigger is VETOED (suppressed). Lower = trust ST-GCN "
            "more (snappier, more false fires); higher = stricter confirmation."
        ),
    )
    parser.add_argument(
        "--legacy-veto",
        action="store_true",
        default=False,
        help=(
            "Restore the legacy confirmation escape hatch: confirm a trigger when the proposed "
            "class merely ranks in BiLSTM's top-2, even at negligible probability. Kept only for "
            "A/B comparison against old checkpoints."
        ),
    )
    parser.add_argument(
        "--motion-gate",
        type=float,
        default=0.025,
        help=(
            "Minimum window motion (mean per-frame displacement of wrist-centered normalized "
            "landmarks) required to run recognition at all; below it the window is IDLE by "
            "definition. On this dataset IDLE sits near 0.013 and the quietest gestures near "
            "0.045. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="EMA factor over per-frame probabilities (0=frozen, 1=no smoothing).",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="How long a detected gesture stays latched on screen.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--no-display-mirror",
        dest="display_mirror",
        action="store_false",
        default=True,
        help="Disable mirroring of the preview (landmarks fed to the model are never mirrored).",
    )
    return parser.parse_args(argv)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def load_dynamic_model(
    checkpoint_path: Path, device: "torch.device"
) -> tuple["torch.nn.Module", dict[str, Any], str]:
    """Load either a BiLSTM or an ST-GCN checkpoint, auto-detected by model_type."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") == "STGCN":
        adjacency = np.asarray(checkpoint["adjacency"], dtype=np.float32)
        model = STGCN(
            in_channels=checkpoint["in_channels"],
            num_classes=checkpoint["num_classes"],
            adjacency=adjacency,
            base_channels=checkpoint["base_channels"],
            t_kernel=checkpoint["t_kernel"],
            dropout=checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint, "stgcn"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    return model, checkpoint, "bilstm"


def checkpoint_anchor(checkpoint: dict[str, Any]) -> str:
    """Wrist-anchoring mode the checkpoint was trained with (older checkpoints: per_frame)."""
    normalization = checkpoint.get("normalization") or {}
    return normalization.get("anchor", "per_frame")


def window_to_features(
    window: list[np.ndarray], target_len: int, add_velocity: bool, anchor: str
) -> np.ndarray | None:
    """Replicate the BiLSTM dataset preprocessing for one rolling window of image landmarks."""
    landmarks = np.stack(window).astype(np.float32)  # (T, 21, 3)
    if not np.isfinite(landmarks).all():
        return None
    resampled = resample_sequence(landmarks, target_len)
    normalized = normalize_sequence(resampled, anchor=anchor)
    if normalized is None:
        return None
    flattened = normalized.reshape(target_len, -1)
    features = add_velocity_features(flattened) if add_velocity else flattened
    return features.astype(np.float32)


def window_to_stgcn_input(
    window_world: list[np.ndarray],
    window_image: list[np.ndarray],
    target_len: int,
    anchor: str,
    channels: int,
) -> np.ndarray | None:
    """Replicate the ST-GCN dataset preprocessing -> (1, C, T, V, M=1).

    3-channel checkpoints see per-frame-normalized world landmarks (pose only).
    6-channel checkpoints additionally see first-frame-anchored image landmarks,
    which carry the hand's global trajectory across the window.
    """
    world = np.stack(window_world).astype(np.float32)  # (T, 21, 3)
    if not np.isfinite(world).all():
        return None
    pose = normalize_sequence(resample_sequence(world, target_len))
    if pose is None:
        return None
    if channels == 6:
        image = np.stack(window_image).astype(np.float32)
        if not np.isfinite(image).all():
            return None
        motion = normalize_sequence(resample_sequence(image, target_len), anchor=anchor)
        if motion is None:
            return None
        combined = np.concatenate([pose, motion], axis=-1)  # (T, V, 6)
    else:
        combined = pose
    graph = np.transpose(combined, (2, 0, 1))  # (C, T, V)
    return graph[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1, C, T, V, 1)


def window_pose_motion(window: list[np.ndarray], target_len: int) -> float:
    """Model-independent motion magnitude of the window: mean per-frame displacement
    of per-frame wrist-centered normalized image landmarks. On this dataset IDLE
    sits near 0.013 and the quietest gestures near 0.045."""
    landmarks = np.stack(window).astype(np.float32)
    if not np.isfinite(landmarks).all():
        return 0.0
    normalized = normalize_sequence(resample_sequence(landmarks, target_len))
    if normalized is None:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(normalized, axis=0), axis=-1)))


def classify_window(
    model: "torch.nn.Module",
    kind: str,
    window_entries: list[tuple[np.ndarray, np.ndarray]],
    target_len: int,
    add_velocity: bool,
    device: "torch.device",
    anchor: str,
    stgcn_channels: int = 3,
) -> np.ndarray | None:
    if kind == "stgcn":
        model_input = window_to_stgcn_input(
            [world for _, world in window_entries],
            [image for image, _ in window_entries],
            target_len,
            anchor,
            stgcn_channels,
        )
    else:
        flat = window_to_features([image for image, _ in window_entries], target_len, add_velocity, anchor)
        model_input = flat[np.newaxis, ...] if flat is not None else None
    if model_input is None:
        return None
    tensor = torch.from_numpy(model_input)
    logits = predict_logits(model, tensor, device, batch_size=1)[0]
    return softmax(logits)


def ensemble_decision(
    stgcn_probs: np.ndarray,
    bilstm_probs: np.ndarray,
    class_names: list[str],
    min_confidence: float,
    veto_floor: float,
    allow_top2: bool = False,
) -> tuple[str | None, bool]:
    """Live dual-model logic: ST-GCN LEADS (fast trigger), BiLSTM CONFIRMS (veto).

    ST-GCN reads the spatial hand pose, so it crosses the confidence threshold earlier
    than the motion/velocity-driven BiLSTM — it drives the trigger. BiLSTM suppresses
    ST-GCN's false fires: the proposed class is CONFIRMED only if BiLSTM gives it at
    least ``veto_floor`` probability; otherwise it is VETOED.

    ``allow_top2`` restores the legacy escape hatch (confirm when the lead merely ranks
    in BiLSTM's top-2). With 8 classes and IDLE absorbing most of the probability mass
    on a still hand, rank #2 can mean ~5% probability — the hatch confirms nearly every
    false trigger, which is why it is off by default.

    Returns (fired_label or None, vetoed).
    """
    lead = int(np.argmax(stgcn_probs))
    lead_label = class_names[lead]
    if lead_label == BASELINE_LABEL or float(stgcn_probs[lead]) < min_confidence:
        return None, False
    confirmed = float(bilstm_probs[lead]) >= veto_floor
    if allow_top2 and not confirmed:
        bilstm_top2 = {int(i) for i in np.argsort(bilstm_probs)[::-1][:2]}
        confirmed = lead in bilstm_top2
    if confirmed:
        return lead_label, False
    return None, True


def draw_ensemble_overlay(
    frame: np.ndarray,
    class_names: list[str],
    stgcn_probs: np.ndarray | None,
    bilstm_probs: np.ndarray | None,
    vetoed: bool,
    gated: bool = False,
) -> None:
    """Show both models' top class and the gate/confirm/veto state (ensemble mode only)."""
    if gated:
        lines = [("GATED (no motion)", (160, 160, 160))]
    elif stgcn_probs is None or bilstm_probs is None:
        return
    else:
        s_i = int(np.argmax(stgcn_probs))
        b_i = int(np.argmax(bilstm_probs))
        lines = [
            (f"ST-GCN lead : {class_names[s_i]:<24} {float(stgcn_probs[s_i]) * 100:4.0f}%", (120, 200, 255)),
            (f"BiLSTM conf : {class_names[b_i]:<24} {float(bilstm_probs[b_i]) * 100:4.0f}%", (200, 200, 120)),
            ("VETOED" if vetoed else "CONFIRMED", (60, 60, 255) if vetoed else (60, 230, 120)),
        ]
    x = max(14, frame.shape[1] - 360)
    y = frame.shape[0] - 96
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 26


def draw_live_overlay(
    frame: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None,
    buffer_len: int,
    window: int,
    hand_detected: bool,
    fps: float,
    min_confidence: float,
    latched_label: str | None,
) -> None:
    top_label = "—"
    top_conf = 0.0
    top3: list[tuple[str, float]] = []
    fired = False
    if probabilities is not None:
        ranking = np.argsort(probabilities)[::-1]
        top_label = class_names[int(ranking[0])]
        top_conf = float(probabilities[int(ranking[0])])
        top3 = [(class_names[int(i)], float(probabilities[int(i)])) for i in ranking[:3]]
        fired = top_label != BASELINE_LABEL and top_conf >= min_confidence

    header_color = (80, 220, 80) if hand_detected else (80, 80, 255)
    lines = [
        (f"{WINDOW_NAME}", (0, 215, 255)),
        (f"HAND {'YES' if hand_detected else 'NO'} | FPS {fps:4.1f} | buffer {buffer_len}/{window}", header_color),
        ("q quit | r reset", (235, 235, 235)),
    ]
    y = 26
    for line, color in lines:
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 26

    # Current top prediction, larger.
    pred_color = (80, 220, 80) if fired else (200, 200, 200)
    pred_text = f"{top_label}  {top_conf * 100:5.1f}%"
    cv2.putText(frame, pred_text, (14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, pred_text, (14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.95, pred_color, 2, cv2.LINE_AA)
    y += 48

    # Confidence bar for the top class.
    bar_x, bar_w, bar_h = 14, 260, 14
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * top_conf), y + bar_h), pred_color, -1)
    y += bar_h + 22

    for name, conf in top3:
        text = f"{name:<26} {conf * 100:5.1f}%"
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22

    if latched_label is not None:
        height = frame.shape[0]
        banner = f"DETECTED: {latched_label}"
        cv2.putText(frame, banner, (14, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, banner, (14, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 230, 120), 2, cv2.LINE_AA)


def probe_camera_indices(max_index: int = 5) -> list[int]:
    """Return camera indices that actually deliver a frame (macOS may prompt for permission)."""
    working: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    working.append(index)
        finally:
            capture.release()
    return working


def open_camera_with_fallback(
    preferred: int, width: int, height: int, fps: int, fourcc: str, probe_fallback: bool
) -> tuple[Any, int]:
    """Open the preferred camera, falling back to other indices if it fails to deliver frames."""
    candidates = [preferred] if not probe_fallback else list(dict.fromkeys([preferred, 0, 1, 2, 3]))
    last_error: Exception | None = None
    for index in candidates:
        settings = CameraSettings(device_id=index, width=width, height=height, fps=fps, fourcc=fourcc)
        try:
            return open_camera(settings), index
        except Exception as exc:  # CameraStream raises RuntimeError when a camera won't open
            last_error = exc
    raise SystemExit(
        f"Could not open any camera (tried indices {candidates}). Last error: {last_error}\n"
        "On macOS: System Settings > Privacy & Security > Camera -> enable your terminal/IDE, then "
        "FULLY quit and reopen it. Close other apps using the camera (Zoom, Photo Booth, browser). "
        "Run with --list-cameras to see which indices work, then pass --camera N."
    )


def main() -> int:
    if not TORCH_AVAILABLE:
        raise SystemExit("PyTorch is required. Install it with: python -m pip install torch")

    args = parse_args()

    if args.list_cameras:
        working = probe_camera_indices()
        if working:
            print(f"Cameras delivering frames at indices: {working}")
            print(f"Run: python scripts/test_dynamic_model.py --camera {working[0]} --checkpoint <path>")
        else:
            print(
                "No working camera found. On macOS, enable Camera for your terminal/IDE in "
                "System Settings > Privacy & Security > Camera, then fully quit and reopen it."
            )
        return 0

    device = resolve_device(args.device)

    ensemble_mode = bool(args.stgcn_checkpoint and args.bilstm_checkpoint)
    if bool(args.stgcn_checkpoint) ^ bool(args.bilstm_checkpoint):
        raise SystemExit("Ensemble mode needs BOTH --stgcn-checkpoint and --bilstm-checkpoint.")

    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise SystemExit(
            f"MediaPipe HandLandmarker model not found at: {model_path}. "
            "Place it at assets/models/mediapipe/hand_landmarker.task."
        )

    # Single-model state (used when not in ensemble mode).
    model = checkpoint = kind = None
    add_velocity = False
    anchor = "per_frame"
    stgcn_channels = 3
    # Ensemble state (used when in ensemble mode).
    stgcn_model = bilstm_model = None
    bilstm_target_len = 0
    bilstm_add_velocity = False
    stgcn_anchor = bilstm_anchor = "per_frame"

    if ensemble_mode:
        stgcn_path = resolve_path(args.stgcn_checkpoint)
        bilstm_path = resolve_path(args.bilstm_checkpoint)
        for label, path in (("--stgcn-checkpoint", stgcn_path), ("--bilstm-checkpoint", bilstm_path)):
            if not path.is_file():
                raise SystemExit(f"{label} not found at: {path}. Train the models first.")
        stgcn_model, stgcn_ckpt, stgcn_kind = load_dynamic_model(stgcn_path, device)
        bilstm_model, bilstm_ckpt, bilstm_kind = load_dynamic_model(bilstm_path, device)
        if stgcn_kind != "stgcn" or bilstm_kind != "bilstm":
            raise SystemExit(
                f"Ensemble expects an ST-GCN for --stgcn-checkpoint (got '{stgcn_kind}') and a "
                f"BiLSTM for --bilstm-checkpoint (got '{bilstm_kind}'). Check / swap the paths."
            )
        if list(stgcn_ckpt["class_names"]) != list(bilstm_ckpt["class_names"]):
            raise SystemExit("The two checkpoints have different class_names; they must match.")
        class_names: list[str] = list(stgcn_ckpt["class_names"])
        target_len: int = int(stgcn_ckpt["target_len"])
        bilstm_target_len = int(bilstm_ckpt["target_len"])
        bilstm_add_velocity = bool(bilstm_ckpt.get("add_velocity", False))
        stgcn_anchor = checkpoint_anchor(stgcn_ckpt)
        bilstm_anchor = checkpoint_anchor(bilstm_ckpt)
        stgcn_channels = int(stgcn_ckpt.get("in_channels", 3))
        window = args.window if args.window is not None else max(target_len, bilstm_target_len)
    else:
        checkpoint_path = resolve_path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise SystemExit(
                f"Checkpoint not found at: {checkpoint_path}. Train the model first with "
                "scripts/train_dynamic_model.py."
            )
        model, checkpoint, kind = load_dynamic_model(checkpoint_path, device)
        class_names = list(checkpoint["class_names"])
        target_len = int(checkpoint["target_len"])
        add_velocity = bool(checkpoint.get("add_velocity", False))
        anchor = checkpoint_anchor(checkpoint)
        stgcn_channels = int(checkpoint.get("in_channels", 3))
        window = args.window if args.window is not None else target_len

    if window < 2:
        raise SystemExit("--window must be at least 2 frames.")

    if ensemble_mode:
        print(f"ENSEMBLE mode | Device: {device} | window={window} frames")
        print(f"  ST-GCN (lead/trigger): {resolve_path(args.stgcn_checkpoint)}")
        print(f"  BiLSTM (confirm/veto): {resolve_path(args.bilstm_checkpoint)}")
        print(
            f"  min_confidence={args.min_confidence} | veto_floor={args.veto_floor} | "
            f"motion_gate={args.motion_gate} | legacy_veto={args.legacy_veto} | smoothing={args.smoothing}"
        )
    else:
        print(f"Loaded checkpoint: {resolve_path(args.checkpoint)}")
        print(f"Model: {kind.upper()} | Device: {device} | window={window} frames | target_len={target_len}")
    print("Recognized gestures:")
    for class_id, name in enumerate(class_names):
        print(f"  {class_id}: {name:<26} {GESTURE_DESCRIPTIONS.get(name, '')}")
    print("\nPerform a gesture in front of the camera. Press 'q' to quit, 'r' to reset the buffer.\n")

    buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=window)
    smoothed_probabilities: np.ndarray | None = None
    smoothed_stgcn: np.ndarray | None = None
    smoothed_bilstm: np.ndarray | None = None
    missed_frames = 0
    latched_label: str | None = None
    latched_until = 0.0

    camera_stream = None
    try:
        camera_stream, used_index = open_camera_with_fallback(
            args.camera, args.width, args.height, args.fps, args.fourcc, args.probe_fallback
        )
        print(f"Camera opened at index {used_index}. Press 'q' to quit, 'r' to reset.")
        with HandLandmarkerSession(model_path=model_path) as landmarker:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            fps_meter = LoopFpsMeter()

            while True:
                packet = camera_stream.read()
                if packet is None:
                    print("Camera returned no frame; stopping.")
                    break

                rgb_packet = bgr_to_rgb(packet)
                detection = landmarker.detect(rgb_packet.frame, packet.timestamp)
                fps = fps_meter.update()

                if detection.hand_detected:
                    missed_frames = 0
                    buffer.append(
                        (
                            landmarks_to_array(
                                detection.image_landmarks, DEFAULT_LANDMARK_COUNT, DEFAULT_LANDMARK_DIMENSIONS
                            ),
                            landmarks_to_array(
                                detection.world_landmarks, DEFAULT_LANDMARK_COUNT, DEFAULT_LANDMARK_DIMENSIONS
                            ),
                        )
                    )
                else:
                    missed_frames += 1
                    if missed_frames >= HAND_LOST_RESET_FRAMES:
                        buffer.clear()
                        smoothed_probabilities = None

                probabilities: np.ndarray | None = None
                fired_label: str | None = None
                ens_vetoed = False
                ens_gated = False
                ens_stgcn: np.ndarray | None = None
                ens_bilstm: np.ndarray | None = None
                if len(buffer) >= window:
                    entries = list(buffer)
                    # Motion gate: a window whose pose barely moves is IDLE by definition —
                    # don't even ask the models. Also drop the probability smoothing state so
                    # a following gesture starts from fresh probabilities, not stale ones.
                    if args.motion_gate > 0:
                        motion = window_pose_motion([image for image, _ in entries], target_len)
                        ens_gated = motion < args.motion_gate
                    if ens_gated:
                        smoothed_probabilities = None
                        smoothed_stgcn = None
                        smoothed_bilstm = None
                    elif ensemble_mode:
                        s = classify_window(
                            stgcn_model, "stgcn", entries, target_len, False, device, stgcn_anchor, stgcn_channels
                        )
                        b = classify_window(
                            bilstm_model, "bilstm", entries, bilstm_target_len, bilstm_add_velocity, device, bilstm_anchor
                        )
                        if s is not None and b is not None:
                            smoothed_stgcn = (
                                s
                                if smoothed_stgcn is None
                                else args.smoothing * s + (1.0 - args.smoothing) * smoothed_stgcn
                            )
                            smoothed_bilstm = (
                                b
                                if smoothed_bilstm is None
                                else args.smoothing * b + (1.0 - args.smoothing) * smoothed_bilstm
                            )
                            fired_label, ens_vetoed = ensemble_decision(
                                smoothed_stgcn,
                                smoothed_bilstm,
                                class_names,
                                args.min_confidence,
                                args.veto_floor,
                                allow_top2=args.legacy_veto,
                            )
                            probabilities = smoothed_stgcn  # ST-GCN is the displayed lead
                            ens_stgcn, ens_bilstm = smoothed_stgcn, smoothed_bilstm
                    else:
                        frame_probabilities = classify_window(
                            model, kind, entries, target_len, add_velocity, device, anchor, stgcn_channels
                        )
                        if frame_probabilities is not None:
                            if smoothed_probabilities is None:
                                smoothed_probabilities = frame_probabilities
                            else:
                                smoothed_probabilities = (
                                    args.smoothing * frame_probabilities
                                    + (1.0 - args.smoothing) * smoothed_probabilities
                                )
                            probabilities = smoothed_probabilities

                now = time.perf_counter()
                if ensemble_mode:
                    if fired_label is not None:
                        latched_label = fired_label
                        latched_until = now + args.hold_seconds
                elif probabilities is not None:
                    top_index = int(np.argmax(probabilities))
                    top_label = class_names[top_index]
                    if top_label != BASELINE_LABEL and float(probabilities[top_index]) >= args.min_confidence:
                        latched_label = top_label
                        latched_until = now + args.hold_seconds
                if latched_label is not None and now >= latched_until:
                    latched_label = None

                display_frame = packet.frame.copy()
                if args.display_mirror:
                    display_frame = mirror_frame(display_frame)
                if detection.hand_detected:
                    draw_landmarks(
                        display_frame,
                        mirrored_landmarks_for_display(detection.image_landmarks, args.display_mirror),
                    )

                draw_live_overlay(
                    display_frame,
                    class_names,
                    probabilities,
                    len(buffer),
                    window,
                    detection.hand_detected,
                    fps,
                    args.min_confidence,
                    latched_label,
                )
                if ensemble_mode:
                    draw_ensemble_overlay(display_frame, class_names, ens_stgcn, ens_bilstm, ens_vetoed, ens_gated)
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                if key == RESET_KEY:
                    buffer.clear()
                    smoothed_probabilities = None
                    smoothed_stgcn = None
                    smoothed_bilstm = None
                    latched_label = None
    finally:
        if camera_stream is not None:
            camera_stream.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
