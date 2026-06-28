# `dynamic_gestures/` — dynamic-gesture subsystem

Everything related to the dynamic-gesture model lives here: the live in-app
recognizer, the offline ML pipeline (collection → preprocessing → training →
verification), the datasets and the trained checkpoints.

## Layout

```
dynamic_gestures/
├── runtime/      In-app recognizer used by the live application.
│                 DynamicGestureRecognizer, SequenceBuffer, the model Protocol.
├── scripts/      Offline ML pipeline (run from the repository root).
├── data/
│   ├── raw/      Raw MediaPipe landmark recordings, one folder per gesture.
│   └── processed/  Preprocessed tensors (.pt), manifests and training reports.
└── models/       Trained PyTorch checkpoints (dynamic_bilstm.pt, dynamic_stgcn.pt, ...).
```

`data/` and `models/*.pt|*.onnx` are git-ignored (only `.gitkeep` is tracked).

## Relationship to the rest of the project

- The runtime recognizer was moved out of `oil_gestures/gestures/dynamic/`. That
  original location now contains thin compatibility shims that re-export from
  `dynamic_gestures.runtime`, so `app/main.py`, `app/app_config.py` and the tests
  keep importing `oil_gestures.gestures.dynamic.*` unchanged.
- The pipeline scripts still import the shared `oil_gestures` runtime package
  (`vision.*`, `core.*`). They add the repository root to `sys.path` themselves,
  so run them from anywhere; paths resolve against the repo root.
- MediaPipe `.task` models stay in `assets/models/mediapipe/` (shared with the app).

## Pipeline (run from the repository root)

```bash
# 1. Record raw landmark sequences (one gesture label per session)
python dynamic_gestures/scripts/collect_dynamic_dataset.py --label SWIPE_LEFT

# 2. Preprocess raw recordings -> processed tensors (BiLSTM + ST-GCN formats)
python dynamic_gestures/scripts/process_dynamic_dataset.py

# 3a. Train the BiLSTM baseline
python dynamic_gestures/scripts/train_dynamic_model.py

# 3b. Train the ST-GCN model
python dynamic_gestures/scripts/train_stgcn_model.py

# 4. Offline verification (no camera): per-gesture metrics + confusion
python dynamic_gestures/scripts/check_dynamic_model.py \
    --checkpoint dynamic_gestures/models/dynamic_bilstm.pt

# 5. Live camera test of a trained checkpoint (BiLSTM or ST-GCN, auto-detected)
python dynamic_gestures/scripts/test_dynamic_model.py \
    --checkpoint dynamic_gestures/models/dynamic_stgcn.pt
```

All scripts accept `--help`; defaults already point at the locations above.
