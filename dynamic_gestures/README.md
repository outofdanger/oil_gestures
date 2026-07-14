# `dynamic_gestures/` — dynamic-gesture training pipeline

This is the **offline** side of the dynamic-gesture model: collection →
preprocessing → training → verification, plus the datasets and the trained
checkpoints it produces. The **live, in-app recognizer is not here** - it
lives in `oil_gestures/gestures/dynamic/` (see below), which is the actual
package `app/main.py` imports.

## Layout

```
dynamic_gestures/
├── scripts/      Offline ML pipeline (run from the repository root).
├── data/
│   ├── raw/      Raw MediaPipe landmark recordings, one folder per gesture.
│   └── processed/  Preprocessed tensors (.pt), manifests and training reports.
└── models/       Trained PyTorch checkpoint variants (dynamic_bilstm.pt,
                  dynamic_stgcn.pt, ...) produced while experimenting. The one
                  actually used by the live app is copied to
                  assets/models/pytorch/ - see below.
```

`data/` is git-ignored (only `.gitkeep` is tracked). `models/*.pt` **are**
committed (despite earlier docs here claiming otherwise) - they're the
training experiment outputs kept for comparison/`check_dynamic_model.py`.

## Relationship to the rest of the project

- The live recognizer (`DynamicGestureRecognizer`, `SequenceBuffer`, the model
  Protocol, and `model_loader.py` which loads a trained checkpoint) lives in
  `oil_gestures/gestures/dynamic/`, not here. There used to be a duplicate
  runtime copy under `dynamic_gestures/runtime/` (this README previously
  claimed it was the canonical one and `oil_gestures/gestures/dynamic/` was
  just a re-export shim) - that was never actually true; both copies were
  full, independent implementations, and nothing outside this package
  imported `dynamic_gestures.runtime`. It has been removed as dead code.
- The live app runs a dual-model ensemble, not a single checkpoint: it loads
  both `assets/models/pytorch/dynamic_stgcn_transition.pt` and
  `assets/models/pytorch/dynamic_bilstm_transition.pt`. ST-GCN (reads
  world-landmark hand pose) leads as the fast trigger; BiLSTM (reads
  image-landmark motion/velocity) confirms or vetoes it - see
  `oil_gestures/gestures/dynamic/model_loader.py`'s `_ensemble_decision`,
  ported from this same `test_dynamic_model.py`'s `ensemble_decision`, where
  the scheme was first validated on a live camera. Config:
  `DynamicRecognizerConfig.stgcn_checkpoint_path` / `.bilstm_checkpoint_path`
  in `configs/gestures.yaml`.
- The `_transition` checkpoints add a 9th class **TRANSITION** (time-reversed
  swipes = the hand's return stroke). The runtime drops TRANSITION predictions
  (not a `GestureName`), so a return stroke reads as "no gesture" instead of
  the opposite swipe - this kills return-stroke false-opposites at the source.
  The older 8-class `_merged` pair (and `_no_point` variants) stay in
  `models/` / `assets/models/pytorch/` for reference/comparison, not loaded.
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
