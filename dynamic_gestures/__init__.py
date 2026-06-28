"""Self-contained dynamic-gesture subsystem.

Bundles everything related to the dynamic-gesture model in one place:

- ``runtime``  -- the live in-app recognizer (``DynamicGestureRecognizer``,
  ``SequenceBuffer``). The original location ``oil_gestures.gestures.dynamic``
  now re-exports from here for backward compatibility.
- ``scripts``  -- the offline ML pipeline: dataset collection, preprocessing,
  BiLSTM / ST-GCN training, and offline / live verification.
- ``data``     -- raw recordings and processed tensors.
- ``models``   -- trained PyTorch checkpoints.

See README.md for how to run each stage.
"""
