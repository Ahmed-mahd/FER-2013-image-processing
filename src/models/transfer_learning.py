"""
transfer_learning.py
--------------------
Stage 5 — EfficientNetB0 Transfer Learning model for FER2013.

Architecture:
  Base   : EfficientNetB0 (imagenet weights, include_top=False, input 224x224x3)
  Head   : GlobalAveragePooling2D → Dense(256,relu)+BN+Dropout(0.5)
           → Dense(128,relu)+Dropout(0.3) → Dense(7, softmax)

Phase 1 : base_model.trainable = False  (train head only, LR=1e-3, 25 epochs)
Phase 2 : last 50 layers unfrozen, LR=1e-5, 30 epochs (fine-tuning)

Why EfficientNetB0 over MobileNetV2:
  - Compound scaling (width + depth + resolution optimized together)
  - Better feature maps for fine-grained facial detail detection
  - Same parameter efficiency, slightly better accuracy (~70-74% vs ~65-70%)
  - Directly available in tf.keras.applications — no extra dependencies
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model


def build_transfer_learning_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int   = 7,
) -> tuple:
    """
    Build the full transfer learning model using EfficientNetB0.

    Returns:
        (model, base_model) — the full Keras Model and the EfficientNetB0 backbone.
        Call unfreeze_top_layers(base_model) before Phase 2 training.

    Phase 1 state: base_model frozen, only head is trainable.
    """
    # ── 1. Pre-trained backbone ────────────────────────────────────────────────
    # weights='imagenet' downloads automatically on first run (~29 MB)
    base_model = EfficientNetB0(
        input_shape = input_shape,
        include_top = False,          # strip the ImageNet classifier
        weights     = "imagenet",
    )
    base_model.trainable = False       # Phase 1: freeze entire backbone

    # ── 2. Custom classification head ─────────────────────────────────────────
    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)

    x = Dense(256, activation="relu", name="dense_256")(x)
    x = BatchNormalization(name="bn_256")(x)
    x = Dropout(0.5, name="drop_256")(x)

    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.3, name="drop_128")(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    # ── 3. Assemble ────────────────────────────────────────────────────────────
    model = Model(
        inputs  = base_model.input,
        outputs = outputs,
        name    = "EfficientNetB0_FER2013",
    )

    # Report trainable parameters
    total_params     = model.count_params()
    trainable_params = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    frozen_params = total_params - trainable_params
    print(f"\n  [Model] EfficientNetB0 + custom head")
    print(f"    Total parameters     : {total_params:,}")
    print(f"    Trainable (Phase 1)  : {trainable_params:,}  (head only)")
    print(f"    Frozen               : {frozen_params:,}  (EfficientNetB0 backbone)")

    return model, base_model


def unfreeze_top_layers(base_model, num_layers_to_unfreeze: int = 50) -> None:
    """
    Unfreeze the top N layers of the EfficientNetB0 backbone for Phase 2 fine-tuning.

    Strategy:
      - Keep BatchNormalization layers frozen throughout (critical for stability).
      - Only unfreeze the last `num_layers_to_unfreeze` non-BN layers.

    Note: EfficientNetB0 has ~237 layers vs MobileNetV2's 155.
          Unfreezing 50 layers exposes the high-level feature extractors
          (block6/7 — responsible for complex texture and shape features)
          without disturbing the low-level edge detectors (block1/2).
    """
    base_model.trainable = True   # re-enable gradient flow for the whole base

    # Freeze everything except the last N layers
    total_layers = len(base_model.layers)
    freeze_until = total_layers - num_layers_to_unfreeze

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Keep BN frozen to preserve running mean/variance learned on ImageNet
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(
        f"\n  [Phase 2] Unfrozen last {num_layers_to_unfreeze} layers of EfficientNetB0 "
        f"({trainable_count} layers now trainable, BN kept frozen)."
    )
