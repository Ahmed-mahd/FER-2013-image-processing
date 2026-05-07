"""
full_preprocessor.py
--------------------
Stage 3 — Full preprocessing pipeline for the entire FER2013 dataset.

Builds TWO complete tf.data pipelines:
  Pipeline A  —  CNN from Scratch
      Input  : 48x48 grayscale
      Norm   : [0, 1]  (rescale=1/255)
      Split  : data/train/ -> 85% train / 15% val  |  data/test/ -> test

  Pipeline B  —  Transfer Learning (MobileNetV2)
      Input  : 224x224 RGB
      Norm   : [-1, 1]  (tf.keras.applications.mobilenet_v2.preprocess_input)
      Split  : same directory split strategy

Both pipelines apply:
  - Data augmentation on training set only
      horizontal_flip, rotation_range=15, zoom_range=0.1,
      width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.8,1.2]
  - NO augmentation on validation / test sets

Imbalance handling:
  - Class weights  : computed via sklearn inverse-frequency weighting
  - Disgust boost  : augmentation multiplier x4 applied to minority class images
                     (saves extra augmented copies back to disk before generators)

Usage:
    from src.full_preprocessor import build_scratch_pipeline, build_tl_pipeline,
                                       compute_class_weights, augment_minority_class
"""

import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


# ── Directory constants ───────────────────────────────────────────────────────
TRAIN_DIR = Path("data/train")
TEST_DIR  = Path("data/test")

# ── Labels ────────────────────────────────────────────────────────────────────
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES     = len(EMOTION_CLASSES)

# ── Pipeline A config (CNN scratch) ──────────────────────────────────────────
SCRATCH_IMG_SIZE = (48, 48)
SCRATCH_CHANNELS = 1            # grayscale
SCRATCH_COLOR    = "grayscale"

# ── Pipeline B config (Transfer Learning) ────────────────────────────────────
TL_IMG_SIZE   = (224, 224)
TL_CHANNELS   = 3              # RGB
TL_COLOR      = "rgb"

# ── Training settings ─────────────────────────────────────────────────────────
BATCH_SIZE      = 32
VALIDATION_SPLIT = 0.15         # 15% of train/ becomes val
SEED            = 42

# ── Minority class augmentation ───────────────────────────────────────────────
MINORITY_CLASS      = "disgust"
MINORITY_MULTIPLIER = 4         # generate 4 extra augmented copies per original


# ─────────────────────────────────────────────────────────────────────────────
# Augment minority class — writes extra images to disk before generators run
# ─────────────────────────────────────────────────────────────────────────────
def augment_minority_class(
    train_dir: Path = TRAIN_DIR,
    emotion: str    = MINORITY_CLASS,
    multiplier: int = MINORITY_MULTIPLIER,
    seed: int       = SEED,
) -> int:
    """
    Generate `multiplier` augmented copies of every image in the minority class
    and save them alongside the originals in the same folder.

    Returns the number of new images written.
    """
    class_dir  = train_dir / emotion
    orig_files = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                  and not f.stem.startswith("aug_")]

    if not orig_files:
        raise FileNotFoundError(f"No original images found in {class_dir}")

    aug_gen = ImageDataGenerator(
        horizontal_flip    = True,
        rotation_range     = 20,
        zoom_range         = 0.15,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        brightness_range   = [0.75, 1.25],
        fill_mode          = "nearest",
    )

    rng       = np.random.default_rng(seed)
    written   = 0

    for img_path in orig_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize to 48x48 if needed, then convert to RGB for augmentation
        img = cv2.resize(img, (48, 48))
        img_arr = img[np.newaxis, :, :, np.newaxis].astype(np.float32)  # (1,48,48,1)

        for i in range(multiplier):
            aug_arr = next(aug_gen.flow(img_arr, batch_size=1, seed=int(rng.integers(9999))))
            aug_img = aug_arr[0, :, :, 0].astype(np.uint8)
            out_name = f"aug_{img_path.stem}_{i:02d}{img_path.suffix}"
            out_path = class_dir / out_name
            if not out_path.exists():
                cv2.imwrite(str(out_path), aug_img)
                written += 1

    return written


def remove_augmented_images(
    train_dir: Path = TRAIN_DIR,
    emotion: str    = MINORITY_CLASS,
) -> int:
    """Remove previously generated augmented copies (prefix: aug_)."""
    class_dir = train_dir / emotion
    removed   = 0
    for f in class_dir.iterdir():
        if f.stem.startswith("aug_"):
            f.unlink()
            removed += 1
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# Class weight computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(train_dir: Path = TRAIN_DIR) -> dict:
    """
    Compute inverse-frequency class weights from the training folder.
    Returns a dict {class_index: weight} to pass to model.fit(class_weight=...).
    """
    counts  = []
    indices = []
    for idx, emotion in enumerate(EMOTION_CLASSES):
        class_path = train_dir / emotion
        if class_path.exists():
            n = len([f for f in class_path.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            counts.append(n)
            indices.append(idx)

    labels_array = np.concatenate([
        np.full(n, i) for i, n in zip(indices, counts)
    ])

    weights = compute_class_weight(
        class_weight = "balanced",
        classes      = np.array(indices),
        y            = labels_array,
    )

    class_weight_dict = {i: float(w) for i, w in zip(indices, weights)}

    print("\n  Class Weights (inverse frequency):")
    for idx, emotion in enumerate(EMOTION_CLASSES):
        w = class_weight_dict.get(idx, 1.0)
        print(f"    [{idx}] {emotion:<10}: {w:.4f}")

    return class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — CNN Scratch (48x48 grayscale)
# ─────────────────────────────────────────────────────────────────────────────
def build_scratch_pipeline(
    train_dir: Path   = TRAIN_DIR,
    test_dir:  Path   = TEST_DIR,
    batch_size: int   = BATCH_SIZE,
    val_split:  float = VALIDATION_SPLIT,
    seed: int         = SEED,
) -> tuple:
    """
    Build training, validation, and test generators for the CNN scratch model.

    Returns:
        (train_gen, val_gen, test_gen) — Keras DirectoryIterator generators
    """
    # Augmented generator for TRAINING only
    train_datagen = ImageDataGenerator(
        rescale            = 1.0 / 255.0,
        validation_split   = val_split,
        horizontal_flip    = True,
        rotation_range     = 15,
        zoom_range         = 0.10,
        width_shift_range  = 0.10,
        height_shift_range = 0.10,
        brightness_range   = [0.80, 1.20],
        fill_mode          = "nearest",
    )

    # No augmentation for validation / test
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size  = SCRATCH_IMG_SIZE,
        color_mode   = SCRATCH_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "training",
        shuffle      = True,
        seed         = seed,
        classes      = EMOTION_CLASSES,
    )

    val_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size  = SCRATCH_IMG_SIZE,
        color_mode   = SCRATCH_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "validation",
        shuffle      = False,
        seed         = seed,
        classes      = EMOTION_CLASSES,
    )

    test_gen = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size  = SCRATCH_IMG_SIZE,
        color_mode   = SCRATCH_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        shuffle      = False,
        classes      = EMOTION_CLASSES,
    )

    print(f"\n  [Pipeline A — CNN Scratch  48x48 grayscale]")
    print(f"    Train samples      : {train_gen.samples:,}")
    print(f"    Validation samples : {val_gen.samples:,}")
    print(f"    Test samples       : {test_gen.samples:,}")
    print(f"    Batch size         : {batch_size}")
    print(f"    Steps/epoch (train): {train_gen.samples // batch_size}")
    print(f"    Class indices      : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — Transfer Learning (224x224 RGB, MobileNetV2 preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def build_tl_pipeline(
    train_dir: Path   = TRAIN_DIR,
    test_dir:  Path   = TEST_DIR,
    batch_size: int   = BATCH_SIZE,
    val_split:  float = VALIDATION_SPLIT,
    seed: int         = SEED,
) -> tuple:
    """
    Build training, validation, and test generators for the MobileNetV2
    transfer learning model.

    Pixel values are scaled to [-1, 1] via mobilenet_v2.preprocess_input.

    Returns:
        (train_gen, val_gen, test_gen) — Keras DirectoryIterator generators
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function = mobilenet_preprocess,
        validation_split       = val_split,
        horizontal_flip        = True,
        rotation_range         = 15,
        zoom_range             = 0.10,
        width_shift_range      = 0.10,
        height_shift_range     = 0.10,
        brightness_range       = [0.80, 1.20],
        fill_mode              = "nearest",
    )

    eval_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size  = TL_IMG_SIZE,
        color_mode   = TL_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "training",
        shuffle      = True,
        seed         = seed,
        classes      = EMOTION_CLASSES,
    )

    val_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size  = TL_IMG_SIZE,
        color_mode   = TL_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "validation",
        shuffle      = False,
        seed         = seed,
        classes      = EMOTION_CLASSES,
    )

    test_gen = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size  = TL_IMG_SIZE,
        color_mode   = TL_COLOR,
        batch_size   = batch_size,
        class_mode   = "categorical",
        shuffle      = False,
        classes      = EMOTION_CLASSES,
    )

    print(f"\n  [Pipeline B — MobileNetV2  224x224 RGB]")
    print(f"    Train samples      : {train_gen.samples:,}")
    print(f"    Validation samples : {val_gen.samples:,}")
    print(f"    Test samples       : {test_gen.samples:,}")
    print(f"    Batch size         : {batch_size}")
    print(f"    Steps/epoch (train): {train_gen.samples // batch_size}")
    print(f"    Class indices      : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen
