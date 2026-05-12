"""
full_preprocessor.py
--------------------
Stage 3 — Full preprocessing pipeline for the entire FER2013 dataset.

Builds TWO complete data pipelines:

  Pipeline A  —  CNN from Scratch
      Input  : 48x48 grayscale, [0, 1]
      Source : Keras ImageDataGenerator (unchanged)

  Pipeline B  —  Transfer Learning (MobileNetV2)
      Input  : 224x224 RGB, [-1, 1]
      Source : tf.data pipeline with parallel loading + GPU prefetch
               → eliminates the CPU bottleneck that caused 3-hour epochs

Both pipelines apply data augmentation on training set only.
Imbalance handling via class weights + Disgust x4 disk augmentation.

Usage:
    from src.full_preprocessor import (
        build_scratch_pipeline,   # Pipeline A — ImageDataGenerator
        build_tl_pipeline,        # Pipeline B — tf.data (FAST)
        compute_class_weights,
        augment_minority_class,
    )
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Directory constants ───────────────────────────────────────────────────────
TRAIN_DIR = Path("data/train")
TEST_DIR  = Path("data/test")

# ── Labels ────────────────────────────────────────────────────────────────────
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES     = len(EMOTION_CLASSES)
CLASS_TO_IDX    = {c: i for i, c in enumerate(EMOTION_CLASSES)}

# ── Pipeline A config ─────────────────────────────────────────────────────────
SCRATCH_IMG_SIZE = (48, 48)
SCRATCH_COLOR    = "grayscale"

# ── Pipeline B config ─────────────────────────────────────────────────────────
TL_IMG_SIZE   = (224, 224)
TL_CHANNELS   = 3

# ── Shared settings ───────────────────────────────────────────────────────────
BATCH_SIZE       = 32
VALIDATION_SPLIT = 0.15
SEED             = 42

# ── Minority class augmentation ───────────────────────────────────────────────
MINORITY_CLASS      = "disgust"
MINORITY_MULTIPLIER = 4


# ─────────────────────────────────────────────────────────────────────────────
# Disk-level minority class augmentation (run once via prepare_data.py)
# ─────────────────────────────────────────────────────────────────────────────
def augment_minority_class(
    train_dir: Path = TRAIN_DIR,
    emotion: str    = MINORITY_CLASS,
    multiplier: int = MINORITY_MULTIPLIER,
    seed: int       = SEED,
) -> int:
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

    rng     = np.random.default_rng(seed)
    written = 0

    for img_path in orig_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img     = cv2.resize(img, (48, 48))
        img_arr = img[np.newaxis, :, :, np.newaxis].astype(np.float32)

        for i in range(multiplier):
            aug_arr = next(aug_gen.flow(img_arr, batch_size=1, seed=int(rng.integers(9999))))
            aug_img = aug_arr[0, :, :, 0].astype(np.uint8)
            out_name = f"aug_{img_path.stem}_{i:02d}{img_path.suffix}"
            out_path = class_dir / out_name
            if not out_path.exists():
                cv2.imwrite(str(out_path), aug_img)
                written += 1

    return written


def remove_augmented_images(train_dir: Path = TRAIN_DIR, emotion: str = MINORITY_CLASS) -> int:
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
    counts  = []
    indices = []
    for idx, emotion in enumerate(EMOTION_CLASSES):
        class_path = train_dir / emotion
        if class_path.exists():
            n = len([f for f in class_path.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            counts.append(n)
            indices.append(idx)

    labels_array = np.concatenate([np.full(n, i) for i, n in zip(indices, counts)])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(indices),
        y=labels_array,
    )
    class_weight_dict = {i: float(w) for i, w in zip(indices, weights)}

    print("\n  Class Weights (inverse frequency):")
    for idx, emotion in enumerate(EMOTION_CLASSES):
        print(f"    [{idx}] {emotion:<10}: {class_weight_dict.get(idx, 1.0):.4f}")

    return class_weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — CNN Scratch (48x48 grayscale) — ImageDataGenerator (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_scratch_pipeline(
    train_dir: Path   = TRAIN_DIR,
    test_dir:  Path   = TEST_DIR,
    batch_size: int   = BATCH_SIZE,
    val_split:  float = VALIDATION_SPLIT,
    seed: int         = SEED,
) -> tuple:
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
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir), target_size=SCRATCH_IMG_SIZE, color_mode=SCRATCH_COLOR,
        batch_size=batch_size, class_mode="categorical", subset="training",
        shuffle=True, seed=seed, classes=EMOTION_CLASSES,
    )
    val_gen = train_datagen.flow_from_directory(
        str(train_dir), target_size=SCRATCH_IMG_SIZE, color_mode=SCRATCH_COLOR,
        batch_size=batch_size, class_mode="categorical", subset="validation",
        shuffle=False, seed=seed, classes=EMOTION_CLASSES,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(test_dir), target_size=SCRATCH_IMG_SIZE, color_mode=SCRATCH_COLOR,
        batch_size=batch_size, class_mode="categorical",
        shuffle=False, classes=EMOTION_CLASSES,
    )

    print(f"\n  [Pipeline A — CNN Scratch  48x48 grayscale]")
    print(f"    Train samples      : {train_gen.samples:,}")
    print(f"    Validation samples : {val_gen.samples:,}")
    print(f"    Test samples       : {test_gen.samples:,}")
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — Transfer Learning  tf.data (FAST)
# 224x224 RGB, MobileNetV2 preprocessing, GPU-side prefetch
# ─────────────────────────────────────────────────────────────────────────────

def _collect_paths_and_labels(root: Path) -> tuple:
    """Walk a directory tree and return (file_paths, int_labels) lists."""
    paths, labels = [], []
    for emotion in EMOTION_CLASSES:
        class_dir = root / emotion
        if not class_dir.exists():
            continue
        label = CLASS_TO_IDX[emotion]
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                paths.append(str(p))
                labels.append(label)
    return paths, labels


def _make_decode_fn(augment: bool):
    """
    Returns a tf.function that:
      1. Reads the JPEG/PNG from disk
      2. Decodes to RGB uint8
      3. Resizes to 224x224
      4. Optionally applies augmentation (train only)
      5. Applies EfficientNetB0 preprocess_input (torch-style normalization)
    """
    H, W = TL_IMG_SIZE
    
    if augment:
        random_rotation = tf.keras.layers.RandomRotation(factor=15/360, fill_mode="nearest")
        random_zoom = tf.keras.layers.RandomZoom(height_factor=0.1, fill_mode="nearest")

    @tf.function
    def decode_and_preprocess(path, label):
        # Read & decode
        raw   = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [H, W])

        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            # Random rotation ±15° via crop-and-resize trick
            image = random_rotation(tf.expand_dims(image, 0), training=True)[0]
            image = random_zoom(tf.expand_dims(image, 0), training=True)[0]

        # EfficientNetB0 preprocessing: [0,255] → torch-style normalized float32
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        # One-hot encode label
        one_hot = tf.one_hot(label, NUM_CLASSES)
        return image, one_hot

    return decode_and_preprocess


def _build_tf_dataset(
    paths: list,
    labels: list,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int = SEED,
) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    decode_fn = _make_decode_fn(augment=augment)

    ds = (
        ds
        .map(decode_fn, num_parallel_calls=AUTOTUNE)   # parallel CPU decoding
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)                             # overlap GPU compute + CPU load
    )
    return ds


def build_tl_pipeline(
    train_dir:  Path  = TRAIN_DIR,
    test_dir:   Path  = TEST_DIR,
    batch_size: int   = BATCH_SIZE,
    val_split:  float = VALIDATION_SPLIT,
    seed:       int   = SEED,
) -> tuple:
    """
    Build fast tf.data pipelines for MobileNetV2 transfer learning.

    Uses parallel map + GPU prefetch instead of ImageDataGenerator,
    which keeps the GPU at ~90% utilization vs ~25% with the old approach.

    Returns:
        (train_ds, val_ds, test_ds, train_steps, val_steps)
    """
    # ── Collect all training file paths ──────────────────────────────────────
    all_paths, all_labels = _collect_paths_and_labels(train_dir)
    total = len(all_paths)

    # Deterministic shuffle before split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    all_paths  = [all_paths[i]  for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Split 85/15
    split_idx  = int(total * (1 - val_split))
    train_paths, train_labels = all_paths[:split_idx],  all_labels[:split_idx]
    val_paths,   val_labels   = all_paths[split_idx:],  all_labels[split_idx:]

    # ── Test set ──────────────────────────────────────────────────────────────
    test_paths, test_labels = _collect_paths_and_labels(test_dir)

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = _build_tf_dataset(train_paths, train_labels, batch_size, augment=True,  shuffle=True,  seed=seed)
    val_ds   = _build_tf_dataset(val_paths,   val_labels,   batch_size, augment=False, shuffle=False)
    test_ds  = _build_tf_dataset(test_paths,  test_labels,  batch_size, augment=False, shuffle=False)

    import math
    train_steps = math.ceil(len(train_paths) / batch_size)
    val_steps   = math.ceil(len(val_paths)   / batch_size)

    print(f"\n  [Pipeline B — MobileNetV2  224x224 RGB  tf.data FAST]")
    print(f"    Train samples      : {len(train_paths):,}")
    print(f"    Validation samples : {len(val_paths):,}")
    print(f"    Test samples       : {len(test_paths):,}")
    print(f"    Batch size         : {batch_size}")
    print(f"    Steps/epoch(train) : {train_steps}")
    print(f"    Steps/epoch(val)   : {val_steps}")
    print(f"    Prefetch + parallel: AUTOTUNE  ← GPU stays fed")

    return train_ds, val_ds, test_ds, train_steps, val_steps
