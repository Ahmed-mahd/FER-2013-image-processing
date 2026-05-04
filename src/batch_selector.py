"""
batch_selector.py
-----------------
Randomly selects a stratified batch of images from the FER2013 train folder.
Ensures equal representation across all 7 emotion classes.
"""

import os
import random
import pandas as pd
from pathlib import Path


# FER2013 emotion label mapping
EMOTION_LABELS = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "neutral":  4,
    "sad":      5,
    "surprise": 6,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_dataset(data_dir: str) -> pd.DataFrame:
    """
    Walk the train directory and build a DataFrame of all images.

    Args:
        data_dir: Path to the train/ folder.

    Returns:
        DataFrame with columns: [filepath, emotion_name, label, split]
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please download the FER2013 train folder and place it at data/train/"
        )

    records = []
    found_classes = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        emotion_name = class_dir.name.lower()
        label = EMOTION_LABELS.get(emotion_name, -1)
        found_classes.append(emotion_name)

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    "filepath":     str(img_file),
                    "emotion_name": emotion_name,
                    "label":        label,
                    "filename":     img_file.name,
                })

    if not records:
        raise ValueError(
            f"No images found in {data_dir}. "
            "Make sure the folder contains subdirectories like angry/, happy/, etc."
        )

    print(f"  Dataset discovered: {len(records):,} images across {len(found_classes)} classes")
    for cls in found_classes:
        count = sum(1 for r in records if r["emotion_name"] == cls)
        print(f"    {cls:10s}: {count:,} images")

    return pd.DataFrame(records)


def select_random_batch(
    df: pd.DataFrame,
    batch_size: int = 210,
    seed: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """
    Select a random batch of images from the dataset.

    Args:
        df:           Full dataset DataFrame from discover_dataset().
        batch_size:   Total number of images to sample (default 210 = 30 per class).
        seed:         Random seed for reproducibility.
        stratified:   If True, sample equally from each class (recommended).

    Returns:
        DataFrame of the selected batch.
    """
    random.seed(seed)

    if stratified:
        classes = df["emotion_name"].unique()
        n_classes = len(classes)
        per_class = batch_size // n_classes
        remainder = batch_size % n_classes

        sampled_parts = []
        for i, cls in enumerate(sorted(classes)):
            cls_df = df[df["emotion_name"] == cls]
            # Give one extra sample to first `remainder` classes if batch_size not divisible
            n = per_class + (1 if i < remainder else 0)
            n = min(n, len(cls_df))  # Don't exceed available images
            sampled_parts.append(cls_df.sample(n=n, random_state=seed))

        batch_df = pd.concat(sampled_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        batch_df = df.sample(n=min(batch_size, len(df)), random_state=seed).reset_index(drop=True)

    print(f"\n  Batch selected: {len(batch_df)} images")
    for cls in sorted(batch_df["emotion_name"].unique()):
        count = len(batch_df[batch_df["emotion_name"] == cls])
        print(f"    {cls:10s}: {count} images")

    return batch_df
