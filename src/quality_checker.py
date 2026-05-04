"""
quality_checker.py
------------------
Validates each image in the batch to ensure it is usable for training.
Flags images that are corrupt, blank, too dark, too bright, or wrong size.
"""

import cv2
import numpy as np
from pathlib import Path


# Thresholds for quality checks
MIN_MEAN_BRIGHTNESS = 10.0    # Below this → image is too dark / nearly black
MAX_MEAN_BRIGHTNESS = 245.0   # Above this → image is too bright / nearly white
MIN_STD_DEV        = 5.0      # Below this → image has no contrast (blank/uniform)
EXPECTED_SIZE      = (48, 48) # FER2013 standard image size


def check_image_quality(filepath: str) -> dict:
    """
    Run quality checks on a single image file.

    Args:
        filepath: Absolute or relative path to the image.

    Returns:
        Dictionary with quality metrics and a list of issues found.
        Keys: filepath, loadable, width, height, mean_brightness, std_dev,
              is_grayscale, issues, is_usable
    """
    result = {
        "filepath":        filepath,
        "loadable":        False,
        "width":           None,
        "height":          None,
        "mean_brightness": None,
        "std_dev":         None,
        "is_grayscale":    None,
        "issues":          [],
        "is_usable":       False,
    }

    # --- 1. Load image ---
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        result["issues"].append("CORRUPT: Cannot be loaded")
        return result

    result["loadable"] = True
    result["height"]   = img.shape[0]
    result["width"]    = img.shape[1]

    # --- 2. Convert to grayscale for analysis ---
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result["is_grayscale"] = False
    elif len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        result["is_grayscale"] = False
    else:
        gray = img
        result["is_grayscale"] = True

    # --- 3. Pixel statistics ---
    mean_brightness = float(np.mean(gray))
    std_dev         = float(np.std(gray))
    result["mean_brightness"] = round(mean_brightness, 2)
    result["std_dev"]         = round(std_dev, 2)

    # --- 4. Quality checks ---
    if mean_brightness < MIN_MEAN_BRIGHTNESS:
        result["issues"].append(f"TOO_DARK: mean={mean_brightness:.1f}")

    if mean_brightness > MAX_MEAN_BRIGHTNESS:
        result["issues"].append(f"TOO_BRIGHT: mean={mean_brightness:.1f}")

    if std_dev < MIN_STD_DEV:
        result["issues"].append(f"LOW_CONTRAST: std={std_dev:.1f}")

    if (result["height"], result["width"]) != EXPECTED_SIZE:
        result["issues"].append(
            f"WRONG_SIZE: {result['width']}x{result['height']} (expected {EXPECTED_SIZE[1]}x{EXPECTED_SIZE[0]})"
        )

    result["is_usable"] = len(result["issues"]) == 0
    return result


def run_quality_checks(batch_df) -> "pd.DataFrame":
    """
    Run quality checks on all images in the batch DataFrame.

    Args:
        batch_df: DataFrame from batch_selector with a 'filepath' column.

    Returns:
        DataFrame with quality metrics merged into the batch.
    """
    import pandas as pd
    from tqdm import tqdm

    print("\n  Running quality checks...")
    results = []
    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="  Checking", unit="img"):
        qc = check_image_quality(row["filepath"])
        results.append(qc)

    qc_df = pd.DataFrame(results)

    # Merge quality info back into batch
    merged = batch_df.copy().reset_index(drop=True)
    merged["loadable"]        = qc_df["loadable"]
    merged["width"]           = qc_df["width"]
    merged["height"]          = qc_df["height"]
    merged["mean_brightness"] = qc_df["mean_brightness"]
    merged["std_dev"]         = qc_df["std_dev"]
    merged["is_grayscale"]    = qc_df["is_grayscale"]
    merged["issues"]          = qc_df["issues"].apply(lambda x: "; ".join(x) if x else "OK")
    merged["is_usable"]       = qc_df["is_usable"]

    # Summary
    total    = len(merged)
    usable   = merged["is_usable"].sum()
    unusable = total - usable

    print(f"\n  Quality Check Summary:")
    print(f"    Total images  : {total}")
    print(f"    Usable        : {usable}  ({usable/total*100:.1f}%)")
    print(f"    Flagged/Unusable: {unusable}  ({unusable/total*100:.1f}%)")

    if unusable > 0:
        print(f"\n  Flagged images:")
        flagged = merged[~merged["is_usable"]]
        for _, row in flagged.iterrows():
            print(f"    [{row['emotion_name']}] {row['filename']}: {row['issues']}")

    return merged
