"""
preprocessor.py
---------------
Full preprocessing pipeline for FER2013 images.

Steps applied to each image:
  1. Load as grayscale
  2. Resize to 48x48 (if not already)
  3. CLAHE histogram equalization (contrast enhancement)
  4. Gaussian blur denoising
  5. Normalize to [0, 1] float32
  6. Return both processed image and array
"""

import cv2
import numpy as np
from pathlib import Path


# ── Pipeline configuration ──────────────────────────────────────────────────
TARGET_SIZE    = (48, 48)     # Output image dimensions (H, W)
CLAHE_CLIP     = 2.0          # CLAHE clip limit (contrast enhancement strength)
CLAHE_TILE     = (8, 8)       # CLAHE tile grid size
GAUSSIAN_KERNEL = (3, 3)      # Gaussian blur kernel (light denoising)
GAUSSIAN_SIGMA  = 0           # Sigma=0 → auto from kernel size


def load_grayscale(filepath: str) -> np.ndarray | None:
    """Load an image as grayscale. Returns None if the file cannot be opened."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img


def resize_image(img: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize image to target size using INTER_AREA (best for downscaling)."""
    if img.shape[:2] == size:
        return img
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast without over-amplifying noise.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(img)


def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Apply a light Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0.0, 1.0] float32."""
    return img.astype(np.float32) / 255.0


def preprocess_image(filepath: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Full preprocessing pipeline for a single image.

    Args:
        filepath: Path to the source image.

    Returns:
        Tuple of (processed_uint8_img, normalized_float_array).
        Both are None if the image cannot be loaded.
    """
    img = load_grayscale(filepath)
    if img is None:
        return None, None

    img = resize_image(img)
    img = apply_clahe(img)
    img = apply_gaussian_blur(img)

    processed_display = img.copy()          # uint8 [0, 255] for saving/display
    normalized_array  = normalize(img)      # float32 [0.0, 1.0] for model input

    return processed_display, normalized_array


def preprocess_batch(
    batch_df,
    output_dir: str = "output/batch_samples",
    save_images: bool = True,
) -> tuple["np.ndarray", "np.ndarray", "list"]:
    """
    Preprocess all usable images in the batch.

    Args:
        batch_df:    DataFrame with quality metrics (from quality_checker).
        output_dir:  Directory to save processed images (organized by class).
        save_images: Whether to save each processed image to disk.

    Returns:
        (X, y, labels) where:
          X      – float32 array of shape (N, 48, 48) normalized images
          y      – int array of emotion labels
          labels – list of emotion names
    """
    import pandas as pd
    from tqdm import tqdm

    out_path = Path(output_dir)

    # Only process usable images
    usable_df = batch_df[batch_df["is_usable"]].reset_index(drop=True)
    print(f"\n  Preprocessing {len(usable_df)} usable images...")

    X_list, y_list, label_list = [], [], []

    for _, row in tqdm(usable_df.iterrows(), total=len(usable_df), desc="  Processing", unit="img"):
        processed, normalized = preprocess_image(row["filepath"])

        if processed is None:
            continue

        X_list.append(normalized)
        y_list.append(int(row["label"]))
        label_list.append(row["emotion_name"])

        # Save processed image organized by class
        if save_images:
            class_dir = out_path / row["emotion_name"]
            class_dir.mkdir(parents=True, exist_ok=True)
            save_path = class_dir / row["filename"]
            cv2.imwrite(str(save_path), processed)

    X = np.array(X_list, dtype=np.float32)    # (N, 48, 48)
    y = np.array(y_list,  dtype=np.int32)      # (N,)

    print(f"\n  Preprocessing complete:")
    print(f"    Output array shape : {X.shape}")
    print(f"    Pixel value range  : [{X.min():.4f}, {X.max():.4f}]")
    print(f"    Mean pixel value   : {X.mean():.4f}")
    print(f"    Std  pixel value   : {X.std():.4f}")

    if save_images:
        print(f"    Saved to           : {out_path.resolve()}")

    return X, y, label_list
