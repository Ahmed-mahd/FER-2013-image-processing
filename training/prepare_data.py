"""
prepare_data.py
---------------
Stage 3 Runner — Prepares the full FER2013 dataset for model training.

What this script does:
  1. Augments the Disgust (minority) class (x4 copies) to reduce imbalance
  2. Computes class weights for use in model.fit()
  3. Builds and validates both data pipelines:
       Pipeline A — CNN scratch  (48x48 grayscale, normalized [0,1])
       Pipeline B — EfficientNetB0  (224x224 RGB, normalized [-1,1])
  4. Verifies a sample batch from each pipeline
  5. Saves pipeline stats and augmentation summary to output/reports/

Run:
    python prepare_data.py
    python prepare_data.py --no-augment   (skip minority augmentation)
    python prepare_data.py --clean        (remove augmented images then exit)
"""

import sys
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.full_preprocessor import (
    TRAIN_DIR, TEST_DIR, EMOTION_CLASSES, BATCH_SIZE,
    augment_minority_class, remove_augmented_images,
    compute_class_weights, build_scratch_pipeline, build_tl_pipeline,
    MINORITY_CLASS, MINORITY_MULTIPLIER,
)

OUT_DIR = Path("output/reports/stage3_preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def step(n, title):
    print(f"\n{CYAN}{BOLD}[Step {n}] {title}{RESET}")
    print("-" * 60)


def verify_batch(gen, name: str, out_dir: Path) -> dict:
    """Pull one batch and verify shapes, value ranges, class distribution."""
    X, y = next(gen)
    stats = {
        "name":        name,
        "batch_shape": str(X.shape),
        "dtype":       str(X.dtype),
        "pixel_min":   float(X.min()),
        "pixel_max":   float(X.max()),
        "pixel_mean":  float(X.mean()),
        "pixel_std":   float(X.std()),
        "label_shape": str(y.shape),
        "classes_in_batch": int((y.sum(axis=0) > 0).sum()),
    }
    print(f"\n  Batch verification  [{name}]")
    print(f"    X shape     : {stats['batch_shape']}")
    print(f"    dtype       : {stats['dtype']}")
    print(f"    pixel range : [{stats['pixel_min']:.4f}, {stats['pixel_max']:.4f}]")
    print(f"    pixel mean  : {stats['pixel_mean']:.4f}")
    print(f"    pixel std   : {stats['pixel_std']:.4f}")
    print(f"    classes hit : {stats['classes_in_batch']} / {len(EMOTION_CLASSES)}")
    return stats


def plot_sample_batch(gen, name: str, out_dir: Path, n: int = 8):
    """Save a grid of augmented training samples from the generator."""
    X, y = next(gen)
    is_gray = (X.ndim == 4 and X.shape[-1] == 1)

    cols = min(n, len(X))
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.2, 2.5))
    fig.patch.set_facecolor("#0F0E17")

    for i, ax in enumerate(axes):
        img = X[i, :, :, 0] if is_gray else X[i]
        # Un-normalise for display
        if img.min() < 0:                    # EfficientNetB0 preprocess_input [-1,1] -> rescale to [0,1]
            img = (img + 1.0) / 2.0
        ax.imshow(img, cmap="gray" if is_gray else None, vmin=0, vmax=1)
        label_idx = int(np.argmax(y[i]))
        ax.set_title(EMOTION_CLASSES[label_idx], color="white", fontsize=8)
        ax.axis("off")

    title = f"Augmented Training Samples — {name}"
    fig.suptitle(title, color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = out_dir / f"augmented_samples_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Sample grid saved -> {fname}")
    return str(fname)


def plot_class_balance_comparison(before: dict, after: dict, out_dir: Path):
    """Bar chart comparing class counts before and after Disgust augmentation."""
    emotions = EMOTION_CLASSES
    b_vals   = [before.get(e, 0) for e in emotions]
    a_vals   = [after.get(e, 0)  for e in emotions]

    x     = np.arange(len(emotions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    ax.bar(x - width/2, b_vals, width, label="Before augmentation",
           color="#2980B9", edgecolor="#0F3460", linewidth=0.8)
    ax.bar(x + width/2, a_vals, width, label="After Disgust augmentation",
           color="#27AE60", edgecolor="#0F3460", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, color="white", fontsize=10)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#0F3460")
    ax.set_title("Class Distribution: Before vs After Disgust Augmentation",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images", color="#B0B0C0")
    ax.legend(facecolor="#1A1A2E", labelcolor="white")

    plt.tight_layout()
    out = out_dir / "class_balance_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Balance chart saved -> {out}")


def count_class_images(train_dir: Path) -> dict:
    counts = {}
    for emotion in EMOTION_CLASSES:
        d = train_dir / emotion
        if d.exists():
            counts[emotion] = len([f for f in d.iterdir()
                                   if f.suffix.lower() in {".jpg",".jpeg",".png"}])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip minority class augmentation")
    parser.add_argument("--clean", action="store_true",
                        help="Remove augmented images and exit")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}")
    print("+============================================================+")
    print("|       Stage 3 — Full Dataset Preprocessing                |")
    print("|       FER2013  |  Dual Pipeline Setup                     |")
    print(f"+============================================================+{RESET}\n")

    t0 = time.time()

    # ── Clean mode ────────────────────────────────────────────────────────────
    if args.clean:
        step("C", "Removing augmented images")
        removed = remove_augmented_images()
        print(f"  Removed {removed} augmented images from {MINORITY_CLASS}/")
        return

    # ── Step 1: Count before augmentation ────────────────────────────────────
    step(1, "Counting class distribution (before augmentation)")
    before_counts = count_class_images(TRAIN_DIR)
    print(f"\n  {'Class':<10} {'Count':>8}")
    print(f"  {'-'*10} {'-'*8}")
    for e in EMOTION_CLASSES:
        print(f"  {e:<10} {before_counts.get(e,0):>8,}")
    total_before = sum(before_counts.values())
    print(f"\n  Total: {total_before:,}")

    # ── Step 2: Augment minority class ────────────────────────────────────────
    if not args.no_augment:
        step(2, f"Augmenting minority class: '{MINORITY_CLASS}' (x{MINORITY_MULTIPLIER})")
        existing_aug = len([f for f in (TRAIN_DIR / MINORITY_CLASS).iterdir()
                            if f.stem.startswith("aug_")])
        if existing_aug > 0:
            print(f"  Found {existing_aug} existing augmented images — skipping re-generation.")
            print(f"  (Run with --clean first to regenerate)")
        else:
            written = augment_minority_class()
            print(f"  Written: {written} new augmented images")

        after_counts = count_class_images(TRAIN_DIR)
        print(f"\n  {MINORITY_CLASS} count: {before_counts[MINORITY_CLASS]:,} -> {after_counts[MINORITY_CLASS]:,}")
        plot_class_balance_comparison(before_counts, after_counts, OUT_DIR)
    else:
        after_counts = before_counts
        print("  [Skipped — --no-augment flag set]")

    # ── Step 3: Class weights ──────────────────────────────────────────────────
    step(3, "Computing class weights")
    class_weights = compute_class_weights(TRAIN_DIR)

    weights_path = OUT_DIR / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)
    print(f"\n  Class weights saved -> {weights_path}")

    # ── Step 4: Build Pipeline A (CNN scratch) ────────────────────────────────
    step(4, "Building Pipeline A — CNN Scratch (48x48 grayscale)")
    train_a, val_a, test_a = build_scratch_pipeline(
        batch_size=args.batch_size
    )

    # Verify and visualise
    stats_a   = verify_batch(train_a, "Pipeline_A_train", OUT_DIR)
    plot_sample_batch(train_a, "CNN Scratch", OUT_DIR)

    # ── Step 5: Build Pipeline B (Transfer Learning) ──────────────────────────
    step(5, "Building Pipeline B — EfficientNetB0 (224x224 RGB)")
    train_b, val_b, test_b = build_tl_pipeline(
        batch_size=args.batch_size
    )

    stats_b = verify_batch(train_b, "Pipeline_B_train", OUT_DIR)
    plot_sample_batch(train_b, "EfficientNetB0", OUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{GREEN}{BOLD}{'='*60}")
    print("  Stage 3 Complete — Pipelines Ready for Training")
    print(f"{'='*60}{RESET}")
    print(f"  Total prep time    : {elapsed:.1f}s")
    print(f"\n  Pipeline A (CNN scratch 48x48 grayscale):")
    print(f"    Train  {train_a.samples:>6,} | Val {val_a.samples:>5,} | Test {test_a.samples:>5,}")
    print(f"    Pixel range: [{stats_a['pixel_min']:.2f}, {stats_a['pixel_max']:.2f}]")
    print(f"\n  Pipeline B (EfficientNetB0 224x224 RGB):")
    print(f"    Train  {train_b.samples:>6,} | Val {val_b.samples:>5,} | Test {test_b.samples:>5,}")
    print(f"    Pixel range: [{stats_b['pixel_min']:.2f}, {stats_b['pixel_max']:.2f}]")
    print(f"\n  Class weights saved to: {weights_path}")
    print(f"  Plots saved to        : {OUT_DIR.resolve()}")
    print(f"\n{YELLOW}  Next: python train_scratch.py{RESET}\n")

    return train_a, val_a, test_a, train_b, val_b, test_b, class_weights


if __name__ == "__main__":
    main()
