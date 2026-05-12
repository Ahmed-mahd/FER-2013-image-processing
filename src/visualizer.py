"""
visualizer.py
-------------
Generates all analysis plots and saves them to output/reports/.

Plots produced:
  1. class_distribution.png      - bar chart of images per emotion
  2. brightness_distribution.png - violin plot of brightness per class
  3. sample_grid_raw.png         - grid of raw images (before preprocessing)
  4. before_after_comparison.png - side-by-side raw vs preprocessed
  5. pixel_distributions.png     - pixel value histograms per class (post-processing)
  6. quality_summary.png         - usable vs flagged stacked bar
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


# ── Color palette (one per FER2013 emotion) ──────────────────────────────────
EMOTION_COLORS = {
    "angry":    "#E74C3C",
    "disgust":  "#8E44AD",
    "fear":     "#2C3E50",
    "happy":    "#F39C12",
    "neutral":  "#95A5A6",
    "sad":      "#2980B9",
    "surprise": "#27AE60",
}

DARK_BG    = "#0F0E17"
PANEL_BG   = "#1A1A2E"
PLOT_BG    = "#16213E"
SPINE_COL  = "#0F3460"
TEXT_WHITE = "white"
TEXT_DIM   = "#B0B0C0"


# ── Helper: apply a consistent dark style to any axes ────────────────────────
def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT_WHITE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COL)
    if title:
        ax.set_title(title, color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Class Distribution Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(batch_df: pd.DataFrame, output_dir: Path) -> str:
    counts = batch_df["emotion_name"].value_counts().sort_index()
    colors = [EMOTION_COLORS.get(e, "#7F8C8D") for e in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PANEL_BG)

    bars = ax.bar(counts.index, counts.values, color=colors,
                  edgecolor=SPINE_COL, linewidth=1.2, width=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom",
                color=TEXT_WHITE, fontsize=12, fontweight="bold")

    _style_ax(ax,
              title="Batch Class Distribution",
              xlabel="Emotion Class",
              ylabel="Number of Images")

    plt.tight_layout()
    path = output_dir / "class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Brightness Violin Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_brightness_distribution(batch_df: pd.DataFrame, output_dir: Path) -> str:
    emotions = sorted(batch_df["emotion_name"].unique())
    data     = [batch_df[batch_df["emotion_name"] == e]["mean_brightness"].dropna().values
                for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#7F8C8D") for e in emotions]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PANEL_BG)

    parts = ax.violinplot(data, positions=range(len(emotions)),
                          showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(TEXT_WHITE if key in ("cmeans", "cmedians") else "#444466")

    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, color=TEXT_WHITE, fontsize=11)
    ax.axhline(10,  color="#E74C3C", linestyle="--", linewidth=1, label="Too dark  (<10)")
    ax.axhline(245, color="#F39C12", linestyle="--", linewidth=1, label="Too bright (>245)")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_WHITE, fontsize=9)

    _style_ax(ax,
              title="Brightness Distribution per Emotion Class",
              xlabel="Emotion",
              ylabel="Mean Pixel Brightness (0-255)")

    plt.tight_layout()
    path = output_dir / "brightness_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Raw Sample Grid
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample_grid(batch_df: pd.DataFrame, output_dir: Path, n_per_class: int = 5) -> str:
    emotions  = sorted(batch_df["emotion_name"].unique())
    n_classes = len(emotions)

    fig = plt.figure(figsize=(n_per_class * 2.2, n_classes * 2.6))
    fig.patch.set_facecolor(DARK_BG)

    for row_idx, emotion in enumerate(emotions):
        cls_df  = batch_df[batch_df["emotion_name"] == emotion]
        samples = cls_df.sample(n=min(n_per_class, len(cls_df)), random_state=0)

        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            ax  = fig.add_subplot(n_classes, n_per_class,
                                  row_idx * n_per_class + col_idx + 1)
            img = cv2.imread(sample["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="red", transform=ax.transAxes)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(emotion,
                              color=EMOTION_COLORS.get(emotion, TEXT_WHITE),
                              fontsize=11, fontweight="bold",
                              rotation=0, labelpad=60, va="center")

    fig.suptitle("Raw Batch Samples (Before Preprocessing)",
                 color=TEXT_WHITE, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "sample_grid_raw.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Before / After Comparison (Raw vs Preprocessed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_before_after(batch_df: pd.DataFrame, output_dir: Path, n_samples: int = 7) -> str:
    """
    Show one representative image per emotion class:
      Left column  = raw image
      Right column = preprocessed image with CLAHE applied
    """
    emotions = sorted(batch_df["emotion_name"].unique())
    n_rows   = len(emotions)
    n_cols   = 2  # raw | preprocessed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, n_rows * 1.9))
    fig.patch.set_facecolor(DARK_BG)

    # Guarantee axes is always 2D even if n_rows == 1
    axes = np.atleast_2d(axes)

    # Column headers
    axes[0, 0].set_title("RAW", color="#F39C12", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].set_title("PREPROCESSED\n(CLAHE + Blur + Normalize)",
                          color="#27AE60", fontsize=11, fontweight="bold", pad=8)

    for row_idx, emotion in enumerate(emotions):
        cls_df = batch_df[(batch_df["emotion_name"] == emotion) & batch_df["is_usable"]]
        if cls_df.empty:
            cls_df = batch_df[batch_df["emotion_name"] == emotion]

        sample = cls_df.sample(n=1, random_state=row_idx).iloc[0]

        # ── Load raw ──
        raw = cv2.imread(sample["filepath"], cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raw = np.zeros((48, 48), dtype=np.uint8)

        # ── Produce preprocessed inline ──
        resized  = cv2.resize(raw, (48, 48), interpolation=cv2.INTER_AREA)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)
        normed   = blurred.astype(np.float32) / 255.0   # [0,1] for display

        # ── Left: raw ──
        ax_raw = axes[row_idx, 0]
        ax_raw.imshow(raw, cmap="gray", vmin=0, vmax=255)
        ax_raw.axis("off")
        ax_raw.set_ylabel(emotion,
                          color=EMOTION_COLORS.get(emotion, TEXT_WHITE),
                          fontsize=10, fontweight="bold",
                          rotation=0, labelpad=55, va="center")

        # ── Right: preprocessed ──
        ax_pre = axes[row_idx, 1]
        ax_pre.imshow(normed, cmap="gray", vmin=0, vmax=1)
        ax_pre.axis("off")

        # Side-by-side pixel value text
        ax_raw.set_xlabel(f"mean={raw.mean():.0f}", color=TEXT_DIM, fontsize=7)
        ax_pre.set_xlabel(f"mean={normed.mean():.2f}  std={normed.std():.2f}",
                          color=TEXT_DIM, fontsize=7)

    fig.suptitle("Before vs After Preprocessing — One Sample per Emotion",
                 color=TEXT_WHITE, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "before_after_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Pixel Value Histograms per Class (Post-Preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pixel_histograms(X: np.ndarray, label_list: list, output_dir: Path) -> str:
    """
    7 subplots (one per emotion) showing the pixel intensity distribution
    after preprocessing (normalized [0, 1]).
    """
    unique_emotions = sorted(set(label_list))
    label_arr = np.array(label_list)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor(PANEL_BG)
    axes_flat = axes.flatten()

    for i, emotion in enumerate(unique_emotions):
        ax     = axes_flat[i]
        mask   = label_arr == emotion
        pixels = X[mask].flatten()
        color  = EMOTION_COLORS.get(emotion, "#7F8C8D")

        ax.hist(pixels, bins=64, color=color, alpha=0.85, edgecolor="none", density=True)

        # Overlay mean line
        mean_v = pixels.mean()
        std_v  = pixels.std()
        ax.axvline(mean_v, color="white", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_v:.3f}\nstd ={std_v:.3f}")
        ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_WHITE, fontsize=8, framealpha=0.7)

        _style_ax(ax,
                  title=emotion.capitalize(),
                  xlabel="Pixel Intensity (0-1)",
                  ylabel="Density")
        ax.set_xlim(0, 1)

    # Hide any unused subplot (8th cell)
    for j in range(len(unique_emotions), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Pixel Intensity Histograms After Preprocessing (Normalized [0,1])",
                 color=TEXT_WHITE, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "pixel_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Stacked Quality Summary
# ─────────────────────────────────────────────────────────────────────────────
def plot_quality_summary(batch_df: pd.DataFrame, output_dir: Path) -> str:
    emotions = sorted(batch_df["emotion_name"].unique())
    usable   = [len(batch_df[(batch_df["emotion_name"] == e) &  batch_df["is_usable"]]) for e in emotions]
    flagged  = [len(batch_df[(batch_df["emotion_name"] == e) & ~batch_df["is_usable"]]) for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PANEL_BG)

    x = np.arange(len(emotions))
    ax.bar(x, usable,  label="Usable",  color="#27AE60", edgecolor=SPINE_COL, linewidth=1)
    ax.bar(x, flagged, label="Flagged", color="#E74C3C", edgecolor=SPINE_COL, linewidth=1,
           bottom=usable)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, color=TEXT_WHITE, fontsize=11)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_WHITE)

    _style_ax(ax,
              title="Image Usability by Emotion Class",
              ylabel="Image Count")

    plt.tight_layout()
    path = output_dir / "quality_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Master function — run everything
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_plots(
    batch_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    label_list: list,
    output_dir: str = "output/reports",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n  Generating visualizations...")
    paths = {}

    paths["class_distribution"]     = plot_class_distribution(batch_df, out)
    paths["brightness_distribution"] = plot_brightness_distribution(batch_df, out)
    paths["sample_grid_raw"]         = plot_sample_grid(batch_df, out)
    paths["before_after_comparison"] = plot_before_after(batch_df, out)
    paths["pixel_histograms"]        = plot_pixel_histograms(X, label_list, out)
    paths["quality_summary"]         = plot_quality_summary(batch_df, out)

    for name, path in paths.items():
        print(f"    OK  {name}: {Path(path).name}")

    return paths
