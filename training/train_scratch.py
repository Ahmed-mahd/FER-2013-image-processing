"""
train_scratch.py
----------------
Stage 4 — Train the CNN from Scratch model on FER2013.

Training strategy:
  - Optimizer    : Adam (lr=0.001, decays via ReduceLROnPlateau)
  - Loss         : categorical_crossentropy
  - Class weights: loaded from output/reports/class_weights.json
  - Epochs       : up to 60 (EarlyStopping patience=12 will stop early)
  - Batch size   : 32
  - Callbacks    : ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

Outputs saved to:
  models/cnn_scratch_best.keras      -- best weights (val_accuracy)
  output/reports/scratch_training_log.csv
  output/reports/scratch_training_curves.png

Run:
    python train_scratch.py
    python train_scratch.py --epochs 30 --lr 0.0005
"""

import sys
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.full_preprocessor import build_scratch_pipeline, EMOTION_CLASSES
from src.models.cnn_scratch import build_cnn_scratch, compile_model

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("output/reports/stage4_cnn_scratch")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH  = MODELS_DIR / "cnn_scratch_best.keras"
CSV_LOG_PATH     = REPORTS_DIR / "scratch_training_log.csv"
CURVES_SAVE_PATH = REPORTS_DIR / "scratch_training_curves.png"
WEIGHTS_JSON     = Path("output/reports/stage3_preprocessing/class_weights.json")

# ── Colors ────────────────────────────────────────────────────────────────────
CYAN   = "\033[96m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RESET = "\033[0m"


def load_class_weights() -> dict:
    if WEIGHTS_JSON.exists():
        with open(WEIGHTS_JSON) as f:
            cw = json.load(f)
        # JSON keys are strings — convert to int
        return {int(k): v for k, v in cw.items()}
    print(f"  [Warning] class_weights.json not found — training without class weights.")
    return None


def plot_training_curves(history, save_path: Path):
    """Plot loss and accuracy curves (train vs val) and save to disk."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")

    panel_cfg = [
        ("loss",     "Training Loss vs Validation Loss",     "#E74C3C", "#F39C12"),
        ("accuracy", "Training Accuracy vs Validation Accuracy", "#2980B9", "#27AE60"),
    ]

    for ax, (metric, title, train_col, val_col) in zip(axes, panel_cfg):
        ax.set_facecolor("#16213E")
        ax.plot(epochs, history[metric],          color=train_col, linewidth=2,
                label=f"Train {metric.capitalize()}", marker="o", markersize=3)
        ax.plot(epochs, history[f"val_{metric}"], color=val_col,   linewidth=2,
                label=f"Val {metric.capitalize()}",   marker="s", markersize=3,
                linestyle="--")

        # Mark best val epoch
        if metric == "accuracy":
            best_ep = int(np.argmax(history["val_accuracy"])) + 1
            best_v  = max(history["val_accuracy"])
            ax.axvline(best_ep, color="white", linestyle=":", linewidth=1)
            ax.text(best_ep + 0.2, best_v - 0.03,
                    f"Best: {best_v:.4f}\nEp {best_ep}",
                    color="white", fontsize=8)
        else:
            best_ep = int(np.argmin(history["val_loss"])) + 1
            best_v  = min(history["val_loss"])
            ax.axvline(best_ep, color="white", linestyle=":", linewidth=1)

        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="#B0B0C0", fontsize=10)
        ax.set_ylabel(metric.capitalize(), color="#B0B0C0", fontsize=10)
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#0F3460")
        ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    fig.suptitle("CNN from Scratch — Training Curves  (FER2013)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Training curves saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--no-weights", action="store_true",
                        help="Disable class weighting")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}")
    print("+============================================================+")
    print("|   Stage 4 — CNN from Scratch Training                     |")
    print("|   FER2013  |  48x48 grayscale  |  7 classes               |")
    print(f"+============================================================+{RESET}\n")

    # ── 1. Data pipelines ────────────────────────────────────────────────────
    print(f"{CYAN}[Step 1] Building data pipeline...{RESET}")
    train_gen, val_gen, test_gen = build_scratch_pipeline(
        batch_size=args.batch_size
    )

    # ── 2. Class weights ──────────────────────────────────────────────────────
    class_weights = None if args.no_weights else load_class_weights()
    if class_weights:
        print(f"\n  Using class weights: {class_weights}")

    # ── 3. Build model ────────────────────────────────────────────────────────
    print(f"\n{CYAN}[Step 2] Building CNN model...{RESET}")
    model = build_cnn_scratch()
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    total_params     = model.count_params()
    trainable_params = sum(w.numpy().size for w in model.trainable_weights)
    print(f"\n  Total parameters      : {total_params:,}")
    print(f"  Trainable parameters  : {trainable_params:,}")

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath        = str(MODEL_SAVE_PATH),
            monitor         = "val_accuracy",
            save_best_only  = True,
            save_weights_only = False,
            verbose         = 1,
        ),
        EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 12,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 6,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        CSVLogger(str(CSV_LOG_PATH), append=False),
    ]

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n{CYAN}[Step 3] Training...{RESET}")
    print(f"  Epochs     : {args.epochs}  (EarlyStopping patience=12)")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR initial : {args.lr}")
    print(f"  Train steps: {train_gen.samples // args.batch_size}/epoch\n")

    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs           = args.epochs,
        validation_data  = val_gen,
        class_weight     = class_weights,
        callbacks        = callbacks,
        verbose          = 1,
    )
    train_time = time.time() - t0

    # ── 6. Plot curves ────────────────────────────────────────────────────────
    print(f"\n{CYAN}[Step 4] Saving training curves...{RESET}")
    plot_training_curves(history.history, CURVES_SAVE_PATH)

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    print(f"\n{CYAN}[Step 5] Evaluating on held-out test set...{RESET}")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    best_val_acc = max(history.history["val_accuracy"])
    best_ep      = int(np.argmax(history.history["val_accuracy"])) + 1
    epochs_run   = len(history.history["accuracy"])

    # ── 8. Save summary JSON ──────────────────────────────────────────────────
    summary = {
        "model":              "CNN_Scratch",
        "input_shape":        [48, 48, 1],
        "total_params":       total_params,
        "trainable_params":   trainable_params,
        "epochs_run":         epochs_run,
        "best_val_acc_epoch": best_ep,
        "best_val_accuracy":  round(best_val_acc, 6),
        "final_train_acc":    round(history.history["accuracy"][-1], 6),
        "test_accuracy":      round(test_acc, 6),
        "test_loss":          round(test_loss, 6),
        "train_time_seconds": round(train_time, 1),
        "batch_size":         args.batch_size,
        "initial_lr":         args.lr,
    }

    summary_path = REPORTS_DIR / "scratch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{GREEN}{BOLD}{'='*60}")
    print("  Stage 4 Complete — CNN Scratch Training Done")
    print(f"{'='*60}{RESET}")
    print(f"  Epochs run         : {epochs_run} / {args.epochs}")
    print(f"  Best val accuracy  : {best_val_acc:.4f}  (epoch {best_ep})")
    print(f"  Final train acc    : {history.history['accuracy'][-1]:.4f}")
    print(f"  Test accuracy      : {test_acc:.4f}")
    print(f"  Test loss          : {test_loss:.4f}")
    print(f"  Total params       : {total_params:,}")
    print(f"  Training time      : {train_time/60:.1f} min")
    print(f"\n  Saved:")
    print(f"    Model  -> {MODEL_SAVE_PATH}")
    print(f"    Log    -> {CSV_LOG_PATH}")
    print(f"    Curves -> {CURVES_SAVE_PATH}")
    print(f"    JSON   -> {summary_path}")
    print(f"\n{YELLOW}  Next: python train_transfer.py{RESET}\n")


if __name__ == "__main__":
    main()
