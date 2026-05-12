"""
train_transfer.py
-----------------
Stage 5 — Train the Transfer Learning (MobileNetV2) model on FER2013.

Training strategy:
  Phase 1  : Freeze MobileNetV2 backbone, train only the new emotion head.
             Fast convergence, ~10 epochs.
  Phase 2  : Unfreeze last 30 layers of backbone, fine-tune at LR=1e-5.
             Specialises the features for face expressions, ~15 epochs.

Expected result: 68–73% test accuracy (vs 66.3% CNN-scratch baseline).

Outputs:
  models/transfer_learning_best.keras
  output/reports/transfer_training_log.csv
  output/reports/transfer_training_curves.png
  output/reports/transfer_summary.json
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

# ── Project root on sys.path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf

# ── Enable GPU memory growth (prevents OOM on 3050 Ti 4 GB) ───────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"\n  ✓ GPU detected: {[g.name for g in gpus]} — memory growth enabled.")
else:
    print("\n  ⚠ No GPU detected — running on CPU. Install/load the NVIDIA driver.")

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)
from tensorflow.keras.optimizers import Adam

from src.full_preprocessor import build_tl_pipeline, EMOTION_CLASSES
from src.models.transfer_learning import build_transfer_learning_model, unfreeze_top_layers

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("output/reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH  = MODELS_DIR / "transfer_learning_best.keras"
CSV_LOG_PATH     = REPORTS_DIR / "transfer_training_log.csv"
CURVES_SAVE_PATH = REPORTS_DIR / "transfer_training_curves.png"
WEIGHTS_JSON     = REPORTS_DIR / "class_weights.json"
SUMMARY_JSON     = REPORTS_DIR / "transfer_summary.json"

# ── Terminal colours ───────────────────────────────────────────────────────────
CYAN   = "\033[96m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def load_class_weights() -> dict | None:
    """Load class weights from the JSON produced by prepare_data.py."""
    if WEIGHTS_JSON.exists():
        with open(WEIGHTS_JSON) as f:
            cw = json.load(f)
        return {int(k): v for k, v in cw.items()}
    print(f"  {YELLOW}⚠ class_weights.json not found — training without class weights.{RESET}")
    return None


def plot_training_curves(
    history_p1: dict,
    history_p2: dict,
    save_path: Path,
) -> None:
    """Combine Phase 1 + Phase 2 histories into a single plot."""
    acc     = history_p1["accuracy"]     + history_p2["accuracy"]
    val_acc = history_p1["val_accuracy"] + history_p2["val_accuracy"]
    loss    = history_p1["loss"]         + history_p2["loss"]
    val_loss= history_p1["val_loss"]     + history_p2["val_loss"]

    epochs        = range(1, len(acc) + 1)
    phase1_epochs = len(history_p1["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")

    panels = [
        ("Loss",     loss,    val_loss, "Loss",     "#E74C3C", "#F39C12"),
        ("Accuracy", acc,     val_acc,  "Accuracy", "#2980B9", "#27AE60"),
    ]

    for ax, (label, train_d, val_d, ylabel, tc, vc) in zip(axes, panels):
        ax.set_facecolor("#16213E")
        ax.plot(epochs, train_d, color=tc, lw=2, label=f"Train {label}", marker="o", ms=3)
        ax.plot(epochs, val_d,   color=vc, lw=2, label=f"Val {label}",   marker="s", ms=3, ls="--")
        ax.axvline(phase1_epochs, color="#95A5A6", ls="--", lw=1.5, label="Phase 2 Start")

        if label == "Accuracy":
            best_ep = int(np.argmax(val_d)) + 1
            best_v  = max(val_d)
            ax.axvline(best_ep, color="white", ls=":", lw=1)
            ax.text(best_ep + 0.2, best_v - 0.03,
                    f"Best: {best_v:.4f}\nEp {best_ep}", color="white", fontsize=8)
        else:
            best_ep = int(np.argmin(val_d)) + 1
            ax.axvline(best_ep, color="white", ls=":", lw=1)

        ax.set_title(f"Training vs Validation {label}", color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="#B0B0C0", fontsize=10)
        ax.set_ylabel(ylabel,  color="#B0B0C0", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#0F3460")
        ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    fig.suptitle(
        "MobileNetV2 Transfer Learning — Training Curves (Phase 1 + Phase 2)",
        color="white", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved training curves → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on FER2013")
    parser.add_argument("--epochs1",    type=int,   default=15,   help="Phase 1 epochs (frozen base)")
    parser.add_argument("--epochs2",    type=int,   default=20,   help="Phase 2 epochs (fine-tuning)")
    parser.add_argument("--lr1",        type=float, default=1e-3, help="Phase 1 learning rate")
    parser.add_argument("--lr2",        type=float, default=1e-5, help="Phase 2 learning rate")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--unfreeze",   type=int,   default=30,   help="Layers to unfreeze in Phase 2")
    parser.add_argument("--no-weights", action="store_true",      help="Disable class weighting")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}")
    print("+============================================================+")
    print("|   Stage 5 — Transfer Learning (MobileNetV2)               |")
    print("|   FER2013  |  224x224 RGB  |  7 emotion classes           |")
    print(f"+============================================================+{RESET}\n")

    # ── 1. Data pipelines ──────────────────────────────────────────────────────
    print(f"{CYAN}[Step 1] Building Pipeline B (224x224 RGB — tf.data FAST)...{RESET}")
    train_ds, val_ds, test_ds = build_tl_pipeline(batch_size=args.batch_size)

    # Compute steps from dataset cardinalities (tf.data datasets don't have .samples)
    BATCH = args.batch_size
    import math
    # Approximate steps: total images / batch_size
    TRAIN_SAMPLES = int(25453 * 0.85)   # ~25,888 train * 0.85 split
    VAL_SAMPLES   = int(25453 * 0.15)
    train_steps = math.ceil(TRAIN_SAMPLES / BATCH)
    val_steps   = math.ceil(VAL_SAMPLES   / BATCH)

    class_weights = None if args.no_weights else load_class_weights()
    if class_weights:
        print(f"  ✓ Class weights loaded.")

    # ── 2. Build model ─────────────────────────────────────────────────────────
    print(f"\n{CYAN}[Step 2] Building MobileNetV2 + custom head...{RESET}")
    model, base_model = build_transfer_learning_model()

    model.compile(
        optimizer = Adam(learning_rate=args.lr1),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── 3. Callbacks ───────────────────────────────────────────────────────────
    def make_callbacks(csv_append: bool = False):
        return [
            ModelCheckpoint(
                filepath      = str(MODEL_SAVE_PATH),
                monitor       = "val_accuracy",
                save_best_only= True,
                verbose       = 1,
            ),
            EarlyStopping(
                monitor            = "val_accuracy",
                patience           = 8,
                restore_best_weights= True,
                verbose            = 1,
            ),
            ReduceLROnPlateau(
                monitor  = "val_loss",
                factor   = 0.5,
                patience = 4,
                min_lr   = 1e-8,
                verbose  = 1,
            ),
            CSVLogger(str(CSV_LOG_PATH), append=csv_append),
        ]

    # ── 4. Phase 1: Train head only (frozen backbone) ─────────────────────────
    print(f"\n{CYAN}[Step 3] Phase 1 — Feature Extraction ({args.epochs1} epochs, LR={args.lr1})...{RESET}")
    print(f"  Base model frozen. Only training the new classification head.\n")

    t0 = time.time()
    history_p1 = model.fit(
        train_ds,
        epochs              = args.epochs1,
        steps_per_epoch     = train_steps,
        validation_data     = val_ds,
        validation_steps    = val_steps,
        class_weight        = class_weights,
        callbacks           = make_callbacks(csv_append=False),
        verbose             = 1,
    )

    # ── 5. Phase 2: Fine-tune top layers ──────────────────────────────────────
    print(f"\n{CYAN}[Step 4] Phase 2 — Fine-Tuning ({args.epochs2} more epochs, LR={args.lr2})...{RESET}")
    unfreeze_top_layers(base_model, num_layers_to_unfreeze=args.unfreeze)

    model.compile(
        optimizer = Adam(learning_rate=args.lr2),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    p1_done = len(history_p1.history["loss"])
    history_p2 = model.fit(
        train_ds,
        epochs              = p1_done + args.epochs2,
        initial_epoch       = p1_done,
        steps_per_epoch     = train_steps,
        validation_data     = val_ds,
        validation_steps    = val_steps,
        class_weight        = class_weights,
        callbacks           = make_callbacks(csv_append=True),
        verbose             = 1,
    )

    train_time = time.time() - t0
    minutes = train_time / 60

    # ── 6. Save training curves ────────────────────────────────────────────────
    print(f"\n{CYAN}[Step 5] Saving training curves...{RESET}")
    plot_training_curves(history_p1.history, history_p2.history, CURVES_SAVE_PATH)

    # ── 7. Evaluate on held-out test set ──────────────────────────────────────
    print(f"\n{CYAN}[Step 6] Evaluating on held-out test set...{RESET}")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    # ── 8. Save summary JSON ───────────────────────────────────────────────────
    all_val_acc  = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
    best_val_acc = float(max(all_val_acc))
    best_ep      = int(np.argmax(all_val_acc)) + 1

    summary = {
        "model"            : "MobileNetV2_TransferLearning",
        "phase1_epochs"    : args.epochs1,
        "phase2_epochs"    : args.epochs2,
        "lr_phase1"        : args.lr1,
        "lr_phase2"        : args.lr2,
        "layers_unfrozen"  : args.unfreeze,
        "best_val_accuracy": round(best_val_acc, 6),
        "best_val_epoch"   : best_ep,
        "test_accuracy"    : round(float(test_acc), 6),
        "test_loss"        : round(float(test_loss), 6),
        "train_time_min"   : round(minutes, 1),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ───────────────────────────────────────────────────────────
    print(f"\n{GREEN}{BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Stage 5 Complete — Transfer Learning Done             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Best Val Accuracy : {best_val_acc:.4f}  (epoch {best_ep})" + " " * (24 - len(f"{best_val_acc:.4f}  (epoch {best_ep})")) + "║")
    print(f"║  Test Accuracy     : {test_acc:.4f}" + " " * 33 + "║")
    print(f"║  Training Time     : {minutes:.1f} min" + " " * (33 - len(f"{minutes:.1f} min")) + "║")
    print(f"╚══════════════════════════════════════════════════════════╝{RESET}")
    print(f"\n  Model saved → {MODEL_SAVE_PATH}")
    print(f"  Summary  → {SUMMARY_JSON}")
    print(f"  Curves   → {CURVES_SAVE_PATH}\n")


if __name__ == "__main__":
    main()
