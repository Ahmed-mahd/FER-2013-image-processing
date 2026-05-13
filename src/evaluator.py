"""
evaluator.py
------------
Stage 6 — Evaluate and compare CNN Scratch vs EfficientNetB0 Transfer Learning
on the held-out FER2013 test set (7,178 images).

Generates (all saved to output/reports/stage6_evaluation/):
  - Confusion matrix PNGs for both models
  - Per-class classification report
  - ROC / AUC curves (one-vs-rest per emotion)
  - Side-by-side model comparison table (terminal + JSON)
  - evaluation_results.json

Run:
    python src/evaluator.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.full_preprocessor import (
    build_scratch_pipeline,
    build_tl_pipeline,
    EMOTION_CLASSES,
)

# ── Output directory ──────────────────────────────────────────────────────────
EVAL_DIR = Path("output/reports/stage6_evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = len(EMOTION_CLASSES)

# ── Terminal colours ──────────────────────────────────────────────────────────
CYAN  = "\033[96m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_generator(model, data_gen):
    """Pipeline A — ImageDataGenerator (CNN Scratch)."""
    data_gen.reset()
    steps = data_gen.samples // data_gen.batch_size + 1
    probs  = model.predict(data_gen, steps=steps, verbose=0)
    y_pred = np.argmax(probs, axis=1)[:data_gen.samples]
    y_true = data_gen.classes
    y_prob = probs[:data_gen.samples]
    return y_true, y_pred, y_prob


def predict_from_tf_dataset(model, dataset):
    """Pipeline B — tf.data (EfficientNetB0)."""
    all_probs, all_labels = [], []
    for images, labels in dataset:
        probs = model.predict_on_batch(images)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob   = np.concatenate(all_probs,  axis=0)
    y_onehot = np.concatenate(all_labels, axis=0)
    y_pred   = np.argmax(y_prob,   axis=1)
    y_true   = np.argmax(y_onehot, axis=1)
    return y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True label",      color="#B0B0C0", fontsize=11)
    ax.set_xlabel("Predicted label", color="#B0B0C0", fontsize=11)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_roc_curves(y_true, y_prob, classes, model_name, save_path):
    """One-vs-rest ROC curves for all emotion classes."""
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))

    COLORS = ["#E74C3C","#8E44AD","#2980B9","#F39C12","#95A5A6","#27AE60","#E67E22"]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, (cls, color) in enumerate(zip(classes, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#B0B0C0", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="#B0B0C0", fontsize=11)
    ax.set_title(f"ROC Curves — {model_name}", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#0F3460")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=9,
              loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ROC curves      → {save_path}")


def plot_comparison_bar(results, save_path):
    """Side-by-side bar chart comparing both models."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.35
    colors  = ["#2980B9", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", color="#B0B0C0", fontsize=11)
    ax.set_title("Model Comparison — CNN Scratch vs EfficientNetB0",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#0F3460")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Comparison bar  → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_prob, model_name, slug):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = float(np.mean(y_true == y_pred))

    print(f"\n  {CYAN}{BOLD}{model_name}{RESET}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                 target_names=EMOTION_CLASSES, zero_division=0))

    plot_confusion_matrix(
        y_true, y_pred, EMOTION_CLASSES,
        title=f"Confusion Matrix — {model_name}",
        save_path=EVAL_DIR / f"{slug}_confusion_matrix.png",
    )
    plot_roc_curves(
        y_true, y_prob, EMOTION_CLASSES,
        model_name=model_name,
        save_path=EVAL_DIR / f"{slug}_roc_curves.png",
    )

    return {
        "accuracy":  round(acc,            6),
        "precision": round(float(precision), 6),
        "recall":    round(float(recall),    6),
        "f1_score":  round(float(f1),        6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print("  Stage 6 — Model Evaluation & Comparison")
    print(f"{'='*60}{RESET}\n")

    results = {}

    # ── CNN Scratch ───────────────────────────────────────────────────────────
    cnn_path = Path("models/cnn_scratch_best.keras")
    if cnn_path.exists():
        print(f"{CYAN}[1/2] CNN Scratch — Pipeline A (ImageDataGenerator){RESET}")
        _, _, test_scratch = build_scratch_pipeline(batch_size=32)
        cnn_model = tf.keras.models.load_model(cnn_path)
        y_true, y_pred, y_prob = predict_from_generator(cnn_model, test_scratch)
        results["CNN_Scratch"] = evaluate_model(y_true, y_pred, y_prob,
                                                 "CNN Scratch", "cnn_scratch")
        del cnn_model
    else:
        print(f"  {YELLOW}[SKIP] {cnn_path} not found.{RESET}")

    # ── EfficientNetB0 Transfer Learning ─────────────────────────────────────
    tl_path = Path("models/transfer_learning_best.keras")
    if tl_path.exists():
        print(f"\n{CYAN}[2/2] EfficientNetB0 — Pipeline B (tf.data){RESET}")
        _, _, test_tl, _, _ = build_tl_pipeline(batch_size=32)
        tl_model = tf.keras.models.load_model(tl_path)
        y_true, y_pred, y_prob = predict_from_tf_dataset(tl_model, test_tl)
        results["EfficientNetB0"] = evaluate_model(y_true, y_pred, y_prob,
                                                    "EfficientNetB0", "efficientnetb0")
        del tl_model
    else:
        print(f"  {YELLOW}[SKIP] {tl_path} not found.{RESET}")

    if not results:
        print("No models found. Run train_scratch.py and train_transfer.py first.")
        return

    # ── Comparison bar chart ──────────────────────────────────────────────────
    if len(results) == 2:
        plot_comparison_bar(results, EVAL_DIR / "model_comparison.png")

    # ── Terminal comparison table ─────────────────────────────────────────────
    print(f"\n{GREEN}{BOLD}{'='*60}")
    print("  Model Comparison")
    print(f"{'='*60}{RESET}")
    print(f"  {'Model':<22} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>7} | {'F1-Score':>9}")
    print("  " + "-" * 57)
    for name, m in results.items():
        print(f"  {name:<22} | {m['accuracy']:>9.4f} | {m['precision']:>9.4f}"
              f" | {m['recall']:>7.4f} | {m['f1_score']:>9.4f}")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = EVAL_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {GREEN}Results saved → {out_path}{RESET}\n")


if __name__ == "__main__":
    main()
