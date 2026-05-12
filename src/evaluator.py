"""
evaluator.py
------------
Stage 6 — Evaluate and compare both models on the held-out test set.

Generates:
  - Confusion matrices for both models
  - Per-class classification report (Precision, Recall, F1)
  - Side-by-side model comparison table
  - output/reports/evaluation_results.json

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
)
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.full_preprocessor import (
    build_scratch_pipeline,
    build_tl_pipeline,
    EMOTION_CLASSES,
)

REPORTS_DIR = Path("output/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Save a styled confusion-matrix heatmap."""
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
    print(f"  Confusion matrix saved → {save_path}")


def predict_from_generator(model, data_gen):
    """
    Run inference on an ImageDataGenerator (Pipeline A — CNN scratch).
    Returns (y_true, y_pred) as integer arrays.
    """
    data_gen.reset()
    steps = data_gen.samples // data_gen.batch_size + 1
    probs  = model.predict(data_gen, steps=steps, verbose=0)
    y_pred = np.argmax(probs, axis=1)[:data_gen.samples]
    y_true = data_gen.classes
    return y_true, y_pred


def predict_from_tf_dataset(model, dataset):
    """
    Run inference on a tf.data.Dataset (Pipeline B — Transfer Learning).
    Returns (y_true, y_pred) as integer arrays.
    """
    all_probs, all_labels = [], []
    for images, labels in dataset:
        probs = model.predict_on_batch(images)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob  = np.concatenate(all_probs,  axis=0)
    y_onehot= np.concatenate(all_labels, axis=0)
    y_pred  = np.argmax(y_prob,   axis=1)
    y_true  = np.argmax(y_onehot, axis=1)
    return y_true, y_pred


def evaluate_model(model_path, y_true, y_pred, model_name):
    """Compute metrics and save confusion matrix."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = np.mean(y_true == y_pred)

    print(f"\n  {model_name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=EMOTION_CLASSES,
        zero_division=0,
    ))

    plot_confusion_matrix(
        y_true, y_pred, EMOTION_CLASSES,
        title=f"Confusion Matrix — {model_name}",
        save_path=REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
    )

    return {
        "accuracy":  round(float(acc),       6),
        "precision": round(float(precision),  6),
        "recall":    round(float(recall),     6),
        "f1_score":  round(float(f1),         6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Stage 6 — Model Evaluation & Comparison")
    print("=" * 60 + "\n")

    results = {}

    # ── CNN Scratch (Pipeline A — ImageDataGenerator) ─────────────────────────
    cnn_path = Path("models/cnn_scratch_best.keras")
    if cnn_path.exists():
        print("[1/2] Loading CNN Scratch model + Pipeline A...")
        _, _, test_scratch = build_scratch_pipeline(batch_size=32)
        cnn_model = tf.keras.models.load_model(cnn_path)
        y_true, y_pred = predict_from_generator(cnn_model, test_scratch)
        results["CNN_Scratch"] = evaluate_model(cnn_path, y_true, y_pred, "CNN Scratch")
        del cnn_model   # free memory before loading next model
    else:
        print(f"  [SKIP] {cnn_path} not found.")

    # ── MobileNetV2 Transfer Learning (Pipeline B — tf.data) ──────────────────
    tl_path = Path("models/transfer_learning_best.keras")
    if tl_path.exists():
        print("[2/2] Loading MobileNetV2 model + Pipeline B...")
        _, _, test_tl = build_tl_pipeline(batch_size=32)
        tl_model = tf.keras.models.load_model(tl_path)
        y_true, y_pred = predict_from_tf_dataset(tl_model, test_tl)
        results["MobileNetV2"] = evaluate_model(tl_path, y_true, y_pred, "MobileNetV2")
    else:
        print(f"  [SKIP] {tl_path} not found.")

    if not results:
        print("No models found. Train at least one model first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(f"  {'Model':<22} | {'Accuracy':>9} | {'Precision':>9} | {'Recall':>7} | {'F1-Score':>9}")
    print("  " + "-" * 57)
    for name, m in results.items():
        print(
            f"  {name:<22} | {m['accuracy']:>9.4f} | {m['precision']:>9.4f}"
            f" | {m['recall']:>7.4f} | {m['f1_score']:>9.4f}"
        )
    print()

    # Save JSON
    out_path = REPORTS_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
