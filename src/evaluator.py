"""
evaluator.py
------------
Stage 6 — Evaluate and compare the models.

Calculates:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Model Comparison Table
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.full_preprocessor import build_scratch_pipeline, build_tl_pipeline, EMOTION_CLASSES

REPORTS_DIR = Path("output/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, data_gen, model_name):
    print(f"Evaluating {model_name}...")
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    steps = data_gen.samples // data_gen.batch_size + 1
    predictions = model.predict(data_gen, steps=steps)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = data_gen.classes
    
    # Trim to match sizes if generator looped over
    y_pred = y_pred[:len(y_true)]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = np.mean(y_true == y_pred)
    
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    plot_confusion_matrix(
        y_true, y_pred, EMOTION_CLASSES, 
        f"Confusion Matrix: {model_name}", 
        REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    )
    
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}

def main():
    print("Loading test data...")
    _, _, test_scratch = build_scratch_pipeline(batch_size=32)
    _, _, test_tl = build_tl_pipeline(batch_size=32)
    
    test_scratch.shuffle = False
    test_tl.shuffle = False

    results = {}
    
    cnn_path = Path("models/cnn_scratch_best.keras")
    if cnn_path.exists():
        results["CNN_Scratch"] = evaluate_model(cnn_path, test_scratch, "CNN_Scratch")
    else:
        print(f"Warning: {cnn_path} not found.")

    tl_path = Path("models/transfer_learning_best.keras")
    if tl_path.exists():
        results["MobileNetV2"] = evaluate_model(tl_path, test_tl, "MobileNetV2")
    else:
        print(f"Warning: {tl_path} not found.")

    print("\n--- Model Comparison Table ---")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 45)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['accuracy']:<10.4f} | {metrics['f1_score']:<10.4f}")

    with open(REPORTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
