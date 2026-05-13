#!/usr/bin/env bash
# scripts/evaluate.sh
# ─────────────────────────────────────────────────────────────────
# Stage 6 — Evaluate both models on the held-out test set
# Generates confusion matrices, ROC curves, comparison chart
# Usage: bash scripts/evaluate.sh
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load CUDA if available
NVIDIA_BASE="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    CUDA_LIBS="$(find "$NVIDIA_BASE" -type d -name "lib" 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Stage 6: Evaluation          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

python src/evaluator.py

echo ""
echo "✅  Evaluation complete."
echo "    Results: output/reports/stage6_evaluation/"
echo "      - cnn_scratch_confusion_matrix.png"
echo "      - cnn_scratch_roc_curves.png"
echo "      - efficientnetb0_confusion_matrix.png"
echo "      - efficientnetb0_roc_curves.png"
echo "      - model_comparison.png"
echo "      - evaluation_results.json"
