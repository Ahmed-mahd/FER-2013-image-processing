#!/usr/bin/env bash
# scripts/train_cpu.sh
# ─────────────────────────────────────────────────────────────────
# Train both models on CPU only (no CUDA required)
# Slower but works on any machine without a GPU
# Usage: bash scripts/train_cpu.sh [--skip-scratch] [--skip-transfer]
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Force CPU — hide GPU from TensorFlow
export CUDA_VISIBLE_DEVICES=""

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Training (CPU Mode)          ║"
echo "║   ⚠️  This will be slow (1–3 hours)       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

SKIP_SCRATCH=false
SKIP_TRANSFER=false
for arg in "$@"; do
    [[ "$arg" == "--skip-scratch"   ]] && SKIP_SCRATCH=true
    [[ "$arg" == "--skip-transfer"  ]] && SKIP_TRANSFER=true
done

if [ "$SKIP_SCRATCH" = false ]; then
    echo "─── Stage 4: CNN from Scratch ───────────────────────────────"
    python training/train_scratch.py
    echo ""
fi

if [ "$SKIP_TRANSFER" = false ]; then
    echo "─── Stage 5: EfficientNetB0 Transfer Learning ───────────────"
    python training/train_transfer.py
    echo ""
fi

echo "✅  Training complete (CPU)."
echo "    Models:  models/cnn_scratch_best.keras"
echo "             models/transfer_learning_best.keras"
