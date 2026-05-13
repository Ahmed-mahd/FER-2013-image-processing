#!/usr/bin/env bash
# scripts/train_gpu.sh
# ─────────────────────────────────────────────────────────────────
# Train both models with GPU acceleration (NVIDIA CUDA)
# Automatically detects NVIDIA libraries from pip-installed packages
# Usage: bash scripts/train_gpu.sh [--skip-scratch] [--skip-transfer]
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Auto-detect CUDA libraries ────────────────────────────────────
NVIDIA_BASE="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    CUDA_LIBS="$(find "$NVIDIA_BASE" -type d -name "lib" 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"
    echo "✅  CUDA libraries loaded from: $NVIDIA_BASE"
else
    echo "⚠️   NVIDIA pip packages not found — GPU may not be available."
    echo "    Run: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Training (GPU Mode)          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

SKIP_SCRATCH=false
SKIP_TRANSFER=false
for arg in "$@"; do
    [[ "$arg" == "--skip-scratch"   ]] && SKIP_SCRATCH=true
    [[ "$arg" == "--skip-transfer"  ]] && SKIP_TRANSFER=true
done

# ── Stage 4: CNN from Scratch ─────────────────────────────────────
if [ "$SKIP_SCRATCH" = false ]; then
    echo "─── Stage 4: CNN from Scratch ───────────────────────────────"
    python training/train_scratch.py
    echo ""
fi

# ── Stage 5: Transfer Learning (EfficientNetB0) ───────────────────
if [ "$SKIP_TRANSFER" = false ]; then
    echo "─── Stage 5: EfficientNetB0 Transfer Learning ───────────────"
    python training/train_transfer.py
    echo ""
fi

echo "✅  Training complete."
echo "    Models:  models/cnn_scratch_best.keras"
echo "             models/transfer_learning_best.keras"
echo ""
echo "Next: bash scripts/evaluate.sh"
