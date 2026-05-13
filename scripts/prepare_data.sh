#!/usr/bin/env bash
# scripts/prepare_data.sh
# ─────────────────────────────────────────────────────────────────
# Run Stage 2 EDA quality checks + Stage 3 preprocessing
# (augments Disgust class ×4, computes class weights)
# Usage: bash scripts/prepare_data.sh
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Stage 2+3: Data Preparation  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

echo "→ Stage 2: EDA quality checks (210-image batch) ..."
python training/main.py

echo ""
echo "→ Stage 3: Augmentation + class weights ..."
python training/prepare_data.py

echo ""
echo "✅  Data preparation complete."
echo "    Outputs: output/reports/stage2_eda/ and output/reports/stage3_preprocessing/"
echo ""
echo "Next: bash scripts/train_gpu.sh  (or train_cpu.sh)"
