#!/usr/bin/env bash
# scripts/full_pipeline.sh
# ─────────────────────────────────────────────────────────────────
# Run the COMPLETE FER2013 pipeline from scratch:
#   setup → prepare data → train (GPU) → evaluate → generate report
#
# Usage:
#   bash scripts/full_pipeline.sh          # run everything
#   bash scripts/full_pipeline.sh --cpu    # force CPU training
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USE_CPU=false
[[ "$1" == "--cpu" ]] && USE_CPU=true

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   FER2013 — FULL PIPELINE                           ║"
echo "║   Stages: Setup → Data → Train → Evaluate → Report  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

T_START=$SECONDS

echo "════════════ [1/5] Dependencies ════════════"
bash "$ROOT/scripts/setup.sh"

echo ""
echo "════════════ [2/5] Prepare Data ════════════"
bash "$ROOT/scripts/prepare_data.sh"

echo ""
echo "════════════ [3/5] Training ════════════════"
if [ "$USE_CPU" = true ]; then
    bash "$ROOT/scripts/train_cpu.sh"
else
    bash "$ROOT/scripts/train_gpu.sh"
fi

echo ""
echo "════════════ [4/5] Evaluation ══════════════"
bash "$ROOT/scripts/evaluate.sh"

echo ""
echo "════════════ [5/5] Report ══════════════════"
bash "$ROOT/scripts/generate_report.sh"

ELAPSED=$(( SECONDS - T_START ))
echo ""
echo "╔══════════════════════════════════════════╗"
printf  "║  ✅ FULL PIPELINE DONE  (%dm %ds)%s║\n" \
        $((ELAPSED/60)) $((ELAPSED%60)) "          "
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Models:  models/cnn_scratch_best.keras"
echo "           models/transfer_learning_best.keras"
echo "  Reports: output/reports/stage6_evaluation/"
echo "  Report:  documentation/report.html"
echo ""
echo "  Run the demo app:  bash scripts/run_app.sh"
