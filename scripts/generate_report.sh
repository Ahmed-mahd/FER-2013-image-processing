#!/usr/bin/env bash
# scripts/generate_report.sh
# ─────────────────────────────────────────────────────────────────
# Stage 8 — Regenerate the HTML documentation report
# Opens it in the default browser when done
# Usage: bash scripts/generate_report.sh
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Stage 8: Generate Report     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

python documentation/generate_report.py

echo ""
echo "✅  Report generated: documentation/report.html"
echo "    → Open in browser and press Ctrl+P → Save as PDF"
echo ""

# Auto-open in browser if available
if command -v xdg-open &> /dev/null; then
    xdg-open "$ROOT/documentation/report.html"
elif command -v open &> /dev/null; then
    open "$ROOT/documentation/report.html"
fi
