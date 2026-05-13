#!/usr/bin/env bash
# scripts/setup.sh
# ─────────────────────────────────────────────────────────────────
# Install all project dependencies
# Usage: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Environment Setup            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Install pip packages
echo "→ Installing Python dependencies from requirements.txt ..."
pip install -r requirements.txt --break-system-packages -q

# Install streamlit separately (optional, for the app)
echo "→ Installing streamlit ..."
pip install streamlit --break-system-packages -q

echo ""
echo "✅  All dependencies installed."
echo ""
echo "Next steps:"
echo "  1. Download FER2013 dataset:"
echo "     kaggle datasets download -d msambare/fer2013"
echo "     unzip fer2013.zip -d data/"
echo "  2. Run: bash scripts/prepare_data.sh"
