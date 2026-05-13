#!/usr/bin/env bash
# scripts/run_app.sh
# ─────────────────────────────────────────────────────────────────
# Launch the Streamlit demo app
# Loads GPU libs if available (speeds up model inference)
# Usage: bash scripts/run_app.sh [--port 8501]
# ─────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load CUDA libs if available
NVIDIA_BASE="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    CUDA_LIBS="$(find "$NVIDIA_BASE" -type d -name "lib" 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"
    echo "✅  GPU libs loaded."
fi

PORT=8501
for arg in "$@"; do
    [[ "$arg" == "--port" ]] && { shift; PORT="$1"; }
done

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FER2013 — Streamlit App                ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Open: http://localhost:${PORT}"
echo "  Stop: Ctrl + C"
echo ""

streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true
