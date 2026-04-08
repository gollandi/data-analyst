#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh
# One-command setup for the notion_stats analysis environment.
# Requires: Python >= 3.11, pip
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on first error
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        notion_stats  — environment setup              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check Python version ─────────────────────────────────────────────────────
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.11"
if [[ "$(printf '%s\n' "$REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]]; then
    echo "❌  Python $REQUIRED+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓  Python $PYTHON_VERSION"

# ── Create virtual environment ───────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "→  Creating virtual environment..."
    python3 -m venv .venv
fi
echo "✓  Virtual environment"

# ── Activate ─────────────────────────────────────────────────────────────────
source .venv/bin/activate
echo "✓  Activated"

# ── Upgrade pip ──────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Install dependencies ─────────────────────────────────────────────────────
echo "→  Installing dependencies (this may take 2-3 min)..."
pip install -r requirements.txt --quiet
echo "✓  Dependencies installed"

# ── Create .env from template ────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "✓  .env created — fill in your NOTION_API_KEY"
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  Open .env and add your Notion integration token    │"
    echo "  │  (Settings → Connections → Develop or manage        │"
    echo "  │   integrations → New integration)                   │"
    echo "  └─────────────────────────────────────────────────────┘"
else
    echo "✓  .env already exists"
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p data/{raw,processed,snapshots} outputs/{figures,tables,reports}
echo "✓  Output directories ready"

# ── JupyterLab ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo ""
echo "  To start JupyterLab:"
echo "    source .venv/bin/activate"
echo "    jupyter lab"
echo ""
echo "  To run the analysis template directly:"
echo "    python notebooks/template_analysis.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
