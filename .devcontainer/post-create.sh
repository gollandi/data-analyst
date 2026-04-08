#!/usr/bin/env bash
set -e

echo "→ Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "→ Creating output directories..."
mkdir -p data/{raw,processed,snapshots} outputs/{figures,tables,reports}

echo "→ Creating .env from template..."
if [ ! -f ".env" ]; then
  cp .env.template .env
fi

echo ""
echo "✅ Environment ready."
echo "   Add NOTION_API_KEY to Codespaces secrets:"
echo "   github.com/settings/codespaces → New secret"
echo ""
