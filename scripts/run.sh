#!/usr/bin/env bash
# Start Ollama (if not running), then Streamlit. From repo root: ./scripts/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! curl -sS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama (ollama serve)…"
  ollama serve &
  sleep 2
fi

if ! ollama list 2>/dev/null | grep -q llama3.2; then
  echo "Pulling llama3.2…"
  ollama pull llama3.2
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt

exec streamlit run app.py "$@"
