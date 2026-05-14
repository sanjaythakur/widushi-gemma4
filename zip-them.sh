#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ARCHIVE="gemma-hackathon-source-$(date +%Y%m%d-%H%M%S).zip"
ARCHIVE_PATH="${1:-$ROOT_DIR/$DEFAULT_ARCHIVE}"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip is not installed or not on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"
ARCHIVE_DIR="$(cd "$(dirname "$ARCHIVE_PATH")" && pwd)"
ARCHIVE_BASENAME="$(basename "$ARCHIVE_PATH")"
ARCHIVE_PATH="$ARCHIVE_DIR/$ARCHIVE_BASENAME"

cd "$ROOT_DIR"
rm -f "$ARCHIVE_PATH"

echo "Creating $ARCHIVE_PATH"

zip -r "$ARCHIVE_PATH" . \
  -x ".git/*" \
  -x ".DS_Store" \
  -x "*/.DS_Store" \
  -x ".idea/*" \
  -x ".vscode/*" \
  -x "*.swp" \
  -x "*.swo" \
  -x "*~" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.pyd" \
  -x ".pytest_cache/*" \
  -x "*/.pytest_cache/*" \
  -x ".mypy_cache/*" \
  -x "*/.mypy_cache/*" \
  -x ".ruff_cache/*" \
  -x "*/.ruff_cache/*" \
  -x ".coverage" \
  -x "*/.coverage" \
  -x "htmlcov/*" \
  -x "*/htmlcov/*" \
  -x ".tox/*" \
  -x "*/.tox/*" \
  -x ".nox/*" \
  -x "*/.nox/*" \
  -x "coverage.xml" \
  -x "*/coverage.xml" \
  -x ".hypothesis/*" \
  -x "*/.hypothesis/*" \
  -x ".venv/*" \
  -x "*/.venv/*" \
  -x "venv/*" \
  -x "*/venv/*" \
  -x "env/*" \
  -x "*/env/*" \
  -x "node_modules/*" \
  -x "*/node_modules/*" \
  -x "build/*" \
  -x "*/build/*" \
  -x "dist/*" \
  -x "*/dist/*" \
  -x "*.egg" \
  -x "*.egg-info/*" \
  -x "*/.eggs/*" \
  -x "pip-wheel-metadata/*" \
  -x "*/pip-wheel-metadata/*" \
  -x "*.sqlite" \
  -x "*.sqlite3" \
  -x "*.gguf" \
  -x "*.gguf.*" \
  -x "*.onnx" \
  -x "*.tflite" \
  -x "*.pt" \
  -x "*.pth" \
  -x "*.safetensors" \
  -x "*/.hf-cache/*" \
  -x "gemma-llama/models/*" \
  -x "gemma-llama/voices/*" \
  -x "gemma-llama/tts_outputs/*" \
  -x "gemma-llama/eval/reports/*.json" \
  -x "docker-compose.override.yml" \
  -x "*/docker-compose.override.yml" \
  -x "$ARCHIVE_BASENAME"

echo "Done."
echo "Copy it with: scp \"$ARCHIVE_PATH\" <user>@<raspberry-pi-host>:~/"
