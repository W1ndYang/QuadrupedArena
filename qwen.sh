#!/usr/bin/env bash
set -euo pipefail
ARENA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ARENA_PYTHON:-python}" "$ARENA_PROJECT_DIR/scripts/qwen_trial.py" "$@"
