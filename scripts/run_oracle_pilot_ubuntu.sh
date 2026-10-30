#!/usr/bin/env bash
set -euo pipefail
PILOT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_PYTHON="${ARENA_PYTHON:-python}"
export ARENA_WORKSPACE="${ARENA_WORKSPACE:-$PILOT_PROJECT_DIR/.arena-workspace}"
exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_pilot.py" "$@"
