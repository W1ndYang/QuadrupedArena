#!/usr/bin/env bash
set -euo pipefail
PILOT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_PYTHON="${ARENA_PYTHON:-python3}"
export ARENA_WORKSPACE="${ARENA_WORKSPACE:-$PILOT_PROJECT_DIR/.arena-workspace}"
case "${1:-}" in
  finite_oracle_run)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_finite_v1.py" "$@"
    ;;
  finite_oracle_long)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_finite_v1_long.py" "$@"
    ;;
  oracle_local_p1)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_local_p1_v1.py" "$@"
    ;;
  oracle_local_full_game)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_local_p1_full_game.py" "$@"
    ;;
  analyze_bundle|replay_validate)
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_pilot_v21.py" "$@"
    ;;
esac
exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_pilot.py" "$@"
