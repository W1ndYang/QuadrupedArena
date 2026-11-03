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
  oracle_p2_second_stage)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_p2_second_stage.py" "$@"
    ;;
  oracle_state_discovery)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_refinement_discovery.py" "$@"
    ;;
  oracle_state_actions)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_refinement_actions.py" "$@"
    ;;
  oracle_state_support)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_refinement_support.py" "$@"
    ;;
  oracle_state_prospective)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_prospective_validation.py" "$@"
    ;;
  oracle_state_new_game)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_prospective_validation.py" \
      --validation-game-seed 2026102108 --validation-synthetic-seed 20261102 \
      --reuse-discovery-bundle "$ARENA_WORKSPACE/experiments/oracle-state-new-game-validation/oracle_state_new_game_validation_bundle.zip" \
      --output "$ARENA_WORKSPACE/experiments/oracle-state-new-game-validation-v2" "$@"
    ;;
  oracle_state_seed_sweep)
    shift
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_state_seed_sweep.py" "$@"
    ;;
  analyze_bundle|replay_validate)
    exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_pilot_v21.py" "$@"
    ;;
esac
exec "$PILOT_PYTHON" "$PILOT_PROJECT_DIR/scripts/oracle_pilot.py" "$@"
