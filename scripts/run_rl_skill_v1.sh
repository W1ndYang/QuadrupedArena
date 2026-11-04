#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

run_stamp="$(date +%Y%m%d-%H%M%S)"
workspace_dir="${ARENA_WORKSPACE:-$project_dir/.arena-workspace}"
train_dir="$workspace_dir/experiments/rl-skill/$run_stamp"
eval_dir="$workspace_dir/experiments/rl-skill-eval/$run_stamp"
bundle_dir="$workspace_dir/experiments/rl-skill-bundles"

PYTHONPATH=scripts python3 -m unittest scripts.test_rl_skill_env -v
PYTHONPATH=scripts python3 scripts/rl_skill_train.py \
  --config configs/rl_skill_v1.json \
  --output-dir "$train_dir"
PYTHONPATH=scripts python3 scripts/rl_skill_eval.py \
  --config configs/rl_skill_v1.json \
  --checkpoint "$train_dir/checkpoints/final.npz" \
  --output-dir "$eval_dir"
mkdir -p "$bundle_dir"
PYTHONPATH=scripts python3 scripts/rl_skill_package.py \
  "$train_dir" "$eval_dir" \
  --output "$bundle_dir/rl_skill_v1_$run_stamp.tar.gz"

echo "TRAIN_DIR=$train_dir"
echo "EVAL_DIR=$eval_dir"
echo "BUNDLE=$bundle_dir/rl_skill_v1_$run_stamp.tar.gz"
