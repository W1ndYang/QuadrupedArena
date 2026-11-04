#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

run_stamp="$(date +%Y%m%d-%H%M%S)"
workspace_dir="${ARENA_WORKSPACE:-$project_dir/.arena-workspace}"
job_dir="$workspace_dir/experiments/rl-skill-jobs/$run_stamp"
train_dir="$job_dir/train"
eval_dir="$job_dir/eval"
bundle_dir="$workspace_dir/experiments/rl-skill-bundles"
bundle="$bundle_dir/rl_skill_v1_200k_$run_stamp.tar.gz"
mkdir -p "$job_dir" "$bundle_dir"

set +e
PYTHONPATH=scripts python3 -m unittest scripts.test_rl_skill_env -v 2>&1 | tee "$job_dir/tests.log"
test_exit="${PIPESTATUS[0]}"

train_exit=125
eval_exit=125
if [[ "$test_exit" -eq 0 ]]; then
  train_args=(--config configs/rl_skill_v1.json --output-dir "$train_dir")
  if [[ -n "${RL_SKILL_TOTAL_STEPS:-}" ]]; then train_args+=(--total-steps "$RL_SKILL_TOTAL_STEPS"); fi
  if [[ -n "${RL_SKILL_ROLLOUT_STEPS:-}" ]]; then train_args+=(--rollout-steps "$RL_SKILL_ROLLOUT_STEPS"); fi
  PYTHONPATH=scripts python3 scripts/rl_skill_train.py "${train_args[@]}" 2>&1 | tee "$job_dir/training_stdout.log"
  train_exit="${PIPESTATUS[0]}"
fi
if [[ "$train_exit" -eq 0 ]]; then
  PYTHONPATH=scripts python3 scripts/rl_skill_eval.py \
    --config configs/rl_skill_v1.json \
    --checkpoint "$train_dir/checkpoints/final.npz" \
    --output-dir "$eval_dir" 2>&1 | tee "$job_dir/evaluation_stdout.log"
  eval_exit="${PIPESTATUS[0]}"
fi

printf '{"test_exit_code":%s,"train_exit_code":%s,"evaluation_exit_code":%s}\n' \
  "$test_exit" "$train_exit" "$eval_exit" > "$job_dir/job_result.json"

PYTHONPATH=scripts python3 scripts/rl_skill_package.py \
  "$job_dir" \
  --output "$bundle"
package_exit="$?"
set -e

echo "TEST_EXIT=$test_exit"
echo "TRAIN_EXIT=$train_exit"
echo "EVALUATION_EXIT=$eval_exit"
echo "PACKAGE_EXIT=$package_exit"
echo "TRAIN_DIR=$train_dir"
echo "EVAL_DIR=$eval_dir"
echo "UPLOAD_THIS=$bundle"

if [[ "$package_exit" -ne 0 ]]; then exit "$package_exit"; fi
if [[ "$test_exit" -ne 0 ]]; then exit "$test_exit"; fi
if [[ "$train_exit" -ne 0 ]]; then exit "$train_exit"; fi
exit "$eval_exit"
