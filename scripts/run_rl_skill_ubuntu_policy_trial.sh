#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/run_rl_skill_ubuntu_policy_trial.sh /absolute/path/to/final.npz" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
checkpoint="$1"
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint does not exist: $checkpoint" >&2
  exit 2
fi

run_stamp="$(date +%Y%m%d-%H%M%S)"
workspace_dir="${ARENA_WORKSPACE:-$project_dir/.arena-workspace}"
run_dir="$workspace_dir/experiments/runs/quadruped-arena/rl-skill-policy-$run_stamp"
bundle_dir="$workspace_dir/experiments/rl-skill-bundles"
bundle="$bundle_dir/rl_skill_isaac_policy_trial_$run_stamp.tar.gz"
wrapper_log="$(mktemp "${TMPDIR:-/tmp}/rl-skill-policy-trial.XXXXXX.log")"
seed="${RL_SKILL_POLICY_SEED:-34001}"

mkdir -p "$bundle_dir"
ARENA_RUN_DIR="$run_dir" bash run.sh \
  --vision-trial \
  --semantic-skills \
  --rl-skill-policy-checkpoint "$checkpoint" \
  --rl-skill-policy-seed "$seed" \
  --headless 2>&1 | tee "$wrapper_log"
trial_exit="${PIPESTATUS[0]}"

mkdir -p "$run_dir"
cp "$wrapper_log" "$run_dir/wrapper_stdout.log"
rm -f "$wrapper_log"
printf '{"trial_exit_code":%s,"match_seed":%s,"run_directory":"%s"}\n' \
  "$trial_exit" "$seed" "$run_dir" > "$run_dir/wrapper_result.json"

PYTHONPATH=scripts python3 scripts/rl_skill_package.py \
  "$run_dir" "$checkpoint" "$project_dir/scripts/run_rl_skill_ubuntu_policy_trial.sh" \
  --output "$bundle"
package_exit="$?"

echo "TRIAL_EXIT=$trial_exit"
echo "PACKAGE_EXIT=$package_exit"
echo "RUN_DIR=$run_dir"
echo "UPLOAD_THIS=$bundle"

if [[ "$package_exit" -ne 0 ]]; then exit "$package_exit"; fi
exit "$trial_exit"
