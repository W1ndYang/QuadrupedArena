#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

run_stamp="$(date +%Y%m%d-%H%M%S)"
workspace_dir="${ARENA_WORKSPACE:-$project_dir/.arena-workspace}"
run_dir="$workspace_dir/experiments/runs/quadruped-arena/rl-skill-10hz-$run_stamp"
bundle_dir="$workspace_dir/experiments/rl-skill-bundles"
bundle="$bundle_dir/rl_skill_10hz_ubuntu_validation_$run_stamp.tar.gz"
wrapper_log="$(mktemp "${TMPDIR:-/tmp}/rl-skill-ubuntu-validation.XXXXXX.log")"

mkdir -p "$bundle_dir"

set +e
ARENA_RUN_DIR="$run_dir" bash run.sh \
  --vision-trial \
  --semantic-skills \
  --rl-skill-10hz-sensor-smoke \
  --headless 2>&1 | tee "$wrapper_log"
validation_exit="${PIPESTATUS[0]}"

if [[ -d "$run_dir" ]]; then
  cp "$wrapper_log" "$run_dir/wrapper_stdout.log"
else
  mkdir -p "$run_dir"
  cp "$wrapper_log" "$run_dir/wrapper_stdout.log"
  printf '{"status":"failed_before_run_directory","exit_code":%s}\n' "$validation_exit" > "$run_dir/wrapper_result.json"
fi
rm -f "$wrapper_log"

printf '{"validation_exit_code":%s,"run_directory":"%s"}\n' "$validation_exit" "$run_dir" > "$run_dir/wrapper_result.json"

PYTHONPATH=scripts python3 scripts/rl_skill_package.py \
  "$run_dir" \
  --output "$bundle"
package_exit="$?"

echo "VALIDATION_EXIT=$validation_exit"
echo "PACKAGE_EXIT=$package_exit"
echo "RUN_DIR=$run_dir"
echo "UPLOAD_THIS=$bundle"

if [[ "$package_exit" -ne 0 ]]; then
  exit "$package_exit"
fi
exit "$validation_exit"
