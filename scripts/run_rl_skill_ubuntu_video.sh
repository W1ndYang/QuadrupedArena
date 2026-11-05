#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
workspace_dir="${ARENA_WORKSPACE:-$project_dir/.arena-workspace}"
checkpoint="${1:-$workspace_dir/experiments/rl-skill-jobs/20261104-153934/train/checkpoints/final.npz}"
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint does not exist: $checkpoint" >&2
  echo "Usage: bash scripts/run_rl_skill_ubuntu_video.sh [absolute/path/to/final.npz]" >&2
  exit 2
fi

stamp="$(date +%Y%m%d-%H%M%S)"
video_root="$workspace_dir/experiments/rl-skill-video/rl-skill-video-$stamp"
online_dir="$video_root/online"
overview_dir="$video_root/overview"
bundle_dir="$workspace_dir/experiments/rl-skill-bundles"
bundle="$bundle_dir/rl_skill_video_$stamp.tar.gz"
seed="${RL_SKILL_POLICY_SEED:-34001}"
mkdir -p "$video_root" "$bundle_dir"

ARENA_RUN_DIR="$online_dir" bash run.sh \
  --vision-trial --semantic-skills \
  --rl-skill-policy-checkpoint "$checkpoint" \
  --rl-skill-policy-seed "$seed" --headless \
  2>&1 | tee "$video_root/online_stdout.log"
online_exit="${PIPESTATUS[0]}"
encode_exit=-1
overview_exit=-1

if [[ "$online_exit" -eq 0 ]]; then
  python3 scripts/rl_skill_video.py \
    --run-dir "$online_dir" \
    --output "$video_root/rl_skill_onboard.mp4" \
    > "$video_root/onboard_encode.log" 2>&1
  encode_exit="$?"
  if [[ "$encode_exit" -eq 0 ]]; then cat "$video_root/onboard_encode.log"; fi

  ARENA_RUN_DIR="$overview_dir" bash run.sh \
    --demo --replay-dir "$online_dir" --industrial --headless \
    2>&1 | tee "$video_root/overview_stdout.log"
  overview_exit="${PIPESTATUS[0]}"
  if [[ "$overview_exit" -eq 0 && -s "$overview_dir/demo.mp4" ]]; then
    cp "$overview_dir/demo.mp4" "$video_root/rl_skill_overview.mp4"
    cp "$overview_dir/poster.png" "$video_root/rl_skill_poster.png"
  else
    overview_exit=1
  fi
fi

python3 - "$video_root" "$checkpoint" "$seed" "$online_exit" "$encode_exit" "$overview_exit" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);checkpoint=Path(sys.argv[2]).resolve()
seed,online,encode,overview=map(int,sys.argv[3:])
result=dict(seed=seed,checkpoint=str(checkpoint),online_exit_code=online,
            onboard_encode_exit_code=encode,overview_exit_code=overview,
            overview_video=str(root/'rl_skill_overview.mp4') if overview==0 else None,
            onboard_video=str(root/'rl_skill_onboard.mp4') if encode==0 else None,
            overview_is_replay=True,policy_rerun_during_overview=False)
(root/'video_wrapper_result.json').write_text(json.dumps(result,indent=2)+'\n')
PY

PYTHONPATH=scripts python3 scripts/rl_skill_package.py \
  "$video_root" "$checkpoint" "$project_dir/scripts/run_rl_skill_ubuntu_video.sh" \
  --output "$bundle"
package_exit="$?"

echo "ONLINE_EXIT=$online_exit"
echo "ONBOARD_ENCODE_EXIT=$encode_exit"
echo "OVERVIEW_EXIT=$overview_exit"
echo "PACKAGE_EXIT=$package_exit"
echo "VIDEO_OVERVIEW=$video_root/rl_skill_overview.mp4"
echo "VIDEO_ONBOARD=$video_root/rl_skill_onboard.mp4"
echo "UPLOAD_THIS=$bundle"

if [[ "$package_exit" -ne 0 ]]; then exit "$package_exit"; fi
if [[ "$online_exit" -ne 0 || "$encode_exit" -ne 0 || "$overview_exit" -ne 0 ]]; then exit 1; fi
