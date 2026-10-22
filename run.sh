#!/usr/bin/env bash
set -euo pipefail
ARENA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARENA_PYTHON="${ARENA_PYTHON:-python}"
export ARENA_WORKSPACE="${ARENA_WORKSPACE:-$ARENA_PROJECT_DIR/.arena-workspace}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$ARENA_WORKSPACE/projects/unitree/IsaacLab}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$ARENA_WORKSPACE/projects/unitree/unitree_ros}"
# Other users must review and explicitly accept NVIDIA's terms themselves.
if [[ " $* " != *" --help "* && "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
  echo "Review NVIDIA's Isaac Sim license, then export OMNI_KIT_ACCEPT_EULA=YES. See docs/INSTALL.md." >&2
  exit 2
fi
ulimit -Sn 65535 2>/dev/null || true
exec "$ARENA_PYTHON" "$ARENA_PROJECT_DIR/scripts/launch.py" "$@"
