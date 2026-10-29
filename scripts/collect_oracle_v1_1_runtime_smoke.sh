#!/usr/bin/env bash
set -u
COLLECTOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$COLLECTOR_DIR/collect_oracle_v1_1_runtime_smoke.py"
