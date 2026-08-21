#!/bin/bash
set -euo pipefail

# Usage: bash policy/act_table_rearrangement_diag/probe_random_reset_events.sh [GPU] [EPISODES]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
GPU_ID="${1:-0}"
EPISODES="${2:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

for variant in baseline no_eef_random no_qpos_random no_robot_random; do
    "$PYTHON_BIN" scripts/probe_table_rearrangement_random_reset_events.py \
        --variant "$variant" \
        --gpu-id "$GPU_ID" \
        --episodes "$EPISODES"
done
