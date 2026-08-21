#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
GPU_ID="${1:-0}"

export MPLCONFIGDIR=/tmp/matplotlib-table-boundary-parallel
export HF_HOME=/tmp/hf-table-boundary
export HF_DATASETS_CACHE=/tmp/hf-table-boundary/datasets
cd "$REPO_ROOT"

pids=()
for spec in \
    "v2 31 20260826" \
    "v3 31 20260827" \
    "v4 31 20260828"
do
    read -r shard episodes seed <<< "$spec"
    "$PYTHON_BIN" policy/act/table_rearrangement_boundary/collect_boundary_fx.py \
        --episodes "$episodes" \
        --max-attempts 100 \
        --seed "$seed" \
        --headless \
        --gpu-id "$GPU_ID" \
        --output-root "lerobot_dataset/table_rearrangement_boundary_fx_${shard}" \
        > "diagnostic_logs/table_boundary_fx_collect31_${shard}.console.log" 2>&1 &
    pids+=("$!")
    echo "started shard=$shard pid=$! episodes=$episodes seed=$seed"
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "collector pid=$pid failed"
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "parallel collection failed: $failures shard(s)"
    exit 1
fi
echo "parallel collection complete: 93 episodes"
