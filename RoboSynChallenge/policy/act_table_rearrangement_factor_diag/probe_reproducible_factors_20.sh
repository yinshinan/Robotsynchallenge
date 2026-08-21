#!/bin/bash
set -euo pipefail

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

"$PYTHON_BIN" scripts/probe_table_rearrangement_reproducible_factors.py \
    --rng-seed 0 \
    --episodes "$EPISODES" \
    --gpu-id "$GPU_ID" \
    --output diagnostic_logs/reproducible_reset_factors_20.csv \
    2>&1 | tee diagnostic_logs/reproducible_reset_factors_20.console.log
