#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
GPU_ID="${1:-0}"
EPISODES="${2:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/probe_table_rearrangement_random_reset_events.py \
    --variant baseline \
    --rng-seed 0 \
    --episodes "$EPISODES" \
    --gpu-id "$GPU_ID" \
    --output diagnostic_logs/random_reset_probe_original_ids.csv

"$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_random_reset_probes.py \
    diagnostic_logs/random_reset_probe_original_ids.csv \
    diagnostic_logs/random_reset_probe_no_qpos_random.csv \
    diagnostic_logs/random_reset_probe_no_robot_random.csv \
    --output diagnostic_logs/random_reset_probe_summary.csv
