#!/bin/bash
set -euo pipefail

# Usage: bash policy/act_table_rearrangement_diag/eval_random_gripper_reset_ab.sh CHECKPOINT [GPU] [EPISODES]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:?checkpoint path is required}"
GPU_ID="${2:-0}"
EPISODES="${3:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"
TRACE="diagnostic_logs/random_gripper_reset_physical_n50.jsonl"
CONSOLE_LOG="diagnostic_logs/random_gripper_reset_physical_n50.console.log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/eval_table_rearrangement_random_gripper_reset_diag.py \
    --config "$CONFIG" \
    --overrides \
    --checkpoint_path "$CHECKPOINT" \
    --setting random \
    --train_config_name multiseed \
    --model_name physical_n50_gripper_reset \
    --seed 0 \
    --eval_fixed_episode_seed None \
    --max_episodes "$EPISODES" \
    --eval_video_log False \
    --diag_gripper_mode physical \
    --n_action_steps 50 \
    --diag_grasp_z_offset_m -0.008 \
    --diag_log_path "$TRACE" \
    2>&1 | tee "$CONSOLE_LOG"

"$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_random_gripper_reset_ab.py \
    --baseline-csv diagnostic_logs/table_rearrangement_multiseed.csv \
    --reset-trace "$TRACE" \
    --reset-console "$CONSOLE_LOG" \
    --episodes "$EPISODES"
