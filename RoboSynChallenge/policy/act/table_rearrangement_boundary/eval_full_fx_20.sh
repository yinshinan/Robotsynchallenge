#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:?usage: $0 CHECKPOINT [GPU_ID] [LABEL] [EPISODES]}"
GPU_ID="${2:-0}"
LABEL="${3:-boundary_ft_fullfx}"
EPISODES="${4:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
TRACE="diagnostic_logs/${LABEL}_${EPISODES}.jsonl"
CONSOLE="diagnostic_logs/${LABEL}_${EPISODES}.console.log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/eval_table_rearrangement_factor_diag.py \
    --config policy/act_table_rearrangement_diag/deploy_policy.yml \
    --overrides \
    --policy_name act_table_rearrangement_factor_diag \
    --checkpoint_path "$CHECKPOINT" \
    --setting random \
    --train_config_name "$LABEL" \
    --model_name full_fx \
    --seed 0 \
    --eval_fixed_episode_seed None \
    --max_episodes "$EPISODES" \
    --eval_video_log False \
    --diag_gripper_mode physical \
    --n_action_steps 50 \
    --diag_grasp_z_offset_m -0.008 \
    --diag_correct_random_joint_ids True \
    --diag_factor baseline \
    --diag_image_hash_every 0 \
    --diag_log_path "$TRACE" \
    2>&1 | tee "$CONSOLE"
