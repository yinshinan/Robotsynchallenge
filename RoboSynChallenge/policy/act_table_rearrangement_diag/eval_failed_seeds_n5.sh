#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"
SEEDS=(209652396 398764591 1537364731)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

for seed in "${SEEDS[@]}"; do
    trace="diagnostic_logs/random_original_ids_seed${seed}_physical_n5.jsonl"
    console="diagnostic_logs/random_original_ids_seed${seed}_physical_n5.console.log"
    "$PYTHON_BIN" scripts/eval_table_rearrangement_diag.py \
        --config "$CONFIG" \
        --overrides \
        --checkpoint_path "$CHECKPOINT" \
        --setting random \
        --train_config_name "seed${seed}_n5" \
        --model_name physical_n5 \
        --seed 0 \
        --eval_fixed_episode_seed "$seed" \
        --max_episodes 1 \
        --eval_video_log False \
        --diag_gripper_mode physical \
        --n_action_steps 5 \
        --diag_grasp_z_offset_m -0.008 \
        --diag_correct_random_joint_ids True \
        --diag_log_path "$trace" \
        2>&1 | tee "$console"
done
