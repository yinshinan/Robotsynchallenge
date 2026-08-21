#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
EPISODES="${3:-10}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_case() {
    local name="$1"
    local x_min="$2"
    local x_max="$3"
    "$PYTHON_BIN" scripts/eval_table_rearrangement_spoon_x_diag.py \
        --config policy/act_table_rearrangement_diag/deploy_policy.yml \
        --overrides \
        --checkpoint_path "$CHECKPOINT" \
        --setting random \
        --train_config_name spoon_x_ab \
        --model_name "$name" \
        --seed 0 \
        --eval_fixed_episode_seed None \
        --max_episodes "$EPISODES" \
        --eval_video_log False \
        --diag_gripper_mode physical \
        --n_action_steps 50 \
        --diag_grasp_z_offset_m -0.008 \
        --diag_correct_random_joint_ids True \
        --diag_spoon_x_min_m "$x_min" \
        --diag_spoon_x_max_m "$x_max" \
        --diag_log_path "diagnostic_logs/spoon_x_${name}_physical_n50.jsonl" \
        2>&1 | tee "diagnostic_logs/spoon_x_${name}_physical_n50.console.log"
}

run_case edge_040_045 0.40 0.45
run_case interior_045_065 0.45 0.65
