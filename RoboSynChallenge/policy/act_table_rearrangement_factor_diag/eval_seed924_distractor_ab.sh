#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
SEED="${3:-924231285}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
if (( $# > 3 )); then
    FACTORS=("${@:4}")
else
    FACTORS=(baseline distractors_hidden)
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_case() {
    local factor="$1"
    "$PYTHON_BIN" scripts/eval_table_rearrangement_factor_diag.py \
        --config policy/act_table_rearrangement_diag/deploy_policy.yml \
        --overrides \
        --policy_name act_table_rearrangement_factor_diag \
        --checkpoint_path "$CHECKPOINT" \
        --setting random \
        --train_config_name factor_ab \
        --model_name "seed${SEED}_${factor}" \
        --eval_fixed_episode_seed "$SEED" \
        --max_episodes 1 \
        --eval_video_log True \
        --diag_gripper_mode physical \
        --n_action_steps 50 \
        --diag_grasp_z_offset_m -0.008 \
        --diag_correct_random_joint_ids True \
        --diag_factor "$factor" \
        --diag_image_hash_every 10 \
        --diag_log_path "diagnostic_logs/factor_seed${SEED}_${factor}.jsonl" \
        2>&1 | tee "diagnostic_logs/factor_seed${SEED}_${factor}.console.log"
}

for factor in "${FACTORS[@]}"; do
    run_case "$factor"
done
