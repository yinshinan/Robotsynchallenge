#!/bin/bash
set -euo pipefail

# Usage: bash policy/act_table_rearrangement_diag/eval_multiseed.sh CHECKPOINT [GPU] [EPISODES]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:?checkpoint path is required}"
GPU_ID="${2:-0}"
EPISODES="${3:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"
RNG_SEED=0

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_setting() {
    local setting="$1"
    local trace="diagnostic_logs/multiseed_${setting}_physical_n50.jsonl"
    local console_log="diagnostic_logs/multiseed_${setting}_physical_n50.console.log"
    echo "Running table_rearrangement multi-seed evaluation: setting=$setting episodes=$EPISODES"
    "$PYTHON_BIN" scripts/eval_table_rearrangement_diag.py \
        --config "$CONFIG" \
        --overrides \
        --checkpoint_path "$CHECKPOINT" \
        --setting "$setting" \
        --train_config_name multiseed \
        --model_name physical_n50 \
        --seed "$RNG_SEED" \
        --eval_fixed_episode_seed None \
        --max_episodes "$EPISODES" \
        --eval_video_log False \
        --diag_gripper_mode physical \
        --n_action_steps 50 \
        --diag_grasp_z_offset_m -0.008 \
        --diag_correct_random_joint_ids True \
        --diag_log_path "$trace" \
        2>&1 | tee "$console_log"
}

run_setting clear
run_setting random

"$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_multiseed.py \
    --clear-trace diagnostic_logs/multiseed_clear_physical_n50.jsonl \
    --random-trace diagnostic_logs/multiseed_random_physical_n50.jsonl \
    --clear-console diagnostic_logs/multiseed_clear_physical_n50.console.log \
    --random-console diagnostic_logs/multiseed_random_physical_n50.console.log \
    --rng-seed "$RNG_SEED" \
    --episodes "$EPISODES"
