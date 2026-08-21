#!/bin/bash
set -euo pipefail

# Re-evaluate random table_rearrangement with the original, validated arm IDs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
EPISODES="${3:-20}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"
TRACE="diagnostic_logs/multiseed_random_original_ids_physical_n50.jsonl"
CONSOLE="diagnostic_logs/multiseed_random_original_ids_physical_n50.console.log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/eval_table_rearrangement_diag.py \
    --config "$CONFIG" \
    --overrides \
    --checkpoint_path "$CHECKPOINT" \
    --setting random \
    --train_config_name multiseed_original_ids \
    --model_name physical_n50 \
    --seed 0 \
    --eval_fixed_episode_seed None \
    --max_episodes "$EPISODES" \
    --eval_video_log False \
    --diag_gripper_mode physical \
    --n_action_steps 50 \
    --diag_grasp_z_offset_m -0.008 \
    --diag_correct_random_joint_ids True \
    --diag_log_path "$TRACE" \
    2>&1 | tee "$CONSOLE"

if [[ "$EPISODES" == "20" ]]; then
    "$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_multiseed.py \
        --clear-trace diagnostic_logs/multiseed_clear_physical_n50.jsonl \
        --random-trace "$TRACE" \
        --clear-console diagnostic_logs/multiseed_clear_physical_n50.console.log \
        --random-console "$CONSOLE" \
        --rng-seed 0 \
        --episodes 20 \
        --output diagnostic_logs/table_rearrangement_multiseed_original_ids.csv \
        --summary-output diagnostic_logs/table_rearrangement_multiseed_original_ids_summary.csv
fi
