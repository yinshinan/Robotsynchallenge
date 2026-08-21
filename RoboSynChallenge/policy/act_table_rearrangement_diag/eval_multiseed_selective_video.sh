#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/eval_table_rearrangement_selective_video_diag.py \
    --config "$CONFIG" \
    --overrides \
    --checkpoint_path "$CHECKPOINT" \
    --setting random \
    --train_config_name multiseed_selective_video \
    --model_name physical_n50 \
    --seed 0 \
    --eval_fixed_episode_seed None \
    --max_episodes 20 \
    --eval_video_log True \
    --eval_video_obs_keys '["cam_high","cam_right_wrist","cam_left_wrist"]' \
    --diag_video_episode_indices 0,1,5 \
    --diag_gripper_mode physical \
    --n_action_steps 50 \
    --diag_grasp_z_offset_m -0.008 \
    --diag_correct_random_joint_ids True \
    --diag_log_path diagnostic_logs/multiseed_random_selective_video_physical_n50.jsonl \
    2>&1 | tee diagnostic_logs/multiseed_random_selective_video_physical_n50.console.log
