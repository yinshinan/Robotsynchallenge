#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:-$REPO_ROOT/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/last/pretrained_model}"
GPU_ID="${2:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
SEEDS=(924231285 1650906866 1879422756)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

for seed in "${SEEDS[@]}"; do
    "$PYTHON_BIN" scripts/eval_table_rearrangement_reproducible_diag.py \
        --config policy/act_table_rearrangement_diag/deploy_policy.yml \
        --overrides \
        --checkpoint_path "$CHECKPOINT" \
        --setting random \
        --train_config_name reproducible_failure_video \
        --model_name "seed${seed}_physical_n50" \
        --seed 0 \
        --eval_fixed_episode_seed "$seed" \
        --max_episodes 1 \
        --eval_video_log True \
        --eval_video_obs_keys '["cam_high","cam_right_wrist","cam_left_wrist"]' \
        --diag_gripper_mode physical \
        --n_action_steps 50 \
        --diag_grasp_z_offset_m -0.008 \
        --diag_correct_random_joint_ids True \
        --diag_log_path "diagnostic_logs/repro_seed${seed}_physical_n50_video.jsonl" \
        2>&1 | tee "diagnostic_logs/repro_seed${seed}_physical_n50_video.console.log"
done
