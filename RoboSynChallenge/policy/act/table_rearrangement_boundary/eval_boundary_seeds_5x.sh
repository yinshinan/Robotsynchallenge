#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:?usage: $0 CHECKPOINT [GPU_ID] [LABEL] [REPEATS]}"
GPU_ID="${2:-0}"
LABEL="${3:-boundary_ft}"
REPEATS="${4:-5}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
SEEDS=(924231285 1879422756)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

for seed in "${SEEDS[@]}"; do
    for ((rep=1; rep<=REPEATS; rep++)); do
        trace="diagnostic_logs/${LABEL}_seed${seed}_rep${rep}.jsonl"
        console="diagnostic_logs/${LABEL}_seed${seed}_rep${rep}.console.log"
        "$PYTHON_BIN" scripts/eval_table_rearrangement_factor_diag.py \
            --config policy/act_table_rearrangement_diag/deploy_policy.yml \
            --overrides \
            --policy_name act_table_rearrangement_factor_diag \
            --checkpoint_path "$CHECKPOINT" \
            --setting random \
            --train_config_name "$LABEL" \
            --model_name "seed${seed}_rep${rep}" \
            --eval_fixed_episode_seed "$seed" \
            --max_episodes 1 \
            --eval_video_log False \
            --diag_gripper_mode physical \
            --n_action_steps 50 \
            --diag_grasp_z_offset_m -0.008 \
            --diag_correct_random_joint_ids True \
            --diag_factor baseline \
            --diag_image_hash_every 0 \
            --diag_log_path "$trace" \
            > "$console" 2>&1
        status=$(tr -d '\000' < "$console" | sed -r 's/\x1B\[[0-9;]*[mK]//g' | rg 'Episode 01/01:' | tail -n 1 || true)
        if [[ -z "$status" ]]; then
            steps=$(wc -l < "$trace")
            if (( steps < 361 )); then status="SUCCESS(trace_steps=$steps)"; else status="FAIL(trace_steps=$steps)"; fi
        fi
        echo "seed=$seed rep=$rep $status"
    done
done
