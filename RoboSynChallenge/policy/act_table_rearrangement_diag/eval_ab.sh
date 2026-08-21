#!/bin/bash
set -euo pipefail

# Usage: bash policy/.../eval_ab.sh CHECKPOINT [GPU] [SEED] [MODE]
# MODE: all (default), policy, or expert
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CHECKPOINT="${1:?checkpoint path is required}"
GPU_ID="${2:-0}"
EPISODE_SEED="${3:-924231285}"
RUN_MODE="${4:-all}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
CONFIG="policy/act_table_rearrangement_diag/deploy_policy.yml"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$WORKSPACE_ROOT/EmbodiChain${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

run_case() {
    local case_name="$1"
    local gripper_mode="$2"
    local action_steps="$3"
    local grasp_z="$4"
    echo "Running diagnostic case: $case_name"
    "$PYTHON_BIN" scripts/eval_table_rearrangement_diag.py \
        --config "$CONFIG" \
        --overrides \
        --checkpoint_path "$CHECKPOINT" \
        --model_name "$case_name" \
        --eval_fixed_episode_seed "$EPISODE_SEED" \
        --diag_gripper_mode "$gripper_mode" \
        --n_action_steps "$action_steps" \
        --diag_grasp_z_offset_m "$grasp_z" \
        --diag_log_path "diagnostic_logs/${case_name}.jsonl"
}

if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "policy" ]]; then
    # Each policy case changes one factor relative to the preceding case. Grasp z is
    # held constant because action_config is used by expert generation, not ACT inference.
    run_case baseline_legacy_n50 legacy_scaled 50 -0.008
    run_case gripper_physical_n50 physical 50 -0.008
    run_case gripper_physical_n5 physical 5 -0.008
    run_case gripper_physical_n1 physical 1 -0.008

    "$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_traces.py \
        diagnostic_logs/baseline_legacy_n50.jsonl \
        diagnostic_logs/gripper_physical_n50.jsonl \
        diagnostic_logs/gripper_physical_n5.jsonl \
        diagnostic_logs/gripper_physical_n1.jsonl
fi

if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "expert" ]]; then
    # Grasp-height comparison is performed with generated expert trajectories.
    for grasp_z in -0.008 0.0 0.005; do
        case_name="expert_z_${grasp_z//./p}"
        "$PYTHON_BIN" scripts/eval_table_rearrangement_expert_z_diag.py \
            --headless \
            --gpu-id "$GPU_ID" \
            --seed "$EPISODE_SEED" \
            --grasp-z-offset-m "$grasp_z" \
            --trace "diagnostic_logs/${case_name}.jsonl" \
            --video-dir "diagnostic_logs/${case_name}_videos"
    done
    "$PYTHON_BIN" policy/act_table_rearrangement_diag/analyze_expert_traces.py \
        diagnostic_logs/expert_z_-0p008.jsonl \
        diagnostic_logs/expert_z_0p0.jsonl \
        diagnostic_logs/expert_z_0p005.jsonl
fi
