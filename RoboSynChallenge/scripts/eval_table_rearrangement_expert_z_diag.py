#!/usr/bin/env python
"""Replay expert trajectories at several grasp heights without editing configs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

import eval_policy as base_eval
from policy.act_table_rearrangement_diag.config_patch import patch_grasp_z

# Register the strict environment.
from policy.act_table_rearrangement_diag.strict_task import (  # noqa: F401
    TableRearrangementDiagnosticEnv,
)


TABLE_TOP_Z_M = 0.825


def tensor_xyz(tensor):
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def physical_gripper_qpos(env):
    robot = env.unwrapped.robot
    qpos = robot.get_qpos()
    left_id = robot.get_joint_ids("left_eef", remove_mimic=True)[0]
    right_id = robot.get_joint_ids("right_eef", remove_mimic=True)[0]
    return float(qpos[0, left_id].item()), float(qpos[0, right_id].item())


def trace_row(env, step, action):
    base_env = env.unwrapped
    metrics = base_env.diagnostic_metrics()
    robot = base_env.robot
    left_link6 = robot.get_link_pose("left_link6", to_matrix=True)[:, :3, 3]
    right_link6 = robot.get_link_pose("right_link6", to_matrix=True)[:, :3, 3]
    left_qpos, right_qpos = physical_gripper_qpos(env)
    return {
        "step": step,
        "expert_gripper_action_m": [
            float(action[0, 6].item()),
            float(action[0, 13].item()),
        ],
        "physical_gripper_qpos_m": [left_qpos, right_qpos],
        "fork_xyz_m": tensor_xyz(metrics["fork_xyz"]),
        "spoon_xyz_m": tensor_xyz(metrics["spoon_xyz"]),
        "plate_xyz_m": tensor_xyz(metrics["plate_xyz"]),
        "left_link6_xyz_m": tensor_xyz(left_link6),
        "right_link6_xyz_m": tensor_xyz(right_link6),
        "left_link6_table_clearance_m": float(
            left_link6[0, 2].item() - TABLE_TOP_Z_M
        ),
        "right_link6_table_clearance_m": float(
            right_link6[0, 2].item() - TABLE_TOP_Z_M
        ),
        "left_link6_to_fork_m": float(
            torch.linalg.vector_norm(
                left_link6 - metrics["fork_xyz"], dim=-1
            )[0].item()
        ),
        "right_link6_to_spoon_m": float(
            torch.linalg.vector_norm(
                right_link6 - metrics["spoon_xyz"], dim=-1
            )[0].item()
        ),
        "fork_lift_m": float(metrics["fork_lift"][0].item()),
        "spoon_lift_m": float(metrics["spoon_lift"][0].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="clear")
    parser.add_argument("--seed", type=int, default=924231285)
    parser.add_argument("--grasp-z-offset-m", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()

    config = {
        "task_name": "table_rearrangement",
        "setting": args.setting,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "headless": args.headless,
        "filter_dataset_saving": True,
        "eval_freeze_interval_events": True,
    }
    gym_config = deepcopy(base_eval.find_gym_config(config))
    gym_config["id"] = "TableRearrangementDiagnostic"
    action_config = deepcopy(base_eval.find_action_config(config))
    changed = patch_grasp_z(action_config, args.grasp_z_offset_m)
    if changed != 2:
        raise RuntimeError(f"Expected two grasp-z patches, got {changed}.")

    env, _ = base_eval.make_env_from_configs(config, gym_config, action_config)
    recorder = None
    if args.video_dir:
        recorder = base_eval.EpisodeVideoRecorder(
            args.video_dir,
            obs_keys=["cam_high", "cam_right_wrist", "cam_left_wrist"],
            fps=10,
        )
        recorder.start_episode(0, args.seed)

    success = False
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    try:
        obs, _ = env.reset(seed=args.seed)
        if recorder:
            recorder.record(obs)
        actions = env.unwrapped.create_demo_action_list()
        if actions is None:
            raise RuntimeError("Expert action generation returned None.")

        with args.trace.open("w", encoding="utf-8") as handle:
            for step, action in enumerate(actions):
                action = torch.as_tensor(
                    action, dtype=torch.float32, device=env.unwrapped.device
                )
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                obs, _, terminated, truncated, _ = env.step(action)
                env.unwrapped.update_diagnostic_state()
                handle.write(
                    json.dumps(trace_row(env, step, action), ensure_ascii=False) + "\n"
                )
                if recorder:
                    recorder.record(obs)
                if bool(torch.as_tensor(terminated).any().item()) or bool(
                    torch.as_tensor(truncated).any().item()
                ):
                    break
        success = bool(env.unwrapped.is_task_success().any().item())
        rows = [
            json.loads(line)
            for line in args.trace.read_text().splitlines()
            if line
        ]
        print(
            "[EXPERT Z DIAG]",
            f"offset_m={args.grasp_z_offset_m}",
            f"success={success}",
            f"min_left_link6_clearance_m={min(r['left_link6_table_clearance_m'] for r in rows):.6f}",
            f"min_right_link6_clearance_m={min(r['right_link6_table_clearance_m'] for r in rows):.6f}",
            f"max_fork_lift_m={max(r['fork_lift_m'] for r in rows):.6f}",
            f"max_spoon_lift_m={max(r['spoon_lift_m'] for r in rows):.6f}",
            flush=True,
        )
    finally:
        if recorder:
            recorder.close_episode(success=success)
        env.close()


if __name__ == "__main__":
    main()
