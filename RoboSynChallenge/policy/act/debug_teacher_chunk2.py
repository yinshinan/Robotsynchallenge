#!/usr/bin/env python
"""Compare ACT predictions with the current expert along an expert rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EMBODICHAIN_ROOT = Path(
    os.environ.get("EMBODICHAIN_ROOT", WORKSPACE_ROOT / "EmbodiChain")
)
for path in reversed((REPO_ROOT, REPO_ROOT / "policy", EMBODICHAIN_ROOT)):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym  # noqa: E402

import robosynchallenge  # noqa: F401,E402
from scripts.eval_policy import make_env_from_configs  # noqa: E402
from policy.act.deploy_policy import get_model  # noqa: E402


def button_press_depth(env) -> float:
    qpos = env.unwrapped.sim.get_articulation("button").get_qpos()
    return float((-qpos[:, 0]).max().item())


def model_batch(model, obs) -> dict[str, torch.Tensor]:
    state = obs
    for key in str(model.state_obs_path).split("/"):
        if key:
            state = state[key]

    batch = {
        "observation.state": torch.as_tensor(
            state, dtype=torch.float32, device=model.act_device
        ),
        "observation.qvel": torch.as_tensor(
            obs["robot"]["qvel"], dtype=torch.float32, device=model.act_device
        ),
        "observation.qf": torch.as_tensor(
            obs["robot"]["qf"], dtype=torch.float32, device=model.act_device
        ),
    }
    for key, value in list(batch.items()):
        if value.ndim == 1:
            batch[key] = value.unsqueeze(0)

    for image_key in model.act_image_keys:
        camera_name = model.image_key_map.get(
            image_key, image_key.removeprefix("observation.images.")
        )
        image = torch.as_tensor(
            obs["sensor"][camera_name]["color"],
            dtype=torch.float32,
            device=model.act_device,
        )
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image[..., :3].permute(0, 3, 1, 2).contiguous()
        if image.max() > 1.5:
            image = image / 255.0
        batch[image_key] = image
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--setting", choices=("clear", "random"), default="clear")
    parser.add_argument("--seed", type=int, default=209652396)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reset-sync-steps", type=int, default=1)
    parser.add_argument(
        "--csv",
        default="/tmp/act_click_bell_teacher_forced.csv",
        help="Output CSV containing state, expert action, prediction, and depth.",
    )
    args = parser.parse_args()

    with (REPO_ROOT / f"configs/click_bell/{args.setting}/gym_config.json").open(
        encoding="utf-8"
    ) as stream:
        gym_config = json.load(stream)
    with (REPO_ROOT / "configs/click_bell/action_config.json").open(
        encoding="utf-8"
    ) as stream:
        action_config = json.load(stream)

    env_config = {
        "num_envs": 1,
        "device": "cpu",
        "headless": args.headless,
        "renderer": "hybrid",
        "gpu_id": args.gpu_id,
        "arena_space": 5.0,
        "max_steps": 361,
        "filter_dataset_saving": True,
    }
    env, _ = make_env_from_configs(env_config, gym_config, action_config)
    model = get_model(
        {
            "checkpoint_path": args.checkpoint,
            "device": "cuda",
            "act_step": 1,
            "n_action_steps": 1,
            "state_obs_path": "robot/qpos",
            "strict_action_dim": True,
            "neutralize_qvel": False,
            "debug_button_press": False,
        }
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    try:
        obs, _ = env.reset(seed=args.seed)
        if args.reset_sync_steps > 0:
            unwrapped = env.unwrapped
            physics_dt = getattr(unwrapped.sim_cfg, "physics_dt", None)
            unwrapped.sim.update(physics_dt, args.reset_sync_steps)
            obs = unwrapped.get_obs()

        expert_actions = env.get_wrapper_attr("create_demo_action_list")(
            action_sentence=0
        )
        if expert_actions is None or len(expert_actions) == 0:
            raise RuntimeError("Current expert action generator returned no actions.")

        print(
            "[TEACHER CONFIG]",
            f"setting={args.setting}",
            f"seed={args.seed}",
            f"steps={len(expert_actions)}",
            f"csv={csv_path}",
            flush=True,
        )

        first_success_step = None
        for step, expert_action in enumerate(expert_actions, start=1):
            # Reset the queue so this is always the first prediction conditioned
            # on the current expert observation, not a cached open-loop action.
            model.reset()
            predicted = model.predict_action_chunk(model_batch(model, obs))[:, 2]
            predicted_cpu = predicted.detach().to("cpu", dtype=torch.float32)
            expert_cpu = torch.as_tensor(expert_action, dtype=torch.float32).cpu()
            if predicted_cpu.ndim == 1:
                predicted_cpu = predicted_cpu.unsqueeze(0)
            if expert_cpu.ndim == 1:
                expert_cpu = expert_cpu.unsqueeze(0)

            state_cpu = torch.as_tensor(
                obs["robot"]["qpos"], dtype=torch.float32
            ).cpu()
            if state_cpu.ndim == 1:
                state_cpu = state_cpu.unsqueeze(0)

            absolute_error = (predicted_cpu - expert_cpu).abs()[0]
            depth_before = button_press_depth(env)
            obs, _, _, _, _ = env.step(expert_action)
            depth_after = button_press_depth(env)
            success = bool(env.get_wrapper_attr("is_task_success")().any().item())
            if success and first_success_step is None:
                first_success_step = step

            record: dict[str, object] = {
                "step": step,
                "depth_before": depth_before,
                "depth_after": depth_after,
                "full_mae": float(absolute_error.mean().item()),
                "left_mae": float(absolute_error[:7].mean().item()),
                "right_mae": float(absolute_error[7:14].mean().item()),
            }
            for joint in range(14):
                record[f"state_{joint}"] = float(state_cpu[0, joint].item())
                record[f"expert_{joint}"] = float(expert_cpu[0, joint].item())
                record[f"predicted_{joint}"] = float(predicted_cpu[0, joint].item())
                record[f"abs_error_{joint}"] = float(absolute_error[joint].item())
            records.append(record)

            if 35 <= step <= 55:
                print(
                    "[TEACHER STEP]",
                    f"step={step}",
                    f"depth={depth_after:.6f}",
                    f"right_mae={record['right_mae']:.6f}",
                    flush=True,
                )

        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        full_mae = np.asarray([record["full_mae"] for record in records])
        right_mae = np.asarray([record["right_mae"] for record in records])
        contact_slice = slice(39, min(55, len(records)))  # one-based steps 40..55
        joint_mae = np.asarray(
            [
                np.mean([record[f"abs_error_{joint}"] for record in records])
                for joint in range(14)
            ]
        )
        contact_joint_mae = np.asarray(
            [
                np.mean(
                    [
                        record[f"abs_error_{joint}"]
                        for record in records[contact_slice]
                    ]
                )
                for joint in range(14)
            ]
        )
        print(
            "[TEACHER SUMMARY]",
            f"mean_mae={full_mae.mean():.6f}",
            f"max_step_mae={full_mae.max():.6f}",
            f"mean_right_mae={right_mae.mean():.6f}",
            f"contact_right_mae={right_mae[contact_slice].mean():.6f}",
            f"first_success_step={first_success_step}",
            flush=True,
        )
        print(
            "[TEACHER JOINT MAE] all=",
            np.array2string(
                joint_mae, precision=6, separator=",", max_line_width=1000
            ),
            flush=True,
        )
        print(
            "[TEACHER JOINT MAE] contact=",
            np.array2string(
                contact_joint_mae,
                precision=6,
                separator=",",
                max_line_width=1000,
            ),
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
