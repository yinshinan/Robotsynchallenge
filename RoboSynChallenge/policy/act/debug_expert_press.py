#!/usr/bin/env python
"""Run the current click_bell expert and measure physical button travel.

This diagnostic deliberately disables dataset saving and does not load a policy.
It tests the current environment, action bank, controller, and button asset together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

import gymnasium as gym

import robosynchallenge  # noqa: F401,E402
from scripts.eval_policy import make_env_from_configs  # noqa: E402


def as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return bool(value)


def button_press_depth(env) -> float:
    qpos = env.unwrapped.sim.get_articulation("button").get_qpos()
    return float((-qpos[:, 0]).max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("clear", "random"), default="clear")
    parser.add_argument("--seed", type=int, default=209652396)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reset-sync-steps", type=int, default=1)
    args = parser.parse_args()

    gym_path = REPO_ROOT / f"configs/click_bell/{args.setting}/gym_config.json"
    action_path = REPO_ROOT / "configs/click_bell/action_config.json"
    with gym_path.open(encoding="utf-8") as stream:
        gym_config = json.load(stream)
    with action_path.open(encoding="utf-8") as stream:
        action_config = json.load(stream)

    config = {
        "num_envs": 1,
        "device": "cpu",
        "headless": args.headless,
        "renderer": "hybrid",
        "gpu_id": args.gpu_id,
        "arena_space": 5.0,
        "max_steps": 361,
        "filter_dataset_saving": True,
    }
    env, _ = make_env_from_configs(config, gym_config, action_config)

    try:
        env.reset(seed=args.seed)
        if args.reset_sync_steps > 0:
            unwrapped = env.unwrapped
            physics_dt = getattr(unwrapped.sim_cfg, "physics_dt", None)
            unwrapped.sim.update(physics_dt, args.reset_sync_steps)

        actions = env.get_wrapper_attr("create_demo_action_list")(
            action_sentence=0
        )
        if actions is None or len(actions) == 0:
            raise RuntimeError("Current expert action generator returned no actions.")

        max_depth = button_press_depth(env)
        first_success_step = None
        final_info = None
        terminated_seen = False
        truncated_seen = False

        print(
            "[EXPERT CONFIG]",
            f"setting={args.setting}",
            f"seed={args.seed}",
            f"actions={len(actions)}",
            f"reset_sync_steps={args.reset_sync_steps}",
            flush=True,
        )

        for step, action in enumerate(actions, start=1):
            _, _, terminated, truncated, final_info = env.step(action)
            depth = button_press_depth(env)
            max_depth = max(max_depth, depth)
            success = as_bool(env.get_wrapper_attr("is_task_success")())
            if success and first_success_step is None:
                first_success_step = step
            terminated_seen |= as_bool(terminated)
            truncated_seen |= as_bool(truncated)

        success = as_bool(env.get_wrapper_attr("is_task_success")())
        print(
            "[EXPERT PRESS DEBUG]",
            f"max_press_depth={max_depth}",
            "success_threshold=0.0048",
            f"first_success_step={first_success_step}",
            flush=True,
        )
        print(
            "[EXPERT RESULT]",
            f"success={success}",
            f"terminated_seen={terminated_seen}",
            f"truncated_seen={truncated_seen}",
            f"final_info={final_info}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
