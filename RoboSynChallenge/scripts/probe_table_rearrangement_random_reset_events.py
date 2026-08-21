#!/usr/bin/env python
"""Probe which random robot reset event opens the left gripper.

This is a read-only simulation experiment: source JSON is copied in memory and
selected reset events are removed only from that copy.
"""

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import eval_policy as base_eval
from policy.act_table_rearrangement_diag.config_patch import (
    patch_grasp_z,
    patch_random_arm_joint_ids,
)
from policy.act_table_rearrangement_diag import strict_task as _strict_task  # noqa: F401


VARIANT_EVENTS_TO_REMOVE = {
    "baseline": (),
    "no_eef_random": ("random_robot_init_eef_pose",),
    "no_qpos_random": ("random_robot_qpos",),
    "no_robot_random": ("random_robot_init_eef_pose", "random_robot_qpos"),
}


def gripper_state(env):
    robot = env.unwrapped.robot
    left_id = robot.get_joint_ids("left_eef", remove_mimic=True)[0]
    right_id = robot.get_joint_ids("right_eef", remove_mimic=True)[0]
    qpos = robot.get_qpos()[0]
    target_qpos = robot.get_qpos(target=True)[0]
    qvel = robot.get_qvel()[0]
    target_qvel = robot.get_qvel(target=True)[0]
    return {
        "left_joint_id": left_id,
        "right_joint_id": right_id,
        "left_qpos_m": float(qpos[left_id].item()),
        "right_qpos_m": float(qpos[right_id].item()),
        "left_target_qpos_m": float(target_qpos[left_id].item()),
        "right_target_qpos_m": float(target_qpos[right_id].item()),
        "left_qvel": float(qvel[left_id].item()),
        "right_qvel": float(qvel[right_id].item()),
        "left_target_qvel": float(target_qvel[left_id].item()),
        "right_target_qvel": float(target_qvel[right_id].item()),
    }


def xyz(value):
    return [float(item) for item in value[0].detach().cpu().tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANT_EVENTS_TO_REMOVE, required=True)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = {
        "task_name": "table_rearrangement",
        "setting": "random",
        "seed": args.rng_seed,
        "num_envs": 1,
        "device": "cpu",
        "headless": True,
        "renderer": "hybrid",
        "gpu_id": args.gpu_id,
        "filter_dataset_saving": True,
        "eval_freeze_interval_events": True,
        "max_steps": 361,
    }
    with open("configs/table_rearrangement/random/gym_config.json", "r") as handle:
        gym_config = deepcopy(json.load(handle))
    with open("configs/table_rearrangement/action_config.json", "r") as handle:
        action_config = deepcopy(json.load(handle))

    gym_config["id"] = "TableRearrangementDiagnostic"
    if not patch_random_arm_joint_ids(gym_config):
        raise RuntimeError("random_robot_qpos event was not found before ablation")
    events = gym_config["env"]["events"]
    for event_name in VARIANT_EVENTS_TO_REMOVE[args.variant]:
        if events.pop(event_name, None) is None:
            raise RuntimeError(f"Expected reset event was not found: {event_name}")
    if patch_grasp_z(action_config, -0.008) != 2:
        raise RuntimeError("Expected to patch two grasp-z entries")

    env, _ = base_eval.make_env_from_configs(config, gym_config, action_config)
    robot = env.unwrapped.robot
    runtime_groups = {
        name: robot.get_joint_ids(name)
        for name in ("left_arm", "right_arm", "left_eef", "right_eef")
    }
    configured_ids = events.get("random_robot_qpos", {}).get("params", {}).get(
        "joint_ids", []
    )
    expected_arm_ids = runtime_groups["left_arm"] + runtime_groups["right_arm"]
    if args.variant not in ("no_qpos_random", "no_robot_random"):
        eef_ids = runtime_groups["left_eef"] + runtime_groups["right_eef"]
        if set(configured_ids) != set(expected_arm_ids) or set(configured_ids) & set(
            eef_ids
        ):
            raise RuntimeError(
                f"qpos event IDs {configured_ids} do not match runtime arm IDs "
                f"{expected_arm_ids}, or overlap EEF IDs {eef_ids}"
            )
    print(
        f"runtime_joint_groups={runtime_groups} configured_qpos_ids={configured_ids}",
        flush=True,
    )
    rng = np.random.RandomState(args.rng_seed)
    rows = []
    try:
        for episode in range(args.episodes):
            seed = int(rng.randint(0, 2**31 - 1))
            env.reset(seed=seed)
            immediate = gripper_state(env)

            env.unwrapped.sim.update(env.unwrapped.sim_cfg.physics_dt, 1)
            after_sync = gripper_state(env)
            env.unwrapped.get_obs()
            after_obs = gripper_state(env)

            robot = env.unwrapped.robot
            left_link6 = robot.get_link_pose("left_link6", to_matrix=True)[:, :3, 3]
            right_link6 = robot.get_link_pose("right_link6", to_matrix=True)[:, :3, 3]
            row = {
                "variant": args.variant,
                "episode": episode + 1,
                "seed": seed,
                **{f"immediate_{key}": value for key, value in immediate.items()},
                **{f"after_sync_{key}": value for key, value in after_sync.items()},
                **{f"after_obs_{key}": value for key, value in after_obs.items()},
                "left_link6_x_m": xyz(left_link6)[0],
                "left_link6_y_m": xyz(left_link6)[1],
                "left_link6_z_m": xyz(left_link6)[2],
                "right_link6_x_m": xyz(right_link6)[0],
                "right_link6_y_m": xyz(right_link6)[1],
                "right_link6_z_m": xyz(right_link6)[2],
            }
            rows.append(row)
            print(
                f"{args.variant} episode={episode + 1} seed={seed} "
                f"immediate=({immediate['left_qpos_m']:.6f},"
                f"{immediate['right_qpos_m']:.6f}) "
                f"after_sync=({after_sync['left_qpos_m']:.6f},"
                f"{after_sync['right_qpos_m']:.6f})",
                flush=True,
            )
        output = args.output or Path(
            f"diagnostic_logs/random_reset_probe_{args.variant}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"probe_csv={output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
