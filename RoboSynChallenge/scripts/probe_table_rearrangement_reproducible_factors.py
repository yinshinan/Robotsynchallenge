#!/usr/bin/env python
"""Collect reproducible reset factors without loading a policy checkpoint."""

import argparse
import csv
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np

import eval_policy as base_eval
from policy.act_table_rearrangement_diag import strict_task as _strict_task  # noqa: F401
from policy.act_table_rearrangement_diag.config_patch import (
    patch_grasp_z,
    patch_random_arm_joint_ids,
)


def _first(value):
    return value[0].detach().cpu().tolist()


def _yaw_degrees(pose_xyz_qwxyz):
    _, _, _, qw, qx, qy, qz = pose_xyz_qwxyz
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny, cosy))


def _hash_image(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _active_distractors(env):
    sim = env.unwrapped.sim
    active = []
    for uid in sorted(sim.get_rigid_object_uid_list()):
        if not uid.startswith("distractor_"):
            continue
        pose = _first(sim.get_rigid_object(uid).get_local_pose(to_matrix=False))
        if pose[2] > -1.0:
            active.append({"uid": uid, "pose_xyz_qwxyz": pose})
    return active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_logs/reproducible_reset_factors_20.csv"),
    )
    args = parser.parse_args()

    config = {
        "task_name": "table_rearrangement",
        "setting": "random",
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
        raise RuntimeError("random_robot_qpos event was not found")
    if patch_grasp_z(action_config, -0.008) != 2:
        raise RuntimeError("Expected two grasp-z entries")

    events = gym_config["env"]["events"]
    distractor = events.pop("randomize_distractor_slots")
    events["randomize_distractor_slots"] = distractor

    env, _ = base_eval.make_env_from_configs(config, gym_config, action_config)
    episode_rng = np.random.RandomState(args.rng_seed)
    rows = []
    try:
        for episode in range(args.episodes):
            seed = int(episode_rng.randint(0, 2**31 - 1))
            random.seed(seed)
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            env.unwrapped.sim.update(env.unwrapped.sim_cfg.physics_dt, 1)
            obs = env.unwrapped.get_obs()

            sim = env.unwrapped.sim
            robot = env.unwrapped.robot
            fork_pose = _first(
                sim.get_rigid_object("fork").get_local_pose(to_matrix=False)
            )
            spoon_pose = _first(
                sim.get_rigid_object("spoon").get_local_pose(to_matrix=False)
            )
            cam_high = sim.get_sensor("cam_high")
            intrinsics = _first(cam_high.get_intrinsics())
            camera_pose = _first(cam_high.get_arena_pose(to_matrix=False))
            row = {
                "episode": episode + 1,
                "trace_episode": episode,
                "seed": seed,
                "fork_x_m": fork_pose[0],
                "fork_y_m": fork_pose[1],
                "fork_yaw_deg": _yaw_degrees(fork_pose),
                "spoon_x_m": spoon_pose[0],
                "spoon_y_m": spoon_pose[1],
                "spoon_yaw_deg": _yaw_degrees(spoon_pose),
                "cam_high_fx": intrinsics[0][0],
                "cam_high_fy": intrinsics[1][1],
                "cam_high_cx": intrinsics[0][2],
                "cam_high_cy": intrinsics[1][2],
                "cam_high_pose_json": json.dumps(camera_pose),
                "robot_qpos_json": json.dumps(_first(robot.get_qpos())),
                "robot_qvel_json": json.dumps(_first(robot.get_qvel())),
                "active_distractors_json": json.dumps(_active_distractors(env)),
                "cam_high_sha256": _hash_image(obs["sensor"]["cam_high"]["color"]),
                "cam_right_wrist_sha256": _hash_image(
                    obs["sensor"]["cam_right_wrist"]["color"]
                ),
            }
            rows.append(row)
            print(
                f"episode={episode + 1} seed={seed} "
                f"spoon=({spoon_pose[0]:.6f},{spoon_pose[1]:.6f}) "
                f"yaw={row['spoon_yaw_deg']:.2f} "
                f"fx={row['cam_high_fx']:.3f} fy={row['cam_high_fy']:.3f}",
                flush=True,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"probe_csv={args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
