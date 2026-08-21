#!/usr/bin/env python
"""Strict, process-isolated evaluation for the complete drawer workflow.

Success is ordered and cannot be triggered by merely moving the object near the
drawer: stable grasp -> object in drawer -> commanded and executed left-gripper
release -> object remains stable -> previously opened drawer returns closed.
Every rollout saves the raw policy action, scaled environment action, robot
qpos, drawer qpos/travel, and object/drawer positions for failure diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EMBODICHAIN_ROOT = Path(
    os.environ.get("EMBODICHAIN_ROOT", WORKSPACE_ROOT / "EmbodiChain")
)
for path in reversed((REPO_ROOT, REPO_ROOT / "policy", EMBODICHAIN_ROOT)):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import robosynchallenge  # noqa: F401,E402
from policy.act.deploy_policy import get_model  # noqa: E402
from scripts.eval_policy import (  # noqa: E402
    EpisodeVideoRecorder,
    make_env_from_configs,
)


LEFT_GRIPPER_ID = 6
RIGHT_GRIPPER_ID = 13
FIXED_EVAL_SEEDS = [
    209652396,
    398764591,
    924231285,
    1478610112,
    441365315,
    1537364731,
    192771779,
    1491434855,
    1819583497,
    530702035,
    626610453,
    1650906866,
    1879422756,
    1277901399,
    1682652230,
    243580376,
    1991416408,
    1171049868,
    1646868794,
    2051556033,
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def duck_position(obs) -> torch.Tensor:
    pose = torch.as_tensor(obs["duck_pose"], dtype=torch.float32)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected duck_pose [B,4,4], got {tuple(pose.shape)}")
    return pose[:, :3, 3]


def scale_action_for_env(env, raw_action: torch.Tensor) -> torch.Tensor:
    action = raw_action.clone()
    if action.ndim == 1:
        action = action.unsqueeze(0)
    low = torch.as_tensor(
        env.unwrapped.single_action_space.low,
        device=action.device,
        dtype=action.dtype,
    )
    high = torch.as_tensor(
        env.unwrapped.single_action_space.high,
        device=action.device,
        dtype=action.dtype,
    )
    ids = [LEFT_GRIPPER_ID, RIGHT_GRIPPER_ID]
    action[:, ids] = low[ids] + action[:, ids].clamp(0.0, 1.0) * (
        high[ids] - low[ids]
    )
    return action


def scalar_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def tensor_values(value) -> list[float]:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim > 1:
        tensor = tensor[0]
    return [float(item) for item in tensor.flatten().tolist()]


def add_vector(record: dict[str, object], prefix: str, values: list[float]) -> None:
    for index, value in enumerate(values):
        record[f"{prefix}_{index:02d}"] = value


def drawer_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    drawer = env.unwrapped.sim.get_articulation("drawer")
    qpos = torch.as_tensor(drawer.get_qpos(), dtype=torch.float32).detach().cpu()
    pose = torch.as_tensor(
        drawer.get_link_pose("outer_box", to_matrix=True), dtype=torch.float32
    ).detach().cpu()
    if qpos.ndim == 1:
        qpos = qpos.unsqueeze(0)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    return qpos[0], pose[0, :3, 3]


def create_env(args: argparse.Namespace):
    gym_path = REPO_ROOT / f"configs/drawer_open_place/{args.setting}/gym_config.json"
    action_path = REPO_ROOT / "configs/drawer_open_place/action_config.json"
    with gym_path.open(encoding="utf-8") as stream:
        gym_config = json.load(stream)
    with action_path.open(encoding="utf-8") as stream:
        action_config = json.load(stream)
    env_config = {
        "task_name": "drawer_open_place",
        "setting": args.setting,
        "num_envs": 1,
        "device": "cpu",
        "headless": args.headless,
        "renderer": "hybrid",
        "gpu_id": args.gpu_id,
        "arena_space": 5.0,
        "max_steps": args.max_steps,
        "filter_dataset_saving": True,
        "eval_freeze_interval_events": True,
    }
    env, _ = make_env_from_configs(env_config, gym_config, action_config)
    return env


def load_model(checkpoint: Path):
    return get_model(
        {
            "checkpoint_path": str(checkpoint),
            "device": "cuda",
            "act_step": 1,
            "n_action_steps": 10,
            "action_chunk_offset": 0,
            "state_obs_path": "robot/qpos",
            "strict_action_dim": True,
            "neutralize_qvel": False,
            "debug_button_press": False,
            "debug_trajectory": False,
        }
    )


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "outputs/act_drawer_strict_completion_ft_from_dual005_lr1e6_20260819/"
    "checkpoints/005000/pretrained_model"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/eval_drawer_strict_dual005_20260819"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", nargs="+", type=int, default=FIXED_EVAL_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--setting", choices=("clear", "random"), default="random")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--reset-sync-steps", type=int, default=1)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--grasp-hold-steps", type=int, default=20)
    parser.add_argument(
        "--inside-distance",
        type=float,
        default=0.14,
        help="3-D distance to the task's drawer drop target.",
    )
    parser.add_argument(
        "--final-inside-distance",
        type=float,
        default=0.10,
        help="Final XY distance to outer_box after drawer closure.",
    )
    parser.add_argument("--release-command-threshold", type=float, default=0.90)
    parser.add_argument("--release-actual-threshold", type=float, default=0.90)
    parser.add_argument("--release-hold-steps", type=int, default=5)
    parser.add_argument("--object-stable-steps", type=int, default=20)
    parser.add_argument("--object-stable-motion", type=float, default=0.02)
    parser.add_argument("--drawer-open-distance", type=float, default=0.08)
    parser.add_argument("--drawer-closed-tolerance", type=float, default=0.02)
    parser.add_argument("--drawer-closed-hold-steps", type=int, default=5)
    parser.add_argument("--video-camera", default="cam_high")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be a non-empty unique list")
    for filename in ("config.json", "model.safetensors"):
        if not (args.checkpoint / filename).is_file():
            raise FileNotFoundError(args.checkpoint / filename)
    positive = (
        args.max_steps,
        args.grasp_hold_steps,
        args.inside_distance,
        args.final_inside_distance,
        args.release_hold_steps,
        args.object_stable_steps,
        args.object_stable_motion,
        args.drawer_open_distance,
        args.drawer_closed_tolerance,
        args.drawer_closed_hold_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("All step counts, distances, and tolerances must be positive")
    if args.drawer_closed_tolerance >= args.drawer_open_distance:
        raise ValueError("Closed tolerance must be smaller than open distance")
    for value in (args.release_command_threshold, args.release_actual_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Release thresholds must be in [0, 1]")
    if args.worker_seed is not None and args.worker_seed not in args.seeds:
        raise ValueError("worker-seed must be included in seeds")


def sustained_start(run_length: int, step: int, required: int) -> int | None:
    return step - required + 1 if run_length == required else None


def max_abs(values: torch.Tensor) -> float:
    return float(torch.abs(values).max().item()) if values.numel() else 0.0


def drawer_drop_target(env) -> torch.Tensor:
    """Reconstruct the expert's intrinsic drawer drop target.

    action_config.json defines ``left_arm_place_drop_pose`` from drawer_pose by
    intrinsic +0.18 m x and +0.14 m z offsets.  This stays valid while the
    sliding drawer is open, unlike distance to the fixed outer_box center.
    """

    drawer = env.unwrapped.sim.get_articulation("drawer")
    pose = torch.as_tensor(
        drawer.get_local_pose(to_matrix=True), dtype=torch.float32
    ).detach().cpu()
    if pose.ndim == 3:
        pose = pose[0]
    if pose.shape != (4, 4):
        raise ValueError(f"Expected drawer_pose [4,4], got {tuple(pose.shape)}")
    intrinsic_offset = torch.tensor([0.18, 0.0, 0.14], dtype=pose.dtype)
    return pose[:3, :3] @ intrinsic_offset + pose[:3, 3]


def failure_label(summary: dict[str, object]) -> str:
    if summary["stable_grasp_step"] is None:
        return "grasp_failed"
    if summary["first_inside_step"] is None:
        return "grasped_but_not_placed"
    if summary["release_command_step"] is None:
        return "placed_but_no_release_command"
    if summary["release_actual_step"] is None:
        return "release_command_not_executed"
    if summary["object_stable_step"] is None:
        return "released_but_object_not_stable_in_drawer"
    if summary["drawer_open_step"] is None:
        return "drawer_never_opened"
    return "object_stable_but_drawer_not_closed"


def write_worker_artifacts(
    output_dir: Path, records: list[dict[str, object]], summary: dict[str, object]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def evaluate_worker(args: argparse.Namespace) -> None:
    seed = int(args.worker_seed)
    if (args.output_dir / "trace.csv").exists() or (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite results in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    seed_everything(seed)
    env = create_env(args)
    model = load_model(args.checkpoint)
    recorder = None if args.no_video else EpisodeVideoRecorder(
        args.output_dir / "videos", obs_keys=args.video_camera, fps=10
    )

    records: list[dict[str, object]] = []
    stable_grasp_step = first_inside_step = None
    release_command_step = release_actual_step = object_stable_step = None
    drawer_open_step = drawer_closed_step = strict_success_step = None
    grasp_run = raw_release_run = actual_release_run = closed_run = 0
    object_window: deque[torch.Tensor] = deque(maxlen=args.object_stable_steps)
    max_lift = float("-inf")
    max_drawer_travel = 0.0
    end_reason = "max_steps"
    started = time.perf_counter()

    try:
        seed_everything(seed)
        obs, _ = env.reset(seed=seed)
        if args.reset_sync_steps:
            physics_dt = getattr(env.unwrapped.sim_cfg, "physics_dt", None)
            env.unwrapped.sim.update(physics_dt, args.reset_sync_steps)
            obs = env.unwrapped.get_obs()
        model.reset()
        if recorder:
            recorder.start_episode(0, seed)
            recorder.record(obs)

        initial_duck = duck_position(obs)[0].detach().cpu()
        initial_drawer_qpos, _ = drawer_state(env)

        for step in range(args.max_steps):
            raw_action = model.select_action(model_batch(model, obs))
            if raw_action.ndim == 1:
                raw_action = raw_action.unsqueeze(0)
            env_action = scale_action_for_env(env, raw_action)
            next_obs, reward, terminated, truncated, info = env.step(
                env_action.detach().to(env.unwrapped.device, dtype=torch.float32)
            )
            if recorder:
                recorder.record(next_obs)

            duck = duck_position(next_obs)[0].detach().cpu()
            drawer_qpos, drawer_xyz = drawer_state(env)
            drawer_delta = drawer_qpos - initial_drawer_qpos
            drawer_travel = max_abs(drawer_delta)
            max_drawer_travel = max(max_drawer_travel, drawer_travel)
            robot_qpos = tensor_values(next_obs["robot"]["qpos"])
            raw_left_release = float(raw_action[0, LEFT_GRIPPER_ID].item())
            actual_left_open = min(1.0, max(0.0, robot_qpos[LEFT_GRIPPER_ID]))
            drop_target = drawer_drop_target(env)
            target_distance = float(torch.linalg.vector_norm(duck - drop_target))
            distance_xy = float(torch.linalg.vector_norm(duck[:2] - drawer_xyz[:2]))
            lift = float(duck[2] - initial_duck[2])
            max_lift = max(max_lift, lift)
            if drawer_open_step is None and drawer_travel >= args.drawer_open_distance:
                drawer_open_step = step

            grasp_run = grasp_run + 1 if lift >= args.lift_threshold else 0
            if stable_grasp_step is None:
                stable_grasp_step = sustained_start(grasp_run, step, args.grasp_hold_steps)

            ordered_inside = (
                stable_grasp_step is not None
                and drawer_open_step is not None
                and target_distance <= args.inside_distance
            )
            if first_inside_step is None and ordered_inside:
                first_inside_step = step

            raw_release_run = (
                raw_release_run + 1
                if first_inside_step is not None
                and raw_left_release >= args.release_command_threshold
                else 0
            )
            if release_command_step is None:
                release_command_step = sustained_start(
                    raw_release_run, step, args.release_hold_steps
                )

            actual_release_run = (
                actual_release_run + 1
                if release_command_step is not None
                and actual_left_open >= args.release_actual_threshold
                else 0
            )
            if release_actual_step is None:
                release_actual_step = sustained_start(
                    actual_release_run, step, args.release_hold_steps
                )

            object_motion = None
            if release_actual_step is not None:
                object_window.append(duck.clone())
                if len(object_window) == args.object_stable_steps:
                    stacked = torch.stack(tuple(object_window))
                    object_motion = float(
                        torch.linalg.vector_norm(stacked - stacked[0], dim=1).max().item()
                    )
                    if object_stable_step is None and object_motion <= args.object_stable_motion:
                        object_stable_step = step - args.object_stable_steps + 1
            else:
                object_window.clear()

            closed_now = (
                object_stable_step is not None
                and drawer_open_step is not None
                and drawer_travel <= args.drawer_closed_tolerance
                and distance_xy <= args.final_inside_distance
            )
            closed_run = closed_run + 1 if closed_now else 0
            if drawer_closed_step is None:
                drawer_closed_step = sustained_start(
                    closed_run, step, args.drawer_closed_hold_steps
                )
            if drawer_closed_step is not None:
                strict_success_step = step

            record: dict[str, object] = {
                "seed": seed,
                "step": step,
                "raw_left_release_command": raw_left_release,
                "actual_left_gripper_qpos": robot_qpos[LEFT_GRIPPER_ID],
                "actual_left_gripper_open_ratio": actual_left_open,
                "raw_right_gripper_command": float(raw_action[0, RIGHT_GRIPPER_ID].item()),
                "actual_right_gripper_qpos": robot_qpos[RIGHT_GRIPPER_ID],
                "duck_x": float(duck[0]), "duck_y": float(duck[1]), "duck_z": float(duck[2]),
                "drawer_x": float(drawer_xyz[0]), "drawer_y": float(drawer_xyz[1]),
                "drawer_z": float(drawer_xyz[2]),
                "drop_target_x": float(drop_target[0]),
                "drop_target_y": float(drop_target[1]),
                "drop_target_z": float(drop_target[2]),
                "duck_drop_target_distance": target_distance,
                "duck_drawer_xy_distance": distance_xy,
                "duck_lift": lift,
                "drawer_abs_travel": drawer_travel,
                "object_window_max_motion": object_motion,
                "stable_grasp_reached": stable_grasp_step is not None,
                "inside_reached": first_inside_step is not None,
                "release_command_reached": release_command_step is not None,
                "release_actual_reached": release_actual_step is not None,
                "object_stable_reached": object_stable_step is not None,
                "drawer_open_reached": drawer_open_step is not None,
                "drawer_closed_reached": drawer_closed_step is not None,
                "strict_success": strict_success_step is not None,
                "official_success_signal_ignored": scalar_bool(info.get("success", False)),
                "reward": float(torch.as_tensor(reward).flatten()[0].item()),
                "terminated": scalar_bool(terminated),
                "truncated": scalar_bool(truncated),
            }
            add_vector(record, "raw_action", tensor_values(raw_action))
            add_vector(record, "env_action", tensor_values(env_action))
            add_vector(record, "robot_qpos", robot_qpos)
            add_vector(record, "raw_right_arm_close_command", tensor_values(raw_action)[7:13])
            add_vector(record, "actual_right_arm_qpos", robot_qpos[7:13])
            add_vector(record, "drawer_qpos", tensor_values(drawer_qpos))
            add_vector(record, "drawer_delta", tensor_values(drawer_delta))
            records.append(record)
            obs = next_obs

            if strict_success_step is not None:
                end_reason = "strict_success"
                break
            if scalar_bool(terminated):
                end_reason = "terminated"
                break
            if scalar_bool(truncated):
                end_reason = "truncated"
                break

        if not records:
            raise RuntimeError("Rollout produced no trace records")
        summary: dict[str, object] = {
            "checkpoint": str(args.checkpoint),
            "setting": args.setting,
            "seed": seed,
            "steps": len(records),
            "end_reason": end_reason,
            "strict_success": strict_success_step is not None,
            "strict_success_step": strict_success_step,
            "stable_grasp_step": stable_grasp_step,
            "first_inside_step": first_inside_step,
            "release_command_step": release_command_step,
            "release_actual_step": release_actual_step,
            "object_stable_step": object_stable_step,
            "drawer_open_step": drawer_open_step,
            "drawer_closed_step": drawer_closed_step,
            "max_lift_m": max_lift,
            "max_abs_drawer_joint_travel": max_drawer_travel,
            "final_abs_drawer_joint_travel": records[-1]["drawer_abs_travel"],
            "final_duck_drawer_xy_distance": records[-1]["duck_drawer_xy_distance"],
            "final_duck_drop_target_distance": records[-1]["duck_drop_target_distance"],
            "failure_stage": None,
            "elapsed_s": time.perf_counter() - started,
        }
        if strict_success_step is None:
            summary["failure_stage"] = failure_label(summary)
        if recorder:
            recorder.close_episode(success=strict_success_step is not None)
        write_worker_artifacts(args.output_dir, records, summary)
        print(
            f"[STRICT] seed={seed} success={summary['strict_success']} "
            f"failure={summary['failure_stage']} grasp={stable_grasp_step} "
            f"release={release_actual_step} stable={object_stable_step} "
            f"closed={drawer_closed_step}", flush=True
        )
    finally:
        env.close()


def worker_complete(output_dir: Path) -> bool:
    try:
        with (output_dir / "summary.json").open(encoding="utf-8") as stream:
            json.load(stream)
        with (output_dir / "trace.csv").open(newline="", encoding="utf-8") as stream:
            return next(csv.DictReader(stream), None) is not None
    except (OSError, ValueError, csv.Error):
        return False


def worker_command(args: argparse.Namespace, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--checkpoint", str(args.checkpoint), "--seeds", str(seed),
        "--worker-seed", str(seed), "--output-dir", str(output_dir),
        "--setting", args.setting, "--gpu-id", str(args.gpu_id),
        "--headless" if args.headless else "--no-headless",
        "--max-steps", str(args.max_steps),
        "--reset-sync-steps", str(args.reset_sync_steps),
        "--lift-threshold", str(args.lift_threshold),
        "--grasp-hold-steps", str(args.grasp_hold_steps),
        "--inside-distance", str(args.inside_distance),
        "--final-inside-distance", str(args.final_inside_distance),
        "--release-command-threshold", str(args.release_command_threshold),
        "--release-actual-threshold", str(args.release_actual_threshold),
        "--release-hold-steps", str(args.release_hold_steps),
        "--object-stable-steps", str(args.object_stable_steps),
        "--object-stable-motion", str(args.object_stable_motion),
        "--drawer-open-distance", str(args.drawer_open_distance),
        "--drawer-closed-tolerance", str(args.drawer_closed_tolerance),
        "--drawer-closed-hold-steps", str(args.drawer_closed_hold_steps),
        "--video-camera", args.video_camera,
    ]
    if args.no_video:
        command.append("--no-video")
    return command


def evaluate_parent(args: argparse.Namespace) -> None:
    final_json = args.output_dir / "strict_completion_summary.json"
    final_csv = args.output_dir / "strict_completion_results.csv"
    if final_json.exists() or final_csv.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate results in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for episode, seed in enumerate(args.seeds):
        child = args.output_dir / f"seed_{seed}"
        if not worker_complete(child):
            command = worker_command(args, seed, child)
            logs: list[str] = []
            for attempt in range(1, 4):
                print(f"[STRICT FRESH] episode={episode:02d} seed={seed} attempt={attempt}/3", flush=True)
                completed = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace"
                )
                logs.append(
                    f"===== attempt {attempt} returncode={completed.returncode} =====\n{completed.stdout}"
                )
                child.mkdir(parents=True, exist_ok=True)
                (child / "worker.log").write_text("\n".join(logs), encoding="utf-8")
                if worker_complete(child):
                    break
                print(f"[STRICT RETRY] seed={seed} returncode={completed.returncode}", flush=True)
                time.sleep(5)
            if not worker_complete(child):
                raise RuntimeError(f"Seed {seed} failed after 3 simulator attempts")
            time.sleep(2)
        with (child / "summary.json").open(encoding="utf-8") as stream:
            row = json.load(stream)
        row["episode"] = episode
        rows.append(row)
        print(
            f"[STRICT COLLECT] seed={seed} success={row['strict_success']} "
            f"failure={row['failure_stage']}", flush=True
        )

    with final_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(bool(row["strict_success"]) for row in rows)
    failure_counts = {
        label: sum(row["failure_stage"] == label for row in rows)
        for label in sorted({str(row["failure_stage"]) for row in rows if row["failure_stage"]})
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "metric": {
            "ordered_stages": [
                "stable_grasp", "inside_drawer", "left_release_command_and_actual",
                "object_stable", "drawer_was_opened_then_closed",
            ],
            "max_steps": args.max_steps,
            "lift_threshold_m": args.lift_threshold,
            "grasp_hold_steps": args.grasp_hold_steps,
            "inside_drop_target_3d_distance_m": args.inside_distance,
            "final_outer_box_xy_distance_m": args.final_inside_distance,
            "release_command_threshold": args.release_command_threshold,
            "release_actual_threshold": args.release_actual_threshold,
            "release_hold_steps": args.release_hold_steps,
            "object_stable_steps": args.object_stable_steps,
            "object_stable_max_motion_m": args.object_stable_motion,
            "drawer_open_distance_m": args.drawer_open_distance,
            "drawer_closed_tolerance_m": args.drawer_closed_tolerance,
            "drawer_closed_hold_steps": args.drawer_closed_hold_steps,
            "official_near_drawer_success": "recorded but ignored",
            "process_isolation": "one fresh simulator process per seed",
            "seeds": args.seeds,
        },
        "episodes": len(rows),
        "strict_successes": successes,
        "strict_success_rate": successes / len(rows),
        "failure_counts": failure_counts,
        "failed_seeds": [int(row["seed"]) for row in rows if not row["strict_success"]],
        "results": rows,
    }
    with final_json.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"[STRICT SUMMARY] success={successes}/{len(rows)} ({successes / len(rows):.1%})", flush=True)
    print(f"[STRICT SUMMARY] failures={failure_counts}", flush=True)
    print(f"[STRICT OUTPUT] summary={final_json}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.worker_seed is None:
        evaluate_parent(args)
    else:
        evaluate_worker(args)


if __name__ == "__main__":
    main()
