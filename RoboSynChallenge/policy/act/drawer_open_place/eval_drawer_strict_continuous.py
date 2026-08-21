#!/usr/bin/env python
"""Run 20 strict drawer rollouts continuously in one deployment process.

Unlike checkpoint screening, this keeps one environment and one loaded model
alive across all resets.  It uses the same ordered completion criteria as
eval_drawer_strict_completion.py and saves per-step commands/joints per episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in reversed((REPO_ROOT, REPO_ROOT / "policy")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from policy.act.drawer_open_place.eval_drawer_strict_completion import (  # noqa: E402
    FIXED_EVAL_SEEDS,
    LEFT_GRIPPER_ID,
    RIGHT_GRIPPER_ID,
    add_vector,
    create_env,
    drawer_drop_target,
    drawer_state,
    duck_position,
    failure_label,
    load_model,
    max_abs,
    model_batch,
    scalar_bool,
    scale_action_for_env,
    seed_everything,
    sustained_start,
    tensor_values,
)
from scripts.eval_policy import EpisodeVideoRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=FIXED_EVAL_SEEDS)
    parser.add_argument("--setting", choices=("clear", "random"), default="random")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--reset-sync-steps", type=int, default=1)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--grasp-hold-steps", type=int, default=20)
    parser.add_argument("--inside-distance", type=float, default=0.14)
    parser.add_argument("--final-inside-distance", type=float, default=0.10)
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if len(args.seeds) != 20 or len(set(args.seeds)) != 20:
        raise ValueError("Continuous deployment stress test requires 20 unique seeds")
    for filename in ("config.json", "model.safetensors"):
        if not (args.checkpoint / filename).is_file():
            raise FileNotFoundError(args.checkpoint / filename)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    positive = (
        args.max_steps, args.grasp_hold_steps, args.inside_distance,
        args.final_inside_distance, args.release_hold_steps,
        args.object_stable_steps, args.object_stable_motion,
        args.drawer_open_distance, args.drawer_closed_tolerance,
        args.drawer_closed_hold_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Step counts, distances, and tolerances must be positive")
    if args.drawer_closed_tolerance >= args.drawer_open_distance:
        raise ValueError("Closed tolerance must be smaller than open distance")


def write_episode(output_dir: Path, records, summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run_episode(args, env, model, recorder, episode: int, seed: int) -> dict[str, object]:
    seed_everything(seed)
    obs, _ = env.reset(seed=seed)
    if args.reset_sync_steps:
        physics_dt = getattr(env.unwrapped.sim_cfg, "physics_dt", None)
        env.unwrapped.sim.update(physics_dt, args.reset_sync_steps)
        obs = env.unwrapped.get_obs()
    model.reset()
    if recorder:
        recorder.start_episode(episode, seed)
        recorder.record(obs)

    initial_duck = duck_position(obs)[0].detach().cpu()
    initial_drawer_qpos, _ = drawer_state(env)
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
        raw_left = float(raw_action[0, LEFT_GRIPPER_ID].item())
        actual_left = min(1.0, max(0.0, robot_qpos[LEFT_GRIPPER_ID]))
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
        if (
            first_inside_step is None and stable_grasp_step is not None
            and drawer_open_step is not None and target_distance <= args.inside_distance
        ):
            first_inside_step = step
        raw_release_run = (
            raw_release_run + 1
            if first_inside_step is not None
            and raw_left >= args.release_command_threshold else 0
        )
        if release_command_step is None:
            release_command_step = sustained_start(raw_release_run, step, args.release_hold_steps)
        actual_release_run = (
            actual_release_run + 1
            if release_command_step is not None
            and actual_left >= args.release_actual_threshold else 0
        )
        if release_actual_step is None:
            release_actual_step = sustained_start(actual_release_run, step, args.release_hold_steps)

        object_motion = None
        if release_actual_step is not None:
            object_window.append(duck.clone())
            if len(object_window) == args.object_stable_steps:
                positions = torch.stack(tuple(object_window))
                object_motion = float(
                    torch.linalg.vector_norm(positions - positions[0], dim=1).max().item()
                )
                if object_stable_step is None and object_motion <= args.object_stable_motion:
                    object_stable_step = step - args.object_stable_steps + 1
        else:
            object_window.clear()

        closed_now = (
            object_stable_step is not None and drawer_open_step is not None
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
            "episode": episode, "seed": seed, "step": step,
            "raw_left_release_command": raw_left,
            "actual_left_gripper_qpos": robot_qpos[LEFT_GRIPPER_ID],
            "actual_left_gripper_open_ratio": actual_left,
            "raw_right_gripper_command": float(raw_action[0, RIGHT_GRIPPER_ID].item()),
            "actual_right_gripper_qpos": robot_qpos[RIGHT_GRIPPER_ID],
            "duck_x": float(duck[0]), "duck_y": float(duck[1]), "duck_z": float(duck[2]),
            "drop_target_x": float(drop_target[0]), "drop_target_y": float(drop_target[1]),
            "drop_target_z": float(drop_target[2]),
            "duck_drop_target_distance": target_distance,
            "duck_drawer_xy_distance": distance_xy,
            "duck_lift": lift, "drawer_abs_travel": drawer_travel,
            "object_window_max_motion": object_motion,
            "strict_success": strict_success_step is not None,
            "official_success_signal_ignored": scalar_bool(info.get("success", False)),
            "reward": float(torch.as_tensor(reward).flatten()[0].item()),
            "terminated": scalar_bool(terminated), "truncated": scalar_bool(truncated),
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

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint), "episode": episode, "seed": seed,
        "steps": len(records), "end_reason": end_reason,
        "strict_success": strict_success_step is not None,
        "strict_success_step": strict_success_step,
        "stable_grasp_step": stable_grasp_step, "first_inside_step": first_inside_step,
        "release_command_step": release_command_step,
        "release_actual_step": release_actual_step,
        "object_stable_step": object_stable_step, "drawer_open_step": drawer_open_step,
        "drawer_closed_step": drawer_closed_step, "max_lift_m": max_lift,
        "max_abs_drawer_joint_travel": max_drawer_travel,
        "final_abs_drawer_joint_travel": records[-1]["drawer_abs_travel"],
        "final_duck_drawer_xy_distance": records[-1]["duck_drawer_xy_distance"],
        "failure_stage": None, "elapsed_s": time.perf_counter() - started,
    }
    if strict_success_step is None:
        summary["failure_stage"] = failure_label(summary)
    if recorder:
        recorder.close_episode(success=strict_success_step is not None)
    write_episode(args.output_dir / f"episode_{episode:02d}_seed_{seed}", records, summary)
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    env = create_env(args)
    model = load_model(args.checkpoint)
    recorder = None if args.no_video else EpisodeVideoRecorder(
        args.output_dir / "videos", obs_keys=args.video_camera, fps=10
    )
    rows: list[dict[str, object]] = []
    try:
        for episode, seed in enumerate(args.seeds):
            row = run_episode(args, env, model, recorder, episode, seed)
            rows.append(row)
            print(
                f"[STRICT CONTINUOUS] episode={episode:02d} seed={seed} "
                f"success={row['strict_success']} failure={row['failure_stage']}", flush=True
            )
        with (args.output_dir / "strict_continuous_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        successes = sum(bool(row["strict_success"]) for row in rows)
        failure_counts = {
            label: sum(row["failure_stage"] == label for row in rows)
            for label in sorted(
                {str(row["failure_stage"]) for row in rows if row["failure_stage"]}
            )
        }
        report = {
            "checkpoint": str(args.checkpoint),
            "execution_mode": "one model and one simulator process across 20 resets",
            "strict_metric": {
                "lift_threshold_m": args.lift_threshold,
                "grasp_hold_steps": args.grasp_hold_steps,
                "inside_drop_target_3d_distance_m": args.inside_distance,
                "release_hold_steps": args.release_hold_steps,
                "object_stable_steps": args.object_stable_steps,
                "object_stable_max_motion_m": args.object_stable_motion,
                "drawer_open_distance_m": args.drawer_open_distance,
                "drawer_closed_tolerance_m": args.drawer_closed_tolerance,
                "drawer_closed_hold_steps": args.drawer_closed_hold_steps,
                "official_near_drawer_success": "recorded but ignored",
            },
            "episodes": 20, "strict_successes": successes,
            "strict_success_rate": successes / 20,
            "failure_counts": failure_counts,
            "failed_seeds": [int(row["seed"]) for row in rows if not row["strict_success"]],
            "results": rows,
        }
        with (args.output_dir / "strict_continuous_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"[STRICT CONTINUOUS SUMMARY] success={successes}/20 ({successes / 20:.1%})", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
