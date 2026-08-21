#!/usr/bin/env python
"""Summarize JSONL traces produced by eval_policy_sample_loading_trace.py."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def first_step(rows: list[dict[str, Any]], predicate: Any) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["step"])
    return None


def summarize_episode(
    path: Path, n_action_steps: int, episode: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reset = rows[0]
    steps = [row for row in rows if row["event"] == "step"]
    initial_z = float(reset["cube_position"][2])
    initial_xy = np.asarray(reset["cube_position"][:2], dtype=float)
    z_values = np.asarray([row["cube_position"][2] for row in steps])
    xy_values = np.asarray([row["cube_position"][:2] for row in steps])
    rack_distances = np.asarray([row["cube_to_rack_xy"] for row in steps])
    actions = np.asarray([row["action"] for row in steps], dtype=float)
    qpos = np.asarray([row["qpos"] for row in steps], dtype=float)

    lift_step = first_step(
        steps, lambda row: float(row["cube_position"][2]) >= initial_z + 0.03
    )
    drop_step = None
    if lift_step is not None:
        drop_step = first_step(
            [row for row in steps if int(row["step"]) > lift_step],
            lambda row: float(row["cube_position"][2]) <= initial_z + 0.015,
        )

    action_jumps = np.mean(np.abs(np.diff(actions, axis=0)), axis=1)
    replan_mask = np.asarray([bool(row["replan_step"]) for row in steps[1:]])
    nonreplan_mask = ~replan_mask
    replan_jump = float(np.mean(action_jumps[replan_mask])) if replan_mask.any() else None
    nonreplan_jump = (
        float(np.mean(action_jumps[nonreplan_mask])) if nonreplan_mask.any() else None
    )

    return {
        "trace": path.name,
        "n_action_steps": n_action_steps,
        "episode": episode,
        "steps": len(steps),
        "initial_cube_x": float(initial_xy[0]),
        "initial_cube_y": float(initial_xy[1]),
        "initial_cube_z": initial_z,
        "max_cube_z": float(z_values.max()),
        "max_cube_z_step": int(steps[int(z_values.argmax())]["step"]),
        "final_cube_z": float(z_values[-1]),
        "min_cube_to_rack_xy": float(rack_distances.min()),
        "min_cube_to_rack_xy_step": int(steps[int(rack_distances.argmin())]["step"]),
        "final_cube_to_rack_xy": float(rack_distances[-1]),
        "cube_xy_displacement": float(np.linalg.norm(xy_values[-1] - initial_xy)),
        "lift_step": lift_step,
        "drop_step": drop_step,
        "right_close_command_step": first_step(
            steps[40:], lambda row: float(row["right_gripper_action"]) < 0.015
        ),
        "right_closed_state_step": first_step(
            steps[40:], lambda row: float(row["right_gripper_qpos"]) < 0.30
        ),
        "right_reopen_after_lift_step": None
        if lift_step is None
        else first_step(
            [row for row in steps if int(row["step"]) > lift_step],
            lambda row: float(row["right_gripper_action"]) > 0.04,
        ),
        "left_close_after_lift_step": None
        if lift_step is None
        else first_step(
            [row for row in steps if int(row["step"]) > lift_step],
            lambda row: float(row["left_gripper_action"]) < 0.015,
        ),
        "left_closed_state_after_lift_step": None
        if lift_step is None
        else first_step(
            [row for row in steps if int(row["step"]) > lift_step],
            lambda row: float(row["left_gripper_qpos"]) < 0.30,
        ),
        "replan_count": sum(bool(row["replan_step"]) for row in steps),
        "mean_action_jump": float(action_jumps.mean()),
        "mean_replan_action_jump": replan_jump,
        "mean_nonreplan_action_jump": nonreplan_jump,
        "arm_qpos_path_length": float(
            np.sum(np.linalg.norm(np.diff(qpos[:, [*range(6), *range(7, 13)]], axis=0), axis=1))
        ),
        "mean_left_tracking_error": mean(
            float(row["left_arm_tracking_error"]) for row in steps
        ),
        "mean_right_tracking_error": mean(
            float(row["right_arm_tracking_error"]) for row in steps
        ),
    }


def main() -> int:
    args = parse_args()
    output_prefix = args.output_prefix or args.trace_dir / "frame_trace_summary"
    summaries: list[dict[str, Any]] = []
    for path in sorted(args.trace_dir.glob("matrix_seed0_first3_n*.jsonl")):
        n_action_steps = int(path.stem.rsplit("n", 1)[1])
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as trace_file:
            for line in trace_file:
                row = json.loads(line)
                if row["event"] != "trace_error":
                    by_episode[int(row["episode"])].append(row)
        for episode, rows in sorted(by_episode.items()):
            summaries.append(summarize_episode(path, n_action_steps, episode, rows))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# sample_loading frame trace summary",
        "",
        "| n | ep | init xy | max z@step | min rack xy@step | lift | drop | right close | left close | right reopen | replans | replan jump | nonreplan jump |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: (item["n_action_steps"], item["episode"])):
        lines.append(
            f"| {row['n_action_steps']} | {row['episode']} | "
            f"({row['initial_cube_x']:.3f},{row['initial_cube_y']:.3f}) | "
            f"{row['max_cube_z']:.3f}@{row['max_cube_z_step']} | "
            f"{row['min_cube_to_rack_xy']:.3f}@{row['min_cube_to_rack_xy_step']} | "
            f"{row['lift_step']} | {row['drop_step']} | "
            f"{row['right_close_command_step']} | {row['left_close_after_lift_step']} | "
            f"{row['right_reopen_after_lift_step']} | {row['replan_count']} | "
            f"{row['mean_replan_action_jump']!s} | {row['mean_nonreplan_action_jump']!s} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
