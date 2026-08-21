#!/usr/bin/env python
"""Produce per-seed and aggregate reports from multi-episode diagnostic traces."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


OPEN_THRESHOLD_M = 0.010
MIN_LIFT_M = 0.020
XY_TOLERANCE_M = 0.035
Z_TOLERANCE_M = 0.035
MIN_TARGET_STREAK_STEPS = 5
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPISODE_RESULT = re.compile(r"Episode\s+(\d+)/\d+:\s+(SUCCESS|FAIL)")


def load_episodes(path):
    episodes = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episodes[int(row["episode"])].append(row)
    if not episodes:
        raise ValueError(f"Trace is empty: {path}")
    return dict(sorted(episodes.items()))


def episode_seeds(seed, count):
    rng = np.random.RandomState(seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(count)]


def load_console_results(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = ANSI_ESCAPE.sub("", text)
    results = {}
    for match in EPISODE_RESULT.finditer(text):
        results[int(match.group(1)) - 1] = match.group(2) == "SUCCESS"
    if not results:
        raise ValueError(f"No per-episode results found in console log: {path}")
    return results


def target_ok(row):
    fork = row["fork_xyz_m"]
    spoon = row["spoon_xyz_m"]
    plate = row["plate_xyz_m"]
    fork_xy_error = float(
        np.linalg.norm(np.asarray(fork[:2]) - np.asarray([plate[0], plate[1] + 0.16]))
    )
    spoon_xy_error = float(
        np.linalg.norm(np.asarray(spoon[:2]) - np.asarray([plate[0], plate[1] - 0.16]))
    )
    z_ok = (
        abs(fork[2] - plate[2]) <= Z_TOLERANCE_M
        and abs(spoon[2] - plate[2]) <= Z_TOLERANCE_M
    )
    return (
        fork_xy_error <= XY_TOLERANCE_M
        and spoon_xy_error <= XY_TOLERANCE_M
        and z_ok
    ), fork_xy_error, spoon_xy_error


def minimum_alias(rows, *keys):
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if values:
            return min(values)
    return float("nan")


def summarize_episode(setting, episode, seed, rows, evaluator_success):
    initial_open_left = rows[0]["physical_gripper_qpos_m"][0]
    initial_open_right = rows[0]["physical_gripper_qpos_m"][1]
    max_open_left = max(row["physical_gripper_qpos_m"][0] for row in rows)
    max_open_right = max(row["physical_gripper_qpos_m"][1] for row in rows)
    max_fork_lift = max(row["fork_lift_m"] for row in rows)
    max_spoon_lift = max(row["spoon_lift_m"] for row in rows)
    min_left_wrist_distance = minimum_alias(
        rows, "left_link6_to_fork_m", "left_tcp_to_fork_m"
    )
    min_right_wrist_distance = minimum_alias(
        rows, "right_link6_to_spoon_m", "right_tcp_to_spoon_m"
    )

    target_streak = 0
    max_target_streak = 0
    reached_target = False
    final_fork_xy_error = float("nan")
    final_spoon_xy_error = float("nan")
    for row in rows:
        placement_ok, final_fork_xy_error, final_spoon_xy_error = target_ok(row)
        reached_target = reached_target or placement_ok
        target_streak = target_streak + 1 if placement_ok else 0
        max_target_streak = max(max_target_streak, target_streak)

    both_lifted = max_fork_lift >= MIN_LIFT_M and max_spoon_lift >= MIN_LIFT_M
    trace_success = both_lifted and max_target_streak >= MIN_TARGET_STREAK_STEPS
    success = bool(evaluator_success)
    both_opened = max_open_left >= OPEN_THRESHOLD_M and max_open_right >= OPEN_THRESHOLD_M

    if success:
        failure_stage = "success"
    elif trace_success:
        failure_stage = "trace_evaluator_mismatch"
    elif not both_opened:
        failure_stage = "gripper_not_opened"
    elif max_fork_lift < MIN_LIFT_M and max_spoon_lift < MIN_LIFT_M:
        failure_stage = "open_but_no_effective_grasp"
    elif max_fork_lift < MIN_LIFT_M:
        failure_stage = "fork_not_lifted"
    elif max_spoon_lift < MIN_LIFT_M:
        failure_stage = "spoon_not_lifted"
    elif reached_target:
        failure_stage = "target_unstable_or_dropped"
    else:
        failure_stage = "both_lifted_but_not_placed"

    return {
        "setting": setting,
        "episode": episode + 1,
        "trace_episode": episode,
        "seed": seed,
        "success": success,
        "trace_success": trace_success,
        "failure_stage": failure_stage,
        "steps_logged": len(rows),
        "initial_open_left_m": initial_open_left,
        "initial_open_right_m": initial_open_right,
        "max_open_left_m": max_open_left,
        "max_open_right_m": max_open_right,
        "max_fork_lift_m": max_fork_lift,
        "max_spoon_lift_m": max_spoon_lift,
        "min_left_link6_to_fork_m": min_left_wrist_distance,
        "min_right_link6_to_spoon_m": min_right_wrist_distance,
        "max_target_streak": max_target_streak,
        "final_fork_target_xy_error_m": final_fork_xy_error,
        "final_spoon_target_xy_error_m": final_spoon_xy_error,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-trace", type=Path, required=True)
    parser.add_argument("--random-trace", type=Path, required=True)
    parser.add_argument("--clear-console", type=Path, required=True)
    parser.add_argument("--random-console", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_logs/table_rearrangement_multiseed.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("diagnostic_logs/table_rearrangement_multiseed_summary.csv"),
    )
    args = parser.parse_args()

    seeds = episode_seeds(args.rng_seed, args.episodes)
    results = []
    for setting, trace_path, console_path in (
        ("clear", args.clear_trace, args.clear_console),
        ("random", args.random_trace, args.random_console),
    ):
        trace_episodes = load_episodes(trace_path)
        evaluator_results = load_console_results(console_path)
        if len(trace_episodes) != args.episodes:
            raise ValueError(
                f"Expected {args.episodes} episodes in {trace_path}, "
                f"found {len(trace_episodes)}"
            )
        if len(evaluator_results) != args.episodes:
            raise ValueError(
                f"Expected {args.episodes} evaluator results in {console_path}, "
                f"found {len(evaluator_results)}"
            )
        for episode, rows in trace_episodes.items():
            if episode < 0 or episode >= len(seeds):
                raise ValueError(f"Unexpected episode index {episode} in {trace_path}")
            results.append(
                summarize_episode(
                    setting,
                    episode,
                    seeds[episode],
                    rows,
                    evaluator_results[episode],
                )
            )

    write_csv(args.output, results)

    summary_rows = []
    for setting in ("clear", "random"):
        selected = [row for row in results if row["setting"] == setting]
        failures = Counter(row["failure_stage"] for row in selected)
        successes = sum(bool(row["success"]) for row in selected)
        initial_left_open = [
            row for row in selected if row["initial_open_left_m"] >= OPEN_THRESHOLD_M
        ]
        summary_rows.append(
            {
                "setting": setting,
                "episodes": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "initial_left_open_episodes": len(initial_left_open),
                "failures_with_initial_left_open": sum(
                    not bool(row["success"]) for row in initial_left_open
                ),
                "gripper_not_opened": failures["gripper_not_opened"],
                "open_but_no_effective_grasp": failures[
                    "open_but_no_effective_grasp"
                ],
                "fork_not_lifted": failures["fork_not_lifted"],
                "spoon_not_lifted": failures["spoon_not_lifted"],
                "both_lifted_but_not_placed": failures[
                    "both_lifted_but_not_placed"
                ],
                "target_unstable_or_dropped": failures[
                    "target_unstable_or_dropped"
                ],
                "trace_evaluator_mismatch": failures["trace_evaluator_mismatch"],
            }
        )
    write_csv(args.summary_output, summary_rows)

    for row in summary_rows:
        print(
            f"{row['setting']}: {row['successes']}/{row['episodes']} "
            f"({100.0 * row['success_rate']:.1f}%)"
        )
        print(
            "  reset state: "
            f"initial_left_open={row['initial_left_open_episodes']}, "
            f"failures_among_them={row['failures_with_initial_left_open']}"
        )
        failure_keys = (
            "gripper_not_opened",
            "open_but_no_effective_grasp",
            "fork_not_lifted",
            "spoon_not_lifted",
            "both_lifted_but_not_placed",
            "target_unstable_or_dropped",
            "trace_evaluator_mismatch",
        )
        print(
            "  failure stages: "
            + ", ".join(
                f"{key}={row[key]}" for key in failure_keys
            )
        )
    print(f"per_episode_csv={args.output}")
    print(f"summary_csv={args.summary_output}")


if __name__ == "__main__":
    main()
