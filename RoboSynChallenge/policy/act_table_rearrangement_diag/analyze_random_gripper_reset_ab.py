#!/usr/bin/env python
"""Compare baseline random evaluation with explicit gripper-reset evaluation."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from policy.act_table_rearrangement_diag.analyze_multiseed import (
    OPEN_THRESHOLD_M,
    episode_seeds,
    load_console_results,
    load_episodes,
    summarize_episode,
    write_csv,
)


def load_csv(path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["seed"]): row for row in rows if row["setting"] == "random"}


def as_bool(value):
    return str(value).lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--reset-trace", type=Path, required=True)
    parser.add_argument("--reset-console", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostic_logs/table_rearrangement_random_gripper_reset_ab.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "diagnostic_logs/table_rearrangement_random_gripper_reset_ab_summary.csv"
        ),
    )
    args = parser.parse_args()

    seeds = episode_seeds(args.rng_seed, args.episodes)
    baseline = load_csv(args.baseline_csv)
    reset_episodes = load_episodes(args.reset_trace)
    reset_results = load_console_results(args.reset_console)
    if len(baseline) != args.episodes:
        raise ValueError(f"Expected {args.episodes} random baseline rows, found {len(baseline)}")
    if len(reset_episodes) != args.episodes or len(reset_results) != args.episodes:
        raise ValueError(
            "Reset trace/console episode count mismatch: "
            f"trace={len(reset_episodes)}, console={len(reset_results)}, "
            f"expected={args.episodes}"
        )

    reset_rows = []
    for episode, seed in enumerate(seeds):
        reset_rows.append(
            summarize_episode(
                "random_gripper_reset",
                episode,
                seed,
                reset_episodes[episode],
                reset_results[episode],
            )
        )

    paired = []
    for reset in reset_rows:
        seed = int(reset["seed"])
        before = baseline[seed]
        before_success = as_bool(before["success"])
        after_success = bool(reset["success"])
        if before_success == after_success:
            change = "unchanged_success" if after_success else "unchanged_failure"
        else:
            change = "improved" if after_success else "regressed"
        paired.append(
            {
                "episode": reset["episode"],
                "seed": seed,
                "baseline_success": before_success,
                "reset_success": after_success,
                "change": change,
                "baseline_failure_stage": before["failure_stage"],
                "reset_failure_stage": reset["failure_stage"],
                "baseline_initial_left_open_m": before["initial_open_left_m"],
                "reset_initial_left_open_m": reset["initial_open_left_m"],
                "baseline_initial_right_open_m": before["initial_open_right_m"],
                "reset_initial_right_open_m": reset["initial_open_right_m"],
                "reset_max_fork_lift_m": reset["max_fork_lift_m"],
                "reset_max_spoon_lift_m": reset["max_spoon_lift_m"],
                "reset_final_fork_target_xy_error_m": reset[
                    "final_fork_target_xy_error_m"
                ],
                "reset_final_spoon_target_xy_error_m": reset[
                    "final_spoon_target_xy_error_m"
                ],
            }
        )
    write_csv(args.output, paired)

    changes = Counter(row["change"] for row in paired)
    failure_stages = Counter(
        row["failure_stage"] for row in reset_rows if not row["success"]
    )
    baseline_successes = sum(as_bool(row["success"]) for row in baseline.values())
    reset_successes = sum(bool(row["success"]) for row in reset_rows)
    reset_initial_open = sum(
        row["initial_open_left_m"] >= OPEN_THRESHOLD_M
        or row["initial_open_right_m"] >= OPEN_THRESHOLD_M
        for row in reset_rows
    )
    summary = [{
        "episodes": args.episodes,
        "baseline_successes": baseline_successes,
        "reset_successes": reset_successes,
        "baseline_success_rate": baseline_successes / args.episodes,
        "reset_success_rate": reset_successes / args.episodes,
        "improved": changes["improved"],
        "regressed": changes["regressed"],
        "unchanged_success": changes["unchanged_success"],
        "unchanged_failure": changes["unchanged_failure"],
        "reset_initial_gripper_open_episodes": reset_initial_open,
        "reset_gripper_not_opened": failure_stages["gripper_not_opened"],
        "reset_open_but_no_effective_grasp": failure_stages[
            "open_but_no_effective_grasp"
        ],
        "reset_fork_not_lifted": failure_stages["fork_not_lifted"],
        "reset_spoon_not_lifted": failure_stages["spoon_not_lifted"],
        "reset_both_lifted_but_not_placed": failure_stages[
            "both_lifted_but_not_placed"
        ],
        "reset_target_unstable_or_dropped": failure_stages[
            "target_unstable_or_dropped"
        ],
    }]
    write_csv(args.summary_output, summary)

    print(
        f"baseline={baseline_successes}/{args.episodes} "
        f"({100 * baseline_successes / args.episodes:.1f}%)"
    )
    print(
        f"gripper_reset={reset_successes}/{args.episodes} "
        f"({100 * reset_successes / args.episodes:.1f}%)"
    )
    print(
        f"paired: improved={changes['improved']} regressed={changes['regressed']} "
        f"unchanged_success={changes['unchanged_success']} "
        f"unchanged_failure={changes['unchanged_failure']}"
    )
    print(f"reset_initial_gripper_open_episodes={reset_initial_open}")
    print(f"paired_csv={args.output}")
    print(f"summary_csv={args.summary_output}")


if __name__ == "__main__":
    main()
