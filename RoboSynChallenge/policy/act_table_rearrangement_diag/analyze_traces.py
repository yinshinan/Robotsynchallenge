#!/usr/bin/env python
"""Summarize JSONL traces produced by the diagnostic ACT adapter."""

import argparse
import csv
import json
from pathlib import Path


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def maximum(rows, key, item=None):
    values = [row[key] if item is None else row[key][item] for row in rows]
    return max(values)


def minimum(rows, key):
    values = [row[key] for row in rows if key in row]
    return min(values) if values else float("nan")


def minimum_alias(rows, *keys):
    for key in keys:
        if any(key in row for row in rows):
            return minimum(rows, key)
    return float("nan")


def summarize(path):
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    last = rows[-1]
    return {
        "case": path.stem,
        "steps": len(rows),
        "gripper_mode": rows[0]["gripper_mode"],
        "max_raw_left_m": maximum(rows, "raw_gripper_action_m", 0),
        "max_env_left_m": maximum(rows, "env_gripper_action_m", 0),
        "max_physical_left_m": maximum(rows, "physical_gripper_qpos_m", 0),
        "max_raw_right_m": maximum(rows, "raw_gripper_action_m", 1),
        "max_env_right_m": maximum(rows, "env_gripper_action_m", 1),
        "max_physical_right_m": maximum(rows, "physical_gripper_qpos_m", 1),
        "min_left_link6_to_fork_m": minimum_alias(
            rows, "left_link6_to_fork_m", "left_tcp_to_fork_m"
        ),
        "min_right_link6_to_spoon_m": minimum_alias(
            rows, "right_link6_to_spoon_m", "right_tcp_to_spoon_m"
        ),
        "max_fork_lift_m": maximum(rows, "fork_lift_m"),
        "max_spoon_lift_m": maximum(rows, "spoon_lift_m"),
        "final_fork_displacement_m": last["fork_planar_displacement_m"],
        "final_spoon_displacement_m": last["spoon_planar_displacement_m"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_logs/table_rearrangement_ab_summary.csv"),
    )
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    columns = (
        "case",
        "max_env_left_m",
        "max_physical_left_m",
        "max_env_right_m",
        "max_physical_right_m",
        "min_left_link6_to_fork_m",
        "min_right_link6_to_spoon_m",
        "max_fork_lift_m",
        "max_spoon_lift_m",
    )
    print("\t".join(columns))
    for summary in summaries:
        print(
            "\t".join(
                str(summary[column]) if column == "case" else f"{summary[column]:.6f}"
                for column in columns
            )
        )
    print(f"summary_csv={args.output}")


if __name__ == "__main__":
    main()
