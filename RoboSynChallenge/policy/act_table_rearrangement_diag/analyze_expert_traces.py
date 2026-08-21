#!/usr/bin/env python
"""Summarize grasp-height expert replays and recompute strict success."""

import argparse
import csv
import json
import math
from pathlib import Path


def distance(a, b, dimensions):
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(dimensions)))


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"Trace is empty: {path}")

    streak = 0
    success_step = None
    for row in rows:
        plate = row["plate_xyz_m"]
        fork_target = [plate[0], plate[1] + 0.16, plate[2]]
        spoon_target = [plate[0], plate[1] - 0.16, plate[2]]
        placed = (
            distance(row["fork_xyz_m"], fork_target, 2) <= 0.035
            and distance(row["spoon_xyz_m"], spoon_target, 2) <= 0.035
            and abs(row["fork_xyz_m"][2] - plate[2]) <= 0.035
            and abs(row["spoon_xyz_m"][2] - plate[2]) <= 0.035
        )
        streak = streak + 1 if placed else 0
        lifted = row["fork_lift_m"] >= 0.020 and row["spoon_lift_m"] >= 0.020
        if success_step is None and streak >= 5 and lifted:
            success_step = row["step"]

    first, last = rows[0], rows[-1]
    return {
        "case": path.stem,
        "frames": len(rows),
        "strict_success": success_step is not None,
        "first_success_step": success_step,
        "max_fork_lift_m": max(row["fork_lift_m"] for row in rows),
        "max_spoon_lift_m": max(row["spoon_lift_m"] for row in rows),
        "fork_planar_displacement_m": distance(
            first["fork_xyz_m"], last["fork_xyz_m"], 2
        ),
        "spoon_planar_displacement_m": distance(
            first["spoon_xyz_m"], last["spoon_xyz_m"], 2
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_logs/table_rearrangement_expert_z_summary.csv"),
    )
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(summary)
    print(f"summary_csv={args.output}")


if __name__ == "__main__":
    main()
