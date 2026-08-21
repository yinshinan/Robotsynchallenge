#!/usr/bin/env python
"""Join the 20-seed fx-zero trace with seeds and baseline outcomes."""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path("diagnostic_logs")
TRACE = LOG_DIR / "multiseed_random_reproducible_fx_zero_n50.jsonl"
CONSOLE = LOG_DIR / "multiseed_random_reproducible_fx_zero_n50.console.log"
RESET = LOG_DIR / "reproducible_reset_factors_20.csv"
BASELINE = LOG_DIR / "table_rearrangement_multiseed_reproducible.csv"
OUTPUT = LOG_DIR / "table_rearrangement_fx_zero_20_summary.csv"
MIN_LIFT_M = 0.020


def bool_value(value):
    return value.strip().lower() == "true"


def planar_error(pose, target_x, target_y):
    return math.hypot(pose[0] - target_x, pose[1] - target_y)


def main():
    with RESET.open(newline="", encoding="utf-8") as handle:
        reset_rows = {int(row["episode"]): row for row in csv.DictReader(handle)}
    with BASELINE.open(newline="", encoding="utf-8") as handle:
        baseline_rows = {
            int(row["episode"]): row
            for row in csv.DictReader(handle)
            if row["setting"] == "random"
        }

    text = CONSOLE.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    outcomes = {
        int(episode): result == "SUCCESS"
        for episode, result in re.findall(
            r"Episode\s+(\d{2})/20:\s*(SUCCESS|FAIL)", text
        )
    }

    grouped = defaultdict(list)
    with TRACE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[int(row["episode"]) + 1].append(row)

    summaries = []
    for episode in sorted(grouped):
        rows = grouped[episode]
        first, last = rows[0], rows[-1]
        plate = last["plate_pose_xyz_qwxyz"]
        fork = last["fork_pose_xyz_qwxyz"]
        spoon = last["spoon_pose_xyz_qwxyz"]
        max_fork_lift = max(row["fork_lift_m"] for row in rows)
        max_spoon_lift = max(row["spoon_lift_m"] for row in rows)
        success = outcomes[episode]
        if success:
            stage = "success"
        elif max_fork_lift < MIN_LIFT_M or max_spoon_lift < MIN_LIFT_M:
            stage = "pick_failure"
        else:
            stage = "placement_failure"
        reset = reset_rows[episode]
        baseline = baseline_rows[episode]
        summaries.append(
            {
                "episode": episode,
                "seed": int(reset["seed"]),
                "baseline_success": bool_value(baseline["success"]),
                "baseline_failure_stage": baseline["failure_stage"],
                "fx_zero_success": success,
                "fx_zero_failure_stage": stage,
                "steps": len(rows),
                "initial_spoon_x_m": float(reset["spoon_x_m"]),
                "initial_spoon_y_m": float(reset["spoon_y_m"]),
                "original_cam_high_fx": float(reset["cam_high_fx"]),
                "fx_zero_cam_high_fx": first["cameras"]["cam_high"][
                    "intrinsics_3x3"
                ][0][0],
                "max_fork_lift_m": max_fork_lift,
                "max_spoon_lift_m": max_spoon_lift,
                "final_fork_target_xy_error_m": planar_error(
                    fork, plate[0], plate[1] + 0.16
                ),
                "final_spoon_target_xy_error_m": planar_error(
                    spoon, plate[0], plate[1] - 0.16
                ),
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    changed = [
        row
        for row in summaries
        if row["baseline_success"] != row["fx_zero_success"]
    ]
    print(
        f"summary_csv={OUTPUT} rows={len(summaries)} "
        f"baseline_success={sum(row['baseline_success'] for row in summaries)}/20 "
        f"fx_zero_success={sum(row['fx_zero_success'] for row in summaries)}/20"
    )
    for row in changed:
        print(
            f"changed episode={row['episode']} seed={row['seed']} "
            f"baseline={row['baseline_success']} fx_zero={row['fx_zero_success']} "
            f"fx_zero_stage={row['fx_zero_failure_stage']}"
        )


if __name__ == "__main__":
    main()
