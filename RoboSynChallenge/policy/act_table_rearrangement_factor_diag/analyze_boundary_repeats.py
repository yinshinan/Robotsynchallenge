#!/usr/bin/env python
"""Summarize repeated seed-1879422756 boundary ablations.

The repeated processes sometimes omit the final FAIL banner while shutting the
simulator down.  A trace with fewer than MAX_STEPS rows necessarily ended on
strict task success; a MAX_STEPS trace is a strict-task failure.
"""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path("diagnostic_logs")
OUTPUT = LOG_DIR / "table_rearrangement_seed1879422756_repeat_summary.csv"
MAX_STEPS = 361
MIN_LIFT_M = 0.020


def planar_error(pose, target_x, target_y):
    return math.hypot(pose[0] - target_x, pose[1] - target_y)


def classify(rows):
    fork_lift = max(row["fork_lift_m"] for row in rows)
    spoon_lift = max(row["spoon_lift_m"] for row in rows)
    if len(rows) < MAX_STEPS:
        return "success"
    if fork_lift < MIN_LIFT_M or spoon_lift < MIN_LIFT_M:
        return "pick_failure"
    return "placement_failure"


def main():
    paths = sorted(LOG_DIR.glob("factor_seed1879422756_*rep*.jsonl"))
    summaries = []
    grouped = defaultdict(list)
    pattern = re.compile(
        r"factor_seed1879422756_(baseline|camera_fx_zero|spoon_y_center)_"
        r"(?:rep|repeat)(\d+)\.jsonl"
    )
    for path in paths:
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        factor, repeat = match.groups()
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        first, last = rows[0], rows[-1]
        plate = last["plate_pose_xyz_qwxyz"]
        fork = last["fork_pose_xyz_qwxyz"]
        spoon = last["spoon_pose_xyz_qwxyz"]
        outcome = classify(rows)
        summary = {
            "factor": factor,
            "repeat": int(repeat),
            "outcome": outcome,
            "steps": len(rows),
            "initial_spoon_y_m": first["spoon_pose_xyz_qwxyz"][1],
            "cam_high_fx": first["cameras"]["cam_high"]["intrinsics_3x3"][0][0],
            "max_fork_lift_m": max(row["fork_lift_m"] for row in rows),
            "max_spoon_lift_m": max(row["spoon_lift_m"] for row in rows),
            "final_fork_target_xy_error_m": planar_error(
                fork, plate[0], plate[1] + 0.16
            ),
            "final_spoon_target_xy_error_m": planar_error(
                spoon, plate[0], plate[1] - 0.16
            ),
        }
        summaries.append(summary)
        grouped[factor].append(outcome)

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"summary_csv={OUTPUT} rows={len(summaries)}")
    for factor in sorted(grouped):
        outcomes = grouped[factor]
        print(
            factor,
            f"success={outcomes.count('success')}/{len(outcomes)}",
            f"pick_failure={outcomes.count('pick_failure')}",
            f"placement_failure={outcomes.count('placement_failure')}",
        )


if __name__ == "__main__":
    main()
