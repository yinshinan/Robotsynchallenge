#!/usr/bin/env python
"""Summarize fixed-seed factor traces and their strict eval outcomes."""

import csv
import json
import math
import re
from pathlib import Path


LOG_DIR = Path("diagnostic_logs")
OUTPUT = LOG_DIR / "table_rearrangement_factor_ab_summary.csv"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome(path):
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    text = ANSI.sub("", text)
    matches = re.findall(r"Episode 01/01:\s*(SUCCESS|FAIL)", text)
    return matches[-1].lower() if matches else "unknown"


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def yaw_degrees(pose):
    _, _, _, qw, qx, qy, qz = pose
    return math.degrees(
        math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def main():
    summaries = []
    for trace_path in sorted(LOG_DIR.glob("factor_seed*_*.jsonl")):
        match = re.fullmatch(r"factor_seed(\d+)_(.+)\.jsonl", trace_path.name)
        if not match:
            continue
        seed, factor = match.groups()
        console_path = trace_path.with_suffix(".console.log")
        rows = load_rows(trace_path)
        first = rows[0]
        closest = min(rows, key=lambda row: norm(row["spoon_minus_right_link6_xyz_m"]))
        plate = rows[-1]["plate_pose_xyz_qwxyz"]
        fork = rows[-1]["fork_pose_xyz_qwxyz"]
        spoon = rows[-1]["spoon_pose_xyz_qwxyz"]
        summaries.append(
            {
                "seed": int(seed),
                "factor": factor,
                "outcome": outcome(console_path) if console_path.exists() else "unknown",
                "steps": len(rows),
                "initial_spoon_x_m": first["spoon_pose_xyz_qwxyz"][0],
                "initial_spoon_y_m": first["spoon_pose_xyz_qwxyz"][1],
                "initial_spoon_yaw_deg": yaw_degrees(first["spoon_pose_xyz_qwxyz"]),
                "cam_high_fx": first["cameras"]["cam_high"]["intrinsics_3x3"][0][0],
                "cam_high_fy": first["cameras"]["cam_high"]["intrinsics_3x3"][1][1],
                "closest_step": closest["step"],
                "closest_spoon_minus_right_x_m": closest[
                    "spoon_minus_right_link6_xyz_m"
                ][0],
                "closest_spoon_minus_right_y_m": closest[
                    "spoon_minus_right_link6_xyz_m"
                ][1],
                "closest_spoon_minus_right_z_m": closest[
                    "spoon_minus_right_link6_xyz_m"
                ][2],
                "max_fork_lift_m": max(row["fork_lift_m"] for row in rows),
                "max_spoon_lift_m": max(row["spoon_lift_m"] for row in rows),
                "final_fork_target_xy_error_m": norm(
                    [fork[0] - plate[0], fork[1] - (plate[1] + 0.16)]
                ),
                "final_spoon_target_xy_error_m": norm(
                    [spoon[0] - plate[0], spoon[1] - (plate[1] - 0.16)]
                ),
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"summary_csv={OUTPUT} rows={len(summaries)}")


if __name__ == "__main__":
    main()
