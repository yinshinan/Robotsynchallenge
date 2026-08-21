#!/usr/bin/env python
"""Summarize gripper state from reset-event probe CSV files."""

import argparse
import csv
from pathlib import Path


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(path, threshold):
    rows = load_rows(path)
    if not rows:
        raise RuntimeError(f"Probe CSV is empty: {path}")
    result = {
        "variant": rows[0]["variant"],
        "episodes": len(rows),
        "left_joint_id": int(rows[0]["immediate_left_joint_id"]),
        "right_joint_id": int(rows[0]["immediate_right_joint_id"]),
    }
    for side in ("left", "right"):
        current = [float(row[f"immediate_{side}_qpos_m"]) for row in rows]
        target = [float(row[f"immediate_{side}_target_qpos_m"]) for row in rows]
        after_sync = [float(row[f"after_sync_{side}_qpos_m"]) for row in rows]
        after_obs = [float(row[f"after_obs_{side}_qpos_m"]) for row in rows]
        result.update(
            {
                f"{side}_open_count": sum(value >= threshold for value in current),
                f"{side}_max_qpos_m": max(current),
                f"{side}_target_open_count": sum(
                    value >= threshold for value in target
                ),
                f"{side}_max_current_target_error_m": max(
                    abs(a - b) for a, b in zip(current, target)
                ),
                f"{side}_max_sync_change_m": max(
                    abs(a - b) for a, b in zip(current, after_sync)
                ),
                f"{side}_max_obs_change_m": max(
                    abs(a - b) for a, b in zip(current, after_obs)
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--threshold-m", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_logs/random_reset_probe_summary.csv"),
    )
    args = parser.parse_args()

    summaries = [summarize(path, args.threshold_m) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(
            f"{row['variant']}: episodes={row['episodes']} "
            f"left_open={row['left_open_count']} "
            f"right_open={row['right_open_count']} "
            f"left_max={row['left_max_qpos_m']:.6f} "
            f"right_max={row['right_max_qpos_m']:.6f}"
        )
    print(f"summary_csv={args.output}")


if __name__ == "__main__":
    main()
