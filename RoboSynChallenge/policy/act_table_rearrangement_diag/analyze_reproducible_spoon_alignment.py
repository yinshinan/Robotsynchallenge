#!/usr/bin/env python
"""Measure right-arm lateral alignment against spoon initial x position."""

import csv
import json
from collections import defaultdict
from pathlib import Path


TRACE = Path("diagnostic_logs/multiseed_random_reproducible_physical_n50.jsonl")
RESULTS = Path("diagnostic_logs/table_rearrangement_multiseed_reproducible.csv")
OUTPUT = Path("diagnostic_logs/table_rearrangement_spoon_alignment.csv")


def main():
    episodes = defaultdict(list)
    with TRACE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episodes[int(row["episode"])].append(row)
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        results = {
            int(row["trace_episode"]): row
            for row in csv.DictReader(handle)
            if row["setting"] == "random"
        }

    rows = []
    for episode, trace_rows in sorted(episodes.items()):
        closest = min(trace_rows, key=lambda row: row["right_link6_to_spoon_m"])
        spoon = closest["spoon_xyz_m"]
        link = closest["right_link6_xyz_m"]
        result = results[episode]
        rows.append(
            {
                "episode": episode + 1,
                "seed": result["seed"],
                "success": result["success"],
                "failure_stage": result["failure_stage"],
                "initial_spoon_x_m": trace_rows[0]["spoon_xyz_m"][0],
                "initial_spoon_y_m": trace_rows[0]["spoon_xyz_m"][1],
                "closest_step": closest["step"],
                "min_right_link6_to_spoon_m": closest[
                    "right_link6_to_spoon_m"
                ],
                "closest_spoon_minus_link6_x_m": spoon[0] - link[0],
                "closest_spoon_minus_link6_y_m": spoon[1] - link[1],
                "closest_spoon_minus_link6_z_m": spoon[2] - link[2],
                "max_spoon_lift_m": result["max_spoon_lift_m"],
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for label, selected in (
        ("spoon_x_lt_0.45", [r for r in rows if r["initial_spoon_x_m"] < 0.45]),
        ("spoon_x_ge_0.45", [r for r in rows if r["initial_spoon_x_m"] >= 0.45]),
    ):
        successes = sum(str(row["success"]).lower() == "true" for row in selected)
        print(f"{label}: {successes}/{len(selected)} success")
    print(f"alignment_csv={OUTPUT}")


if __name__ == "__main__":
    main()
