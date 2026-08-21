#!/usr/bin/env python
"""Audit sample_loading episode lengths and expert gripper timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def transitions(values: np.ndarray) -> list[dict[str, object]]:
    opened = values >= 0.5
    indices = np.where(opened[1:] != opened[:-1])[0] + 1
    return [
        {
            "step": int(index),
            "direction": "open" if opened[index] else "close",
            "previous": float(values[index - 1]),
            "value": float(values[index]),
        }
        for index in indices
    ]


def main() -> int:
    args = parse_args()
    output = args.output or args.dataset / "sample_loading_dataset_audit.json"
    episodes = [json.loads(line) for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    rows = []
    schedule_hashes: dict[str, int] = {}
    for item in episodes:
        episode = int(item["episode_index"])
        parquet = args.dataset / (
            f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        )
        table = pq.read_table(parquet, columns=["action", "cube_pose", "rack_pose"])
        actions = np.asarray(table["action"].to_pylist())
        grippers = actions[:, [6, 13]].astype(np.float32)
        cube_positions = np.asarray(table["cube_pose"].to_pylist())[:, :3, 3]
        rack_positions = np.asarray(table["rack_pose"].to_pylist())[:, :3, 3]
        cube_speeds = np.linalg.norm(np.diff(cube_positions, axis=0), axis=1) * 25.0
        stable_from_end = 0
        for speed in cube_speeds[::-1]:
            if speed >= 0.05:
                break
            stable_from_end += 1
        digest = hashlib.sha256(grippers.tobytes()).hexdigest()
        schedule_hashes[digest] = schedule_hashes.get(digest, 0) + 1
        rows.append(
            {
                "episode": episode,
                "source": "official" if episode < 1000 else "custom",
                "length": int(len(actions)),
                "left_transitions": transitions(grippers[:, 0]),
                "right_transitions": transitions(grippers[:, 1]),
                "gripper_schedule_sha256": digest,
                "max_cube_z": float(cube_positions[:, 2].max()),
                "final_cube_z": float(cube_positions[-1, 2]),
                "final_cube_to_rack_xy": float(
                    np.linalg.norm(cube_positions[-1, :2] - rack_positions[-1, :2])
                ),
                "max_last_8_cube_speed": float(cube_speeds[-8:].max()),
                "stable_frames_at_end": stable_from_end,
            }
        )

    groups = {}
    for source in ("official", "custom", "all"):
        selected = rows if source == "all" else [row for row in rows if row["source"] == source]
        lengths = [int(row["length"]) for row in selected]
        groups[source] = {
            "episodes": len(selected),
            "frames": sum(lengths),
            "min_length": min(lengths),
            "median_length": median(lengths),
            "mean_length": mean(lengths),
            "max_length": max(lengths),
            "unique_lengths": sorted(set(lengths)),
            "max_final_cube_to_rack_xy": max(float(row["final_cube_to_rack_xy"]) for row in selected),
            "episodes_final_xy_over_5cm": sum(
                float(row["final_cube_to_rack_xy"]) >= 0.05 for row in selected
            ),
            "episodes_never_lifted_3cm": sum(
                float(row["max_cube_z"]) < 0.86 for row in selected
            ),
            "median_stable_frames_at_end": median(
                int(row["stable_frames_at_end"]) for row in selected
            ),
            "episodes_with_under_8_stable_end_frames": sum(
                int(row["stable_frames_at_end"]) < 8 for row in selected
            ),
            "median_max_last_8_cube_speed": median(
                float(row["max_last_8_cube_speed"]) for row in selected
            ),
        }

    canonical = rows[0]
    result = {
        "groups": groups,
        "unique_full_gripper_schedules": len(schedule_hashes),
        "schedule_frequencies": sorted(schedule_hashes.values(), reverse=True),
        "canonical_left_transitions": canonical["left_transitions"],
        "canonical_right_transitions": canonical["right_transitions"],
        "all_episodes_share_transitions": all(
            row["left_transitions"] == canonical["left_transitions"]
            and row["right_transitions"] == canonical["right_transitions"]
            for row in rows
        ),
        "handoff_threshold_overlap_frames": 244 - 228,
        "handoff_threshold_overlap_seconds_at_25fps": (244 - 228) / 25,
        "episodes_final_xy_over_5cm": [
            int(row["episode"])
            for row in rows
            if float(row["final_cube_to_rack_xy"]) >= 0.05
        ],
        "episodes_with_under_8_stable_end_frames": [
            int(row["episode"])
            for row in rows
            if int(row["stable_frames_at_end"]) < 8
        ],
        "notes": [
            "A full-schedule hash differs when episodes have different tail lengths.",
            "Transition equality compares every threshold transition and is tail-length independent.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
