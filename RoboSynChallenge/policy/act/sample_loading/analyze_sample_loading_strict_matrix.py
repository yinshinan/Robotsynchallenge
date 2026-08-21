#!/usr/bin/env python
"""Build a six-stage funnel from strict sample_loading matrix logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ACTION_STEPS = (8, 10, 12, 15)
BOOL_KEYS = (
    "strict_left_released",
    "strict_right_released",
    "strict_released",
    "strict_right_approached",
    "strict_lifted",
    "strict_right_held_while_lifted",
    "strict_left_handoff_reached",
    "strict_reached_rack",
    "strict_metrics_active",
)
FLOAT_KEYS = (
    "cube_xy_dist",
    "cube_z",
    "cube_lin_vel_norm",
    "cube_to_left_eef_dist",
    "cube_to_right_eef_dist",
    "left_gripper_q_mean",
    "right_gripper_q_mean",
    "strict_max_cube_z",
    "strict_min_rack_xy",
    "strict_min_left_eef_dist",
    "strict_min_right_eef_dist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--action-steps", type=int, nargs="+")
    return parser.parse_args()


def clean_log(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").replace(b"\r", b"\n").decode(
        "utf-8", errors="replace"
    )


def tensor_bool(text: str, key: str) -> bool:
    match = re.search(rf"'{re.escape(key)}': tensor\(\[(True|False)\]\)", text)
    if match is None:
        raise ValueError(f"Missing boolean metric {key}")
    return match.group(1) == "True"


def tensor_float(text: str, key: str) -> float:
    match = re.search(rf"'{re.escape(key)}': tensor\(\[([-+0-9.eE]+)\]\)", text)
    if match is None:
        raise ValueError(f"Missing float metric {key}")
    return float(match.group(1))


def classify(row: dict[str, object]) -> str:
    if not row["strict_right_approached"]:
        return "未接近右手抓取范围"
    if not row["strict_lifted"]:
        return "已接近但未抬起"
    if not row["strict_right_held_while_lifted"]:
        return "抬起时未保持右手近距"
    if not row["strict_left_handoff_reached"]:
        return "右手持有但未到左手交接范围"
    if not row["strict_reached_rack"]:
        return "到达交接范围但未到料架"
    if not row["success"]:
        return "到达料架但未释放稳定"
    return "严格成功"


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - half_width, center + half_width


def main() -> int:
    args = parse_args()
    action_steps_values = args.action_steps or [
        int(match.group(1))
        for path in sorted(args.matrix_dir.glob("n_action_steps_*.log"))
        if (match := re.fullmatch(r"n_action_steps_(\d+)\.log", path.name))
    ]
    if not action_steps_values:
        action_steps_values = list(ACTION_STEPS)
    seeds = np.random.RandomState(args.seed).randint(0, 2**31 - 1, size=args.episodes)
    rows: list[dict[str, object]] = []
    for action_steps in action_steps_values:
        text = clean_log(args.matrix_dir / f"n_action_steps_{action_steps}.log")
        done_lines = [line for line in text.splitlines() if "[DONE DEBUG]" in line]
        if len(done_lines) != args.episodes:
            raise ValueError(
                f"Expected {args.episodes} DONE rows for n={action_steps}, got {len(done_lines)}"
            )
        for episode, line in enumerate(done_lines):
            success_match = re.search(r"'success': tensor\(\[(True|False)\]\)", line)
            if success_match is None:
                raise ValueError("Missing success metric")
            row: dict[str, object] = {
                "n_action_steps": action_steps,
                "episode": episode,
                "seed": int(seeds[episode]),
                "success": success_match.group(1) == "True",
            }
            row.update({key: tensor_bool(line, key) for key in BOOL_KEYS})
            row.update({key: tensor_float(line, key) for key in FLOAT_KEYS})
            row["stage"] = classify(row)
            rows.append(row)

    csv_path = args.matrix_dir / "strict_episode_stages.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for action_steps in action_steps_values:
        group = [row for row in rows if row["n_action_steps"] == action_steps]
        stages = {}
        for row in group:
            stages[str(row["stage"])] = stages.get(str(row["stage"]), 0) + 1
        success_count = sum(bool(row["success"]) for row in group)
        low, high = wilson(success_count, len(group))
        aggregate.append(
            {
                "n_action_steps": action_steps,
                "episodes": len(group),
                "right_approached": sum(bool(row["strict_right_approached"]) for row in group),
                "lifted": sum(bool(row["strict_lifted"]) for row in group),
                "right_held_while_lifted": sum(
                    bool(row["strict_right_held_while_lifted"]) for row in group
                ),
                "left_handoff_reached": sum(
                    bool(row["strict_left_handoff_reached"]) for row in group
                ),
                "reached_rack": sum(bool(row["strict_reached_rack"]) for row in group),
                "successes": success_count,
                "success_wilson_95": [low, high],
                "stage_counts": stages,
                "median_min_rack_xy": float(
                    np.median([float(row["strict_min_rack_xy"]) for row in group])
                ),
                "median_max_cube_z": float(
                    np.median([float(row["strict_max_cube_z"]) for row in group])
                ),
            }
        )

    json_path = args.matrix_dir / "strict_stage_summary.json"
    json_path.write_text(
        json.dumps({"aggregate": aggregate, "episodes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        f"# sample_loading strict {args.episodes}-episode matrix",
        "",
        "| n | 接近右手 | 抬起 | 右手近距持有 | 到左手交接范围 | 到料架5cm | 严格成功 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        report.append(
            f"| {item['n_action_steps']} | {item['right_approached']}/{args.episodes} | "
            f"{item['lifted']}/{args.episodes} | "
            f"{item['right_held_while_lifted']}/{args.episodes} | "
            f"{item['left_handoff_reached']}/{args.episodes} | "
            f"{item['reached_rack']}/{args.episodes} | "
            f"{item['successes']}/{args.episodes} |"
        )
    report.extend(["", "## 最深失败阶段", ""])
    for item in aggregate:
        report.append(f"- n={item['n_action_steps']}: {item['stage_counts']}")
    report_path = args.matrix_dir / "strict_stage_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
