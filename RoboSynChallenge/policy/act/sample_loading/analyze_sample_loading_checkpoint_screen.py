#!/usr/bin/env python
"""Summarize strict sample_loading logs across fine-tuning checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.act.sample_loading.analyze_sample_loading_strict_matrix import (  # noqa: E402
    BOOL_KEYS,
    FLOAT_KEYS,
    classify,
    clean_log,
    tensor_bool,
    tensor_float,
    wilson,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dirs = sorted(args.screen_dir.glob("ft*_n15_*eps"))
    if not candidate_dirs:
        raise FileNotFoundError(f"No checkpoint screen directories in {args.screen_dir}")
    rows: list[dict[str, object]] = []
    for candidate_dir in candidate_dirs:
        name_match = re.fullmatch(r"ft(\d+)_n15_(\d+)eps", candidate_dir.name)
        if name_match is None:
            continue
        checkpoint_step, expected_episodes = map(int, name_match.groups())
        seeds = np.random.RandomState(args.seed).randint(0, 2**31 - 1, size=expected_episodes)
        text = clean_log(candidate_dir / "n_action_steps_15.log")
        done_lines = [line for line in text.splitlines() if "[DONE DEBUG]" in line]
        if len(done_lines) != expected_episodes:
            raise ValueError(
                f"{candidate_dir.name}: expected {expected_episodes} DONE rows, got {len(done_lines)}"
            )
        for episode, line in enumerate(done_lines):
            success_match = re.search(r"'success': tensor\(\[(True|False)\]\)", line)
            if success_match is None:
                raise ValueError("Missing success metric")
            row: dict[str, object] = {
                "checkpoint_step": checkpoint_step,
                "episode": episode,
                "seed": int(seeds[episode]),
                "success": success_match.group(1) == "True",
            }
            row.update({key: tensor_bool(line, key) for key in BOOL_KEYS})
            row.update({key: tensor_float(line, key) for key in FLOAT_KEYS})
            row["stage"] = classify(row)
            rows.append(row)

    aggregates = []
    for checkpoint_step in sorted({int(row["checkpoint_step"]) for row in rows}):
        group = [row for row in rows if row["checkpoint_step"] == checkpoint_step]
        stages: dict[str, int] = {}
        for row in group:
            stages[str(row["stage"])] = stages.get(str(row["stage"]), 0) + 1
        successes = sum(bool(row["success"]) for row in group)
        low, high = wilson(successes, len(group))
        aggregates.append({
            "checkpoint_step": checkpoint_step,
            "episodes": len(group),
            "right_approached": sum(bool(row["strict_right_approached"]) for row in group),
            "lifted": sum(bool(row["strict_lifted"]) for row in group),
            "left_handoff_reached": sum(bool(row["strict_left_handoff_reached"]) for row in group),
            "reached_rack": sum(bool(row["strict_reached_rack"]) for row in group),
            "successes": successes,
            "success_wilson_95": [low, high],
            "stage_counts": stages,
        })

    with (args.screen_dir / "checkpoint_screen_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.screen_dir / "checkpoint_screen_summary.json").write_text(
        json.dumps({"aggregate": aggregates, "episodes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# sample_loading clean_v1 checkpoint screen",
        "",
        "| checkpoint | 接近 | 抬起 | 到交接 | 到料架 | 严格成功 | Wilson 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        total = item["episodes"]
        low, high = item["success_wilson_95"]
        report.append(
            f"| {item['checkpoint_step']} | {item['right_approached']}/{total} | "
            f"{item['lifted']}/{total} | {item['left_handoff_reached']}/{total} | "
            f"{item['reached_rack']}/{total} | {item['successes']}/{total} | "
            f"{low:.2%}–{high:.2%} |"
        )
    report.extend(["", "## 最深失败阶段", ""])
    for item in aggregates:
        report.append(f"- {item['checkpoint_step']}: {item['stage_counts']}")
    report_path = args.screen_dir / "checkpoint_screen_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
