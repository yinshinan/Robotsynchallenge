#!/usr/bin/env python
"""Extract stage-level outcomes from sample_loading diagnostic logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


STATE_RE = re.compile(
    r"\[sample_loading state\] step=(\d+) success=(True|False) "
    r"xy=([-+0-9.eE]+) cube_z=([-+0-9.eE]+) "
    r"rack_z=([-+0-9.eE]+) placement=(True|False) stable=(\d+)"
)
GRIPPER_RE = re.compile(
    r"\[GRIPPER DEBUG\].*?left_state=([-+0-9.eE]+).*?"
    r"left_action=([-+0-9.eE]+).*?right_state=([-+0-9.eE]+).*?"
    r"right_action=([-+0-9.eE]+).*?right_arm_error=([-+0-9.eE]+)"
)
VIDEO_RE = re.compile(
    r"Video saved: .*?episode_(\d+)_seed_(\d+)_(success|fail)\.mp4"
)
FINAL_RE = re.compile(
    r"\[DONE DEBUG\].*?cube_xy_dist': tensor\(\[([-+0-9.eE]+)\]\).*?"
    r"cube_z': tensor\(\[([-+0-9.eE]+)\]\).*?"
    r"place_stable_count': tensor\(\[(\d+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    return parser.parse_args()


def clean_log(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").replace(b"\r", b"\n").decode(
        "utf-8", errors="replace"
    )


def classify(max_z: float, min_xy: float, final_z: float, success: bool) -> str:
    if success:
        return "成功稳定放置"
    if max_z < 0.90:
        return "未有效抬起"
    if min_xy >= 0.05:
        return "已抬起但未对准料架"
    if final_z >= 0.89:
        return "已到料架但未稳定放置"
    return "到达料架后掉落或偏离"


def parse_one_log(path: Path, action_steps: int) -> list[dict[str, object]]:
    text = clean_log(path)
    starts = list(re.finditer(r"Episode (\d{3})/\d{3}:\s+0%", text))
    episodes = []
    for index, start in enumerate(starts):
        end_offset = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start() : end_offset]
        video_match = VIDEO_RE.search(block)
        if video_match is None:
            continue

        raw_state_rows = [
            {
                "step": int(match.group(1)),
                "success": match.group(2) == "True",
                "xy": float(match.group(3)),
                "cube_z": float(match.group(4)),
                "rack_z": float(match.group(5)),
                "placement": match.group(6) == "True",
                "stable": int(match.group(7)),
            }
            for match in STATE_RE.finditer(block)
        ]
        state_rows = []
        observed_positive_step = False
        for row in raw_state_rows:
            # The evaluator resets the next episode before it creates the next
            # progress bar, so the following episode's step-0 diagnostics are
            # present at the end of this block. Exclude that reset spillover.
            if row["step"] == 0 and observed_positive_step:
                break
            state_rows.append(row)
            observed_positive_step = observed_positive_step or row["step"] > 0
        gripper_rows = [
            {
                "sample": sample_index,
                "approx_step": sample_index * 10,
                "left_state": float(match.group(1)),
                "left_action": float(match.group(2)),
                "right_state": float(match.group(3)),
                "right_action": float(match.group(4)),
                "right_arm_error": float(match.group(5)),
            }
            for sample_index, match in enumerate(GRIPPER_RE.finditer(block))
        ]
        final_match = FINAL_RE.search(block)
        final_xy = float(final_match.group(1)) if final_match else state_rows[-1]["xy"]
        final_z = float(final_match.group(2)) if final_match else state_rows[-1]["cube_z"]
        final_stable = int(final_match.group(3)) if final_match else state_rows[-1]["stable"]
        max_z = max(row["cube_z"] for row in state_rows)
        min_xy = min(row["xy"] for row in state_rows)
        success = video_match.group(3) == "success"
        post_grasp_rows = [row for row in gripper_rows if row["approx_step"] >= 180]
        min_left_post_grasp = min(
            (row["left_state"] for row in post_grasp_rows), default=float("nan")
        )
        max_right_post_grasp = max(
            (row["right_state"] for row in post_grasp_rows), default=float("nan")
        )

        episodes.append(
            {
                "n_action_steps": action_steps,
                "episode": int(video_match.group(1)),
                "seed": int(video_match.group(2)),
                "success": success,
                "initial_xy": state_rows[0]["xy"],
                "max_cube_z": max_z,
                "min_cube_xy": min_xy,
                "final_cube_z": final_z,
                "final_cube_xy": final_xy,
                "final_stable_count": final_stable,
                "left_closed_after_step_180": min_left_post_grasp < 0.30,
                "right_open_after_step_180": max_right_post_grasp > 0.80,
                "stage": classify(max_z, min_xy, final_z, success),
            }
        )
    return episodes


def main() -> None:
    args = parse_args()
    rows = []
    for action_steps in (50, 10, 5, 1):
        rows.extend(
            parse_one_log(
                args.matrix_dir / f"n_action_steps_{action_steps}.log",
                action_steps,
            )
        )

    fieldnames = list(rows[0])
    with (args.matrix_dir / "episode_stage_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for action_steps in (50, 10, 5, 1):
        group = [row for row in rows if row["n_action_steps"] == action_steps]
        stage_counts: dict[str, int] = {}
        for row in group:
            stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1
        aggregate.append(
            {
                "n_action_steps": action_steps,
                "episodes": len(group),
                "successes": sum(bool(row["success"]) for row in group),
                "lifted": sum(float(row["max_cube_z"]) >= 0.90 for row in group),
                "reached_rack_xy": sum(float(row["min_cube_xy"]) < 0.05 for row in group),
                "left_close_attempt": sum(
                    bool(row["left_closed_after_step_180"]) for row in group
                ),
                "stage_counts": stage_counts,
            }
        )

    (args.matrix_dir / "stage_summary.json").write_text(
        json.dumps({"aggregate": aggregate, "episodes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# sample_loading ACT 动作块诊断汇总",
        "",
        "| n_action_steps | 成功 | 曾抬起 | 到达料架 XY | 左手闭合尝试 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        report_lines.append(
            f"| {item['n_action_steps']} | {item['successes']}/{item['episodes']} "
            f"| {item['lifted']}/{item['episodes']} "
            f"| {item['reached_rack_xy']}/{item['episodes']} "
            f"| {item['left_close_attempt']}/{item['episodes']} |"
        )
    report_lines.extend(["", "## 分阶段结果", ""])
    for item in aggregate:
        report_lines.append(f"- n={item['n_action_steps']}: {item['stage_counts']}")
    (args.matrix_dir / "stage_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
