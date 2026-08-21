#!/usr/bin/env python
"""Build a non-destructive, fail-closed quality manifest for sample_loading.

The source LeRobot dataset is read only.  The output manifest can be passed to
the companion clean-subset trainer so no parquet/video file is copied, renamed,
or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-final-xy", type=float, default=0.035)
    parser.add_argument("--min-z-above-rack", type=float, default=0.040)
    parser.add_argument("--max-z-above-rack", type=float, default=0.075)
    parser.add_argument("--max-upright-angle-deg", type=float, default=10.0)
    parser.add_argument("--lift-height", type=float, default=0.030)
    parser.add_argument("--stable-speed", type=float, default=0.050)
    parser.add_argument("--min-stable-tail-frames", type=int, default=20)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(value: Any) -> bool:
    return bool(np.isfinite(value).all())


def count_stable_tail(positions: np.ndarray, fps: float, threshold: float) -> int:
    if len(positions) < 2:
        return 0
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
    count = 0
    for speed in speeds[::-1]:
        if speed >= threshold:
            break
        count += 1
    return count


def transition_steps(values: np.ndarray, threshold: float) -> list[int]:
    opened = values >= threshold
    return (np.where(opened[1:] != opened[:-1])[0] + 1).astype(int).tolist()


def episode_row(
    dataset: Path,
    episode: int,
    fps: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parquet = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    schema_names = set(pq.read_schema(parquet).names)
    state_key = "observation.qpos" if "observation.qpos" in schema_names else "observation.state"
    table = pq.read_table(
        parquet,
        columns=["action", state_key, "observation.qvel", "observation.qf", "cube_pose", "rack_pose"],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    qpos = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
    qvel = np.asarray(table["observation.qvel"].to_pylist(), dtype=np.float32)
    qf = np.asarray(table["observation.qf"].to_pylist(), dtype=np.float32)
    cube_pose = np.asarray(table["cube_pose"].to_pylist(), dtype=np.float32)
    rack_pose = np.asarray(table["rack_pose"].to_pylist(), dtype=np.float32)

    cube_position = cube_pose[:, :3, 3]
    rack_position = rack_pose[:, :3, 3]
    final_xy = float(np.linalg.norm(cube_position[-1, :2] - rack_position[-1, :2]))
    final_z_above_rack = float(cube_position[-1, 2] - rack_position[-1, 2])
    final_z_axis = cube_pose[-1, :3, 2]
    upright_angle_deg = math.degrees(math.acos(float(np.clip(final_z_axis[2], -1.0, 1.0))))
    max_lift = float(cube_position[:, 2].max() - cube_position[0, 2])
    stable_tail = count_stable_tail(cube_position, fps, args.stable_speed)
    left_transitions = transition_steps(actions[:, 6], args.gripper_open_threshold)
    right_transitions = transition_steps(actions[:, 13], args.gripper_open_threshold)

    reasons: list[str] = []
    arrays = (actions, qpos, qvel, qf, cube_pose, rack_pose)
    if not all(finite_array(array) for array in arrays):
        reasons.append("non_finite_value")
    if max_lift < args.lift_height:
        reasons.append("never_lifted_3cm")
    if final_xy > args.max_final_xy:
        reasons.append("final_xy_outside_3.5cm")
    if not args.min_z_above_rack <= final_z_above_rack <= args.max_z_above_rack:
        reasons.append("final_height_outside_rack_band")
    if upright_angle_deg > args.max_upright_angle_deg:
        reasons.append("final_cube_not_upright")
    if actions[-1, 6] < args.gripper_open_threshold:
        reasons.append("left_gripper_not_open_at_end")
    if actions[-1, 13] < args.gripper_open_threshold:
        reasons.append("right_gripper_not_open_at_end")
    if stable_tail < args.min_stable_tail_frames:
        reasons.append("stable_tail_under_minimum")

    return {
        "episode": episode,
        "keep": not reasons,
        "source": "official" if episode < 1000 else "custom",
        "frames": len(actions),
        "reasons": reasons,
        "max_lift": round(max_lift, 6),
        "final_xy": round(final_xy, 6),
        "final_z_above_rack": round(final_z_above_rack, 6),
        "final_upright_angle_deg": round(upright_angle_deg, 6),
        "stable_tail_frames": stable_tail,
        "left_transition_steps": left_transitions,
        "right_transition_steps": right_transitions,
        "final_left_gripper": round(float(actions[-1, 6]), 6),
        "final_right_gripper": round(float(actions[-1, 13]), 6),
        "qf_abs_max": round(float(np.abs(qf).max()), 9),
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = [
        int(json.loads(line)["episode_index"])
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) != int(info["total_episodes"]):
        raise RuntimeError("meta/info.json and meta/episodes.jsonl episode counts disagree")

    rows = []
    for index, episode in enumerate(episodes, 1):
        rows.append(episode_row(dataset, episode, float(info["fps"]), args))
        if index % 100 == 0 or index == len(episodes):
            print(f"[QUALITY] {index}/{len(episodes)}", flush=True)

    kept = [int(row["episode"]) for row in rows if row["keep"]]
    rejected = [row for row in rows if not row["keep"]]
    reason_counts = Counter(reason for row in rejected for reason in row["reasons"])
    left_schedules = {tuple(row["left_transition_steps"]) for row in rows}
    right_schedules = {tuple(row["right_transition_steps"]) for row in rows}
    qf_abs_max = max(float(row["qf_abs_max"]) for row in rows)
    thresholds = {
        "max_final_xy": args.max_final_xy,
        "min_z_above_rack": args.min_z_above_rack,
        "max_z_above_rack": args.max_z_above_rack,
        "max_upright_angle_deg": args.max_upright_angle_deg,
        "lift_height": args.lift_height,
        "stable_speed": args.stable_speed,
        "min_stable_tail_frames": args.min_stable_tail_frames,
        "gripper_open_threshold": args.gripper_open_threshold,
    }
    manifest = {
        "schema_version": 1,
        "task": "sample_loading",
        "source_dataset": str(dataset),
        "source_fingerprint": {
            "info_json_sha256": sha256(info_path),
            "episodes_jsonl_sha256": sha256(episodes_path),
        },
        "thresholds": thresholds,
        "source_episode_count": len(rows),
        "kept_episode_count": len(kept),
        "rejected_episode_count": len(rejected),
        "kept_episodes": kept,
        "rejected_episodes": [int(row["episode"]) for row in rejected],
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "dataset_warnings": {
            "all_qf_zero": qf_abs_max == 0.0,
            "unique_left_gripper_transition_schedules": len(left_schedules),
            "unique_right_gripper_transition_schedules": len(right_schedules),
            "fixed_gripper_timing": len(left_schedules) == 1 and len(right_schedules) == 1,
        },
        "selection_valid": bool(kept) and len(kept) + len(rejected) == len(rows),
    }
    (output_dir / "clean_v1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "kept_episodes.txt").write_text(
        "\n".join(map(str, kept)) + "\n", encoding="utf-8"
    )
    with (output_dir / "episode_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["reasons"] = ";".join(row["reasons"])
            serializable["left_transition_steps"] = json.dumps(row["left_transition_steps"])
            serializable["right_transition_steps"] = json.dumps(row["right_transition_steps"])
            writer.writerow(serializable)
    with (output_dir / "rejected_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = [
        "# sample_loading clean_v1 data quality report",
        "",
        f"- Source: `{dataset}`",
        f"- Kept: **{len(kept)}/{len(rows)}**",
        f"- Rejected: **{len(rejected)}/{len(rows)}**",
        f"- Selection valid: **{manifest['selection_valid']}**",
        "",
        "## Rejection reasons",
        "",
    ]
    report.extend(f"- `{reason}`: {count}" for reason, count in sorted(reason_counts.items()))
    report.extend([
        "",
        "## Dataset-level warnings",
        "",
        f"- qf all zero: `{manifest['dataset_warnings']['all_qf_zero']}`",
        f"- fixed gripper timing: `{manifest['dataset_warnings']['fixed_gripper_timing']}`",
        "",
        "The manifest is a read-only episode selection view. The source dataset was not modified.",
        "",
    ])
    (output_dir / "quality_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "source_episode_count", "kept_episode_count", "rejected_episode_count",
        "rejection_reason_counts", "dataset_warnings", "selection_valid"
    )}, ensure_ascii=False, indent=2))
    print(output_dir / "clean_v1_manifest.json")
    return 0 if manifest["selection_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
