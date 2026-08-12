#!/usr/bin/env python3
"""Safely merge local LeRobot v2.1 datasets without modifying the sources."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACT_LEGACY_TO_CANONICAL = {
    "observation.state": "observation.qpos",
    "observation.images.cam_high": "cam_high.color",
    "observation.images.cam_left_wrist": "cam_left_wrist.color",
    "observation.images.cam_right_wrist": "cam_right_wrist.color",
}
ACT_UNUSED_FEATURES = {"left_ee_pose", "right_ee_pose"}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=4)
        f.write("\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def replace_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Missing required parquet column: {name}")
    field = table.schema.field(column_index)
    array = pa.array(values, type=field.type)
    return table.set_column(column_index, field, array)


def as_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def json_value(value: np.ndarray) -> object:
    return value.tolist()


def merge_stats(all_stats: list[dict]) -> dict:
    """Pool count/mean/std exactly and pool quantiles by sample-weighted mean."""
    if not all_stats:
        raise ValueError("No stats.json files found")
    keys = list(all_stats[0])
    if any(set(stats) != set(keys) for stats in all_stats[1:]):
        raise ValueError("Source stats.json files have different feature keys")

    merged: dict[str, dict] = {}
    for key in keys:
        feature_stats = [stats[key] for stats in all_stats]
        counts = np.asarray(
            [float(as_array(stats["count"]).reshape(-1)[0]) for stats in feature_stats],
            dtype=np.float64,
        )
        total_count = float(counts.sum())
        if total_count <= 0:
            raise ValueError(f"Invalid statistics count for feature {key}")

        means = [as_array(stats["mean"]) for stats in feature_stats]
        stds = [as_array(stats["std"]) for stats in feature_stats]
        mean = sum(count * item for count, item in zip(counts, means, strict=True)) / total_count
        variance = sum(
            count * (std**2 + (item_mean - mean) ** 2)
            for count, std, item_mean in zip(counts, stds, means, strict=True)
        ) / total_count

        output: dict[str, object] = {
            "min": json_value(np.minimum.reduce([as_array(stats["min"]) for stats in feature_stats])),
            "max": json_value(np.maximum.reduce([as_array(stats["max"]) for stats in feature_stats])),
            "mean": json_value(mean),
            "std": json_value(np.sqrt(np.maximum(variance, 0.0))),
            "count": [int(total_count)],
        }

        for stat_name in feature_stats[0]:
            if stat_name in output or stat_name == "count":
                continue
            values = [as_array(stats[stat_name]) for stats in feature_stats]
            pooled = sum(count * value for count, value in zip(counts, values, strict=True)) / total_count
            output[stat_name] = json_value(pooled)
        merged[key] = output
    return merged


def normalize_act_key(key: str) -> str:
    return ACT_LEGACY_TO_CANONICAL.get(key, key)


def normalize_act_stats(stats: dict) -> dict:
    normalized: dict = {}
    for key, value in stats.items():
        if key in ACT_UNUSED_FEATURES:
            continue
        new_key = normalize_act_key(key)
        if new_key in normalized:
            raise ValueError(f"ACT normalization produced duplicate statistics key: {new_key}")
        normalized[new_key] = value
    return normalized


def feature_signature(feature: dict) -> dict:
    """Compare storage semantics while allowing cosmetic joint-name differences."""
    return {key: value for key, value in feature.items() if key != "names"}


def normalize_act_info(info: dict, canonical_features: dict | None = None) -> tuple[dict, dict[str, str]]:
    normalized = dict(info)
    features: dict = {}
    output_to_source: dict[str, str] = {}
    for source_key, feature in info["features"].items():
        if source_key in ACT_UNUSED_FEATURES:
            continue
        output_key = normalize_act_key(source_key)
        if output_key in features:
            raise ValueError(f"ACT normalization produced duplicate feature: {output_key}")
        features[output_key] = feature
        output_to_source[output_key] = source_key

    if canonical_features is not None:
        if set(features) != set(canonical_features):
            missing = sorted(set(canonical_features) - set(features))
            extra = sorted(set(features) - set(canonical_features))
            raise ValueError(
                f"ACT feature set mismatch after normalization; missing={missing}, extra={extra}"
            )
        for key in canonical_features:
            if feature_signature(features[key]) != feature_signature(canonical_features[key]):
                raise ValueError(f"ACT feature storage mismatch after normalization: {key}")
        # Use the official/reference metadata, including canonical uppercase joint names.
        features = dict(canonical_features)

    normalized["features"] = features
    return normalized, output_to_source


def normalize_act_table(table: pa.Table) -> pa.Table:
    names = list(table.column_names)
    if "observation.state" in names:
        if "observation.qpos" in names:
            raise ValueError("Parquet contains both observation.state and observation.qpos")
        column_index = table.schema.get_field_index("observation.state")
        field = table.schema.field(column_index).with_name("observation.qpos")
        table = table.set_column(column_index, field, table.column(column_index))
    keep = [name for name in table.column_names if name not in ACT_UNUSED_FEATURES]
    return table.select(keep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Directory containing repo-id folders")
    parser.add_argument("--output", required=True, help="New repo id below --root")
    parser.add_argument(
        "--act-compatible",
        action="store_true",
        help="Normalize official and legacy ACT feature names and omit unused EE-pose fields",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Rewrite all source episodes to one semantic task name (for same-task mixed training)",
    )
    parser.add_argument("sources", nargs="+", help="Source repo ids below --root, in merge order")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = (root / args.output).resolve()
    source_dirs = [(root / source).resolve() for source in args.sources]
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("Output must be a child of --root")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    if len(set(source_dirs)) != len(source_dirs):
        raise ValueError("A source dataset was listed more than once")
    for source_dir in source_dirs:
        if root not in source_dir.parents or not source_dir.is_dir():
            raise FileNotFoundError(f"Invalid source dataset: {source_dir}")

    source_infos = [read_json(path / "meta/info.json") for path in source_dirs]
    if args.act_compatible:
        reference, first_video_map = normalize_act_info(source_infos[0])
        infos = [reference]
        video_key_maps = [first_video_map]
        for info in source_infos[1:]:
            normalized, video_map = normalize_act_info(info, reference["features"])
            infos.append(normalized)
            video_key_maps.append(video_map)
    else:
        infos = source_infos
        video_key_maps = [
            {key: key for key, feature in info["features"].items() if feature.get("dtype") == "video"}
            for info in infos
        ]

    reference = infos[0]
    if reference.get("codebase_version") != "v2.1":
        raise ValueError("This script only supports LeRobot v2.1")
    comparable = ("codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path", "features")
    for source_dir, info in zip(source_dirs[1:], infos[1:], strict=True):
        for key in comparable:
            if info.get(key) != reference.get(key):
                raise ValueError(f"Dataset schema mismatch in {source_dir}: {key}")

    source_episodes = [read_jsonl(path / "meta/episodes.jsonl") for path in source_dirs]
    source_episode_stats = [read_jsonl(path / "meta/episodes_stats.jsonl") for path in source_dirs]
    source_tasks = [read_jsonl(path / "meta/tasks.jsonl") for path in source_dirs]
    source_stats = [read_json(path / "meta/stats.json") for path in source_dirs]
    if args.act_compatible:
        source_stats = [normalize_act_stats(stats) for stats in source_stats]
        source_episode_stats = [
            [
                {**row, "stats": normalize_act_stats(row["stats"])}
                for row in rows
            ]
            for rows in source_episode_stats
        ]

    if args.task_name:
        source_tasks = [[{"task_index": 0, "task": args.task_name}] for _ in source_tasks]
        source_episodes = [
            [{**episode, "tasks": [args.task_name]} for episode in episodes]
            for episodes in source_episodes
        ]

    task_to_index: dict[str, int] = {}
    for tasks in source_tasks:
        for task in tasks:
            task_to_index.setdefault(task["task"], len(task_to_index))
    merged_tasks = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]

    video_keys = [key for key, feature in reference["features"].items() if feature.get("dtype") == "video"]
    chunks_size = int(reference["chunks_size"])
    data_template = reference["data_path"]
    video_template = reference["video_path"]

    reference_columns: list[str] | None = None
    reference_schema: pa.Schema | None = None

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    merged_episodes: list[dict] = []
    merged_episode_stats: list[dict] = []
    next_episode = 0
    next_frame = 0

    try:
        for source_dir, source_info, video_key_map, episodes, episode_stats_rows in zip(
            source_dirs,
            source_infos,
            video_key_maps,
            source_episodes,
            source_episode_stats,
            strict=True,
        ):
            if len(episodes) != len(episode_stats_rows):
                raise ValueError(f"Episode metadata/statistics count mismatch in {source_dir}")
            for episode, episode_stats in zip(episodes, episode_stats_rows, strict=True):
                old_episode = int(episode["episode_index"])
                if int(episode_stats["episode_index"]) != old_episode:
                    raise ValueError(f"Episode statistics index mismatch in {source_dir}: {old_episode}")
                length = int(episode["length"])
                tasks = episode.get("tasks", [])
                if len(tasks) != 1 or tasks[0] not in task_to_index:
                    raise ValueError(f"Episode {old_episode} in {source_dir} has unsupported tasks: {tasks}")

                old_chunk = old_episode // chunks_size
                new_chunk = next_episode // chunks_size
                old_data = source_dir / data_template.format(
                    episode_chunk=old_chunk, episode_index=old_episode
                )
                new_data = temp_dir / data_template.format(
                    episode_chunk=new_chunk, episode_index=next_episode
                )
                table = pq.read_table(old_data)
                if args.act_compatible:
                    table = normalize_act_table(table)
                    if reference_columns is None:
                        reference_columns = list(table.column_names)
                        reference_schema = table.schema
                    else:
                        if set(table.column_names) != set(reference_columns):
                            missing = sorted(set(reference_columns) - set(table.column_names))
                            extra = sorted(set(table.column_names) - set(reference_columns))
                            raise ValueError(
                                f"Parquet schema mismatch after ACT normalization in {old_data}; "
                                f"missing={missing}, extra={extra}"
                            )
                        table = table.select(reference_columns)
                        if table.schema != reference_schema:
                            try:
                                # Collected datasets may store 14-D vectors as
                                # fixed_size_list<float>[14], while official datasets
                                # use list<double>.  Canonicalize every later source
                                # to the first (official) source schema.  Arrow's safe
                                # cast still rejects lossy or structurally incompatible
                                # conversions.
                                table = table.cast(reference_schema, safe=True)
                            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                                raise ValueError(
                                    f"Parquet types cannot be safely converted to the official "
                                    f"schema in {old_data}: {exc}"
                                ) from exc
                if table.num_rows != length:
                    raise ValueError(f"Frame count mismatch in {old_data}: {table.num_rows} != {length}")
                table = replace_column(table, "episode_index", [next_episode] * length)
                table = replace_column(table, "index", list(range(next_frame, next_frame + length)))
                table = replace_column(table, "task_index", [task_to_index[tasks[0]]] * length)
                new_data.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, new_data, compression="snappy")

                for video_key in video_keys:
                    source_video_key = video_key_map.get(video_key)
                    if source_video_key is None:
                        raise ValueError(f"Missing source video mapping for canonical key: {video_key}")
                    old_video = source_dir / source_info["video_path"].format(
                        episode_chunk=old_chunk,
                        video_key=source_video_key,
                        episode_index=old_episode,
                    )
                    new_video = temp_dir / video_template.format(
                        episode_chunk=new_chunk,
                        video_key=video_key,
                        episode_index=next_episode,
                    )
                    if not old_video.is_file():
                        raise FileNotFoundError(f"Missing canonical video: {old_video}")
                    new_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_video, new_video)

                merged_episodes.append(
                    {"episode_index": next_episode, "tasks": tasks, "length": length}
                )
                merged_episode_stats.append(
                    {"episode_index": next_episode, "stats": episode_stats["stats"]}
                )
                next_episode += 1
                next_frame += length

        merged_info = dict(reference)
        merged_info.update(
            {
                "total_episodes": next_episode,
                "total_frames": next_frame,
                "total_tasks": len(merged_tasks),
                "splits": {"train": f"0:{next_episode}"},
                "total_chunks": math.ceil(next_episode / chunks_size),
                "total_videos": next_episode * len(video_keys),
            }
        )
        write_json(temp_dir / "meta/info.json", merged_info)
        write_json(temp_dir / "meta/stats.json", merge_stats(source_stats))
        write_jsonl(temp_dir / "meta/tasks.jsonl", merged_tasks)
        write_jsonl(temp_dir / "meta/episodes.jsonl", merged_episodes)
        write_jsonl(temp_dir / "meta/episodes_stats.jsonl", merged_episode_stats)
        temp_dir.rename(output_dir)
    except Exception:
        print(f"Merge stopped. Partial temporary output was kept for inspection: {temp_dir}")
        raise

    print(f"Merged dataset: {output_dir}")
    print(f"Episodes: {next_episode}")
    print(f"Frames: {next_frame}")
    print(f"Canonical videos: {next_episode * len(video_keys)}")


if __name__ == "__main__":
    main()
