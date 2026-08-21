#!/usr/bin/env python
"""Wait for a v2 batch, then run the read-only validation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), **values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("[PIPELINE RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_summary(source: Path, audit_dir: Path, smoke_output: Path) -> dict:
    info = load_json(source / "meta/info.json")
    provenance = load_json(source / "sample_loading_v2_provenance.json")
    manifest = load_json(audit_dir / "clean_v1_manifest.json")
    kept = set(int(value) for value in manifest["kept_episodes"])
    totals: Counter[str] = Counter()
    retained: Counter[str] = Counter()
    durations: set[tuple[tuple[str, int], ...]] = set()
    stable_targets: list[int] = []
    for schedule in provenance["saved_schedules"]:
        mode = str(schedule["actual_mode"])
        episode = int(schedule["saved_episode"])
        totals[mode] += 1
        if episode in kept:
            retained[mode] += 1
        durations.add(tuple(sorted((str(k), int(v)) for k, v in schedule["durations"].items())))
        stable_targets.append(int(schedule["stable_target"]))
    modes = {
        mode: {"total": totals[mode], "kept": retained[mode]}
        for mode in sorted(totals)
    }
    return {
        "source_dataset": str(source),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "attempts": int(provenance["attempts"]),
        "mode_retention": modes,
        "unique_duration_schedules": len(durations),
        "stable_target_min": min(stable_targets),
        "stable_target_max": max(stable_targets),
        "quality": {
            "kept": int(manifest["kept_episode_count"]),
            "rejected": int(manifest["rejected_episode_count"]),
            "rejection_reason_counts": manifest["rejection_reason_counts"],
            "warnings": manifest["dataset_warnings"],
            "selection_valid": bool(manifest["selection_valid"]),
        },
        "smoke_training": load_json(smoke_output / "clean_subset_provenance.json"),
    }


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    smoke_output = args.smoke_output.expanduser().resolve()
    status_path = audit_dir / "pipeline_status.json"
    provenance_path = source / "sample_loading_v2_provenance.json"

    while True:
        info_path = source / "meta/info.json"
        info = load_json(info_path) if info_path.is_file() else {}
        episodes = int(info.get("total_episodes", 0))
        frames = int(info.get("total_frames", 0))
        alive = process_alive(args.collector_pid)
        write_status(
            status_path,
            state="collecting",
            episodes=episodes,
            expected_episodes=args.expected_episodes,
            frames=frames,
            collector_alive=alive,
        )
        print(
            f"[PIPELINE WAIT] episodes={episodes}/{args.expected_episodes} "
            f"frames={frames} collector_alive={alive}",
            flush=True,
        )
        if episodes == args.expected_episodes and provenance_path.is_file():
            break
        if not alive:
            write_status(
                status_path,
                state="failed",
                error="collector exited before finalized batch provenance was available",
                episodes=episodes,
                expected_episodes=args.expected_episodes,
                frames=frames,
            )
            raise RuntimeError("Collector exited before the complete batch was finalized")
        time.sleep(args.poll_seconds)

    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Refusing to overwrite destination: {destination}")
    if smoke_output.exists() and any(smoke_output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite smoke output: {smoke_output}")

    write_status(status_path, state="converting", episodes=args.expected_episodes)
    run([
        str(REPO_ROOT.parent / ".venv/bin/python"),
        "policy/act/sample_loading/convert_sample_loading_v2_readonly.py",
        str(source),
        str(destination),
    ])
    write_status(status_path, state="auditing", episodes=args.expected_episodes)
    run([
        str(REPO_ROOT / "policy/act/.venv/bin/python"),
        "policy/act/sample_loading/build_sample_loading_quality_manifest.py",
        str(destination),
        str(audit_dir),
        "--min-stable-tail-frames",
        "25",
    ])
    write_status(status_path, state="smoke_training", episodes=args.expected_episodes)
    run([
        str(REPO_ROOT / "policy/act/.venv/bin/python"),
        "policy/act/sample_loading/train_sample_loading_clean_subset.py",
        "--dataset-root",
        str(destination),
        "--manifest",
        str(audit_dir / "clean_v1_manifest.json"),
        "--output-dir",
        str(smoke_output),
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--log-freq",
        "1",
        "--save-freq",
        "1",
        "--eval-freq",
        "0",
        "--no-save-checkpoint",
        "--drop-qf",
    ])
    summary = build_summary(source, audit_dir, smoke_output)
    (audit_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_status(
        status_path,
        state="complete",
        episodes=args.expected_episodes,
        kept=summary["quality"]["kept"],
        rejected=summary["quality"]["rejected"],
        summary=str(audit_dir / "batch_summary.json"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("[PIPELINE COMPLETE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
