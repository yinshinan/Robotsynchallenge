#!/usr/bin/env python
"""Run item_assembly seeds in isolated simulator processes.

The task's relative reset events can retain state across episodes.  Starting a
fresh process per seed guarantees that the environment, ACT queues, robot and
physics scene all begin from construction state.  This file only calls the
non-invasive evaluator in this directory and writes logs/manifest outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SEEDS = [209652396, 398764591, 924231285, 1478610112, 441365315]


def classify_result(exit_code: int, video: Path | None) -> str:
    """Distinguish simulator/runtime crashes from completed task failures."""
    if exit_code != 0 or video is None:
        return "runtime_error"
    if video.name.endswith("_success.mp4"):
        return "success"
    return "task_failure"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "outputs/act_item_assembly_official1000_custom_v0_bs8_v0/"
            "checkpoints/080000/pretrained_model"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--config",
        default=str(HERE / "deploy_policy_fixed.yml"),
    )
    return parser.parse_args()


def newest_seed_video(seed: int, started_at: float) -> Path | None:
    root = REPO_ROOT / "eval_result" / "item_assembly"
    candidates = [
        path
        for path in root.glob(f"**/episode_000_seed_{seed}_*.mp4")
        if path.stat().st_mtime >= started_at - 1.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def main() -> int:
    args = parse_args()
    run_dir = (
        REPO_ROOT
        / "eval_result"
        / "item_assembly_isolated"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []

    for index, seed in enumerate(args.seeds):
        log_path = run_dir / f"episode_{index:03d}_seed_{seed}.log"
        command = [
            sys.executable,
            str(HERE / "eval_item_assembly_fixed.py"),
            "--config",
            args.config,
            "--overrides",
            "--checkpoint_path",
            args.checkpoint_path,
            "--eval_fixed_episode_seed",
            str(seed),
            "--max_episodes",
            "1",
            "--eval_video_log",
            "True",
        ]
        print(f"[{index + 1}/{len(args.seeds)}] seed={seed} starting", flush=True)
        started_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        video = newest_seed_video(seed, started_at)
        status = classify_result(completed.returncode, video)
        success = status == "success"
        result = {
            "episode": index,
            "seed": seed,
            "status": status,
            "success": success,
            "completed_episode": video is not None,
            "exit_code": completed.returncode,
            "video": str(video.relative_to(REPO_ROOT)) if video else None,
            "log": str(log_path.relative_to(REPO_ROOT)),
        }
        results.append(result)
        print(
            f"[{index + 1}/{len(args.seeds)}] seed={seed} "
            f"result={status.upper()}",
            flush=True,
        )

    manifest = {
        "isolated_process_per_seed": True,
        "success_count": sum(item["success"] for item in results),
        "completed_episode_count": sum(
            item["completed_episode"] for item in results
        ),
        "runtime_error_count": sum(
            item["status"] == "runtime_error" for item in results
        ),
        "episode_count": len(results),
        "results": results,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path.relative_to(REPO_ROOT)}", flush=True)
    return 0 if all(item["exit_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
