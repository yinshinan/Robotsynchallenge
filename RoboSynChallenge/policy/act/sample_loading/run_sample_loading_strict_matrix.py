#!/usr/bin/env python
"""Run the strict sample_loading action-step matrix without recording videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/act_sample_loading_mixed1100_bs8_v0/checkpoints/080000/pretrained_model"
DEFAULT_PYTHON = REPO_ROOT / "policy/act/.venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--action-steps", type=int, nargs="+", default=[8, 10, 12, 15])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep complete entries in an existing output directory and rerun only incomplete ones.",
    )
    return parser.parse_args()


def parse_summary(text: str, action_steps: int, returncode: int) -> dict[str, object]:
    success_matches = re.findall(r"Success Rate:\s*(\d+)/(\d+)", text)
    success_count, episode_count = map(int, success_matches[-1]) if success_matches else (0, 0)
    strict_active = "[STRICT METRICS ACTIVE]" in text
    return {
        "n_action_steps": action_steps,
        "returncode": returncode,
        "strict_metrics_active": strict_active,
        "success_count": success_count,
        "episode_count": episode_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
    }


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or REPO_ROOT / "eval_result/sample_loading/act_strict" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "policy"), str(REPO_ROOT.parent / "EmbodiChain")]
    )

    summary_path = output_dir / "summary.json"
    previous_by_steps: dict[int, dict[str, object]] = {}
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_by_steps = {int(item["n_action_steps"]): item for item in previous}

    summaries = []
    for action_steps in args.action_steps:
        log_path = output_dir / f"n_action_steps_{action_steps}.log"
        previous = previous_by_steps.get(action_steps)
        if (
            previous
            and previous.get("returncode") == 0
            and previous.get("strict_metrics_active") is True
            and previous.get("episode_count") == args.episodes
            and log_path.exists()
        ):
            summaries.append(previous)
            print(
                f"[STRICT MATRIX] Resume: keeping complete n_action_steps={action_steps} "
                f"({previous['success_count']}/{previous['episode_count']})",
                flush=True,
            )
            continue
        command = [
            str(args.python),
            "scripts/eval_policy_sample_loading_strict.py",
            "--config",
            "policy/act/deploy_policy.yml",
            "--overrides",
            "--task_name", "sample_loading",
            "--setting", "random",
            "--checkpoint_path", str(args.checkpoint),
            "--model_name", f"pretrained_model_strict_n{action_steps}",
            "--max_episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--headless", "True",
            "--eval_video_log", "False",
            "--eval_reset_sync_steps", "1",
            "--n_action_steps", str(action_steps),
            "--act_step", "10",
            "--debug_trajectory", "False",
        ]
        print(f"[STRICT MATRIX] Starting n_action_steps={action_steps}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                if any(marker in line for marker in (
                    "[ACT CONFIG]", "[STRICT METRICS ACTIVE]", "Episode 10/", "Episode 20/",
                    "Episode 30/", "Episode 40/", "Episode 50/", "Success Rate:",
                )):
                    print(line.rstrip(), flush=True)
            returncode = process.wait()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        summary = parse_summary(text, action_steps, returncode)
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        summary_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[STRICT MATRIX] Finished n_action_steps={action_steps}: "
            f"{summary['success_count']}/{summary['episode_count']}",
            flush=True,
        )
        if returncode != 0 or not summary["strict_metrics_active"]:
            return returncode or 2
    print(f"[STRICT MATRIX] Summary: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
