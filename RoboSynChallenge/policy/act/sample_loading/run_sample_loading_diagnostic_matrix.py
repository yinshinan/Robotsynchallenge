#!/usr/bin/env python
"""Run and summarize the reproducible sample_loading ACT evaluation matrix."""

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
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/act_sample_loading_mixed1100_bs8_v0/checkpoints/080000/pretrained_model"
)
DEFAULT_PYTHON = REPO_ROOT / "policy/act/.venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--setting", default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--action-steps", type=int, nargs="+", default=[50, 10, 5, 1])
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_log(text: str, action_steps: int, returncode: int) -> dict[str, object]:
    success_matches = re.findall(r"Success Rate:\s*(\d+)/(\d+)", text)
    success_count, episode_count = (0, 0)
    if success_matches:
        success_count, episode_count = map(int, success_matches[-1])

    episode_rows = re.findall(
        r"^\s*(\d+)\s*\|\s*(SUCCESS|FAIL)\s*\|\s*(\d+)\s*\|\s*(\d+)",
        text,
        flags=re.MULTILINE,
    )
    final_metrics = []
    for match in re.finditer(
        r"\[DONE DEBUG\].*?cube_xy_dist': tensor\(\[([-+0-9.eE]+)\]\).*?"
        r"cube_z': tensor\(\[([-+0-9.eE]+)\]\).*?"
        r"left_gripper_q_mean': tensor\(([-+0-9.eE]+)\).*?"
        r"right_gripper_q_mean': tensor\(([-+0-9.eE]+)\)",
        text,
    ):
        final_metrics.append(
            {
                "cube_xy_dist": float(match.group(1)),
                "cube_z": float(match.group(2)),
                "left_gripper_q_mean": float(match.group(3)),
                "right_gripper_q_mean": float(match.group(4)),
            }
        )

    initial_xy = [
        float(value)
        for value in re.findall(
            r"\[sample_loading state\] step=0 .*?xy=([-+0-9.eE]+)", text
        )
    ]
    deduplicated_initial_xy = initial_xy[::2] if len(initial_xy) >= 2 else initial_xy

    return {
        "n_action_steps": action_steps,
        "returncode": returncode,
        "success_count": success_count,
        "episode_count": episode_count,
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "initial_cube_xy_dist": deduplicated_initial_xy,
        "final_metrics": final_metrics,
        "episodes": [
            {
                "episode": int(row[0]),
                "result": row[1],
                "action_steps": int(row[2]),
                "inference_calls": int(row[3]),
            }
            for row in episode_rows
        ],
    }


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (
        REPO_ROOT / "eval_result/sample_loading/act_diagnostic" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    environment["PYTHONUNBUFFERED"] = "1"
    python_paths = [
        str(REPO_ROOT),
        str(REPO_ROOT / "policy"),
        str(REPO_ROOT.parent / "EmbodiChain"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    summaries = []
    for action_steps in args.action_steps:
        log_path = output_dir / f"n_action_steps_{action_steps}.log"
        command = [
            str(args.python),
            "scripts/eval_policy_reproducible.py",
            "--config",
            "policy/act/deploy_policy.yml",
            "--overrides",
            "--task_name",
            "sample_loading",
            "--setting",
            args.setting,
            "--checkpoint_path",
            str(args.checkpoint),
            "--model_name",
            f"pretrained_model_n{action_steps}",
            "--max_episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--headless",
            "True",
            "--eval_video_log",
            str(not args.no_video),
            "--n_action_steps",
            str(action_steps),
            "--act_step",
            "10",
            "--debug_trajectory",
            "False",
        ]
        print(f"[MATRIX] Starting n_action_steps={action_steps}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                if any(
                    marker in line
                    for marker in (
                        "[ACT CONFIG]",
                        "[sample_loading state] step=0",
                        "Episode 01/",
                        "Evaluation Results Summary",
                        "Success Rate:",
                    )
                ):
                    print(line.rstrip(), flush=True)
            returncode = process.wait()

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        summary = parse_log(log_text, action_steps, returncode)
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[MATRIX] Finished n_action_steps={action_steps}: "
            f"{summary['success_count']}/{summary['episode_count']}",
            flush=True,
        )
        if returncode != 0:
            print(f"[MATRIX] Aborting after return code {returncode}", file=sys.stderr)
            return returncode

    print(f"[MATRIX] Summary: {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
