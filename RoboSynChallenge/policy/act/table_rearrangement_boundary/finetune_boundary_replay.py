#!/usr/bin/env python
"""Fine-tune table ACT from the 080000 checkpoint on mixed replay data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/080000/pretrained_model",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/act_table_boundary_fx_ft10k_lr1e6_v1",
    )
    parser.add_argument("--repo-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=2_500)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for filename in ("config.json", "model.safetensors"):
        if not (args.checkpoint / filename).is_file():
            raise FileNotFoundError(args.checkpoint / filename)
    for filename in ("meta/info.json", "meta/stats.json", "meta/episodes.jsonl"):
        if not (args.dataset_root / filename).is_file():
            raise FileNotFoundError(args.dataset_root / filename)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    if args.steps <= 0 or args.save_freq <= 0 or args.lr <= 0:
        raise ValueError("steps, save-freq and lr must be positive")


def main() -> int:
    args = parse_args()
    validate(args)

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.utils import init_logging
    from policy.act.scripts.train import _patch_lerobot_dataset_factory

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if (
        policy_cfg.type != "act"
        or policy_cfg.chunk_size != 50
        or policy_cfg.n_action_steps != 50
    ):
        raise ValueError("Expected ACT checkpoint with chunk_size=n_action_steps=50")
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp
    policy_cfg.push_to_hub = False
    policy_cfg.pretrained_path = str(args.checkpoint)
    policy_cfg.optimizer_lr = args.lr
    policy_cfg.optimizer_lr_backbone = args.lr

    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    print("[TABLE BOUNDARY FT] Experiment configuration:", flush=True)
    print(f"  checkpoint={args.checkpoint}", flush=True)
    print(
        f"  dataset={args.dataset_root} episodes={info['total_episodes']} "
        f"frames={info['total_frames']}",
        flush=True,
    )
    print(
        f"  output={args.output_dir} steps={args.steps} batch={args.batch_size} "
        f"lr={args.lr:g}",
        flush=True,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id or args.dataset_root.name,
            root=str(args.dataset_root),
            use_imagenet_stats=True,
            video_backend=args.video_backend,
        ),
        policy=policy_cfg,
        output_dir=args.output_dir,
        job_name=args.output_dir.name,
        resume=False,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=not args.no_save_checkpoint,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=False),
    )
    provenance = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_model_sha256": sha256(args.checkpoint / "model.safetensors"),
        "dataset": str(args.dataset_root),
        "dataset_info_sha256": sha256(args.dataset_root / "meta/info.json"),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "chunk_size": 50,
        "n_action_steps": 50,
    }

    init_logging()
    lerobot_train = _patch_lerobot_dataset_factory()
    lerobot_train(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "boundary_finetune_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
