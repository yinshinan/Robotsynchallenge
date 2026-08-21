#!/usr/bin/env python
"""Train ACT from a read-only sample_loading episode quality manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.act.scripts.train import _patch_lerobot_dataset_factory  # noqa: E402


def patch_noncontiguous_episode_indexing(drop_qf: bool = False) -> None:
    """Work around LeRobot v0.3.3 treating source episode ids as subset offsets."""
    import lerobot.scripts.train as train_module

    original_make_dataset = train_module.make_dataset

    def make_dataset_with_episode_map(cfg):
        dataset = original_make_dataset(cfg)
        base_dataset = getattr(dataset, "_base_dataset", dataset)
        if base_dataset.episodes is None:
            return dataset
        source_to_local = {
            int(source_episode): local_episode
            for local_episode, source_episode in enumerate(base_dataset.episodes)
        }
        original_get_query_indices = base_dataset._get_query_indices

        def get_query_indices_with_episode_map(self, idx, source_episode):
            try:
                local_episode = source_to_local[int(source_episode)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Frame references episode {source_episode}, which is absent from the clean manifest"
                ) from exc
            return original_get_query_indices(idx, local_episode)

        base_dataset._get_query_indices = types.MethodType(
            get_query_indices_with_episode_map, base_dataset
        )
        # LeRobot calculates subset statistics on base_dataset.stats, whereas
        # the compatibility alias wrapper normally exposes full-dataset stats.
        if hasattr(dataset, "_aliases") and hasattr(dataset.meta, "_stats"):
            subset_stats = dict(base_dataset.stats)
            for alias_key, source_key in dataset._aliases.items():
                if source_key in subset_stats:
                    subset_stats[alias_key] = subset_stats[source_key]
            dataset.meta._stats = subset_stats
            print("[CLEAN TRAIN] Activated clean-subset normalization statistics", flush=True)
        if drop_qf:
            if hasattr(dataset.meta, "_features"):
                dataset.meta._features.pop("observation.qf", None)
            elif hasattr(dataset.meta, "info"):
                dataset.meta.info.get("features", {}).pop("observation.qf", None)
            else:
                raise RuntimeError("Cannot locate dataset features for --drop-qf")
            if hasattr(dataset.meta, "_stats"):
                dataset.meta._stats.pop("observation.qf", None)
            print(
                "[CLEAN TRAIN] Dropped observation.qf from policy inputs; "
                "recorded qf remains untouched on disk",
                flush=True,
            )
        print(
            f"[CLEAN TRAIN] Installed non-contiguous episode map for "
            f"{len(source_to_local)} episodes",
            flush=True,
        )
        return dataset

    train_module.make_dataset = make_dataset_with_episode_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        help="Optional pretrained_model directory for clean-subset fine-tuning.",
    )
    parser.add_argument("--repo-id")
    parser.add_argument("--job-name")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-obs-steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-imagenet-stats", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--drop-qf",
        action="store_true",
        help="Exclude observation.qf from ACT inputs without modifying the source dataset.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_episodes(dataset_root: Path, manifest_path: Path) -> tuple[list[int], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "sample_loading" or not manifest.get("selection_valid"):
        raise RuntimeError("Manifest is not a valid sample_loading clean selection")
    if Path(manifest["source_dataset"]).resolve() != dataset_root:
        raise RuntimeError("Manifest source_dataset does not match --dataset-root")
    expected = manifest["source_fingerprint"]
    actual = {
        "info_json_sha256": sha256(dataset_root / "meta/info.json"),
        "episodes_jsonl_sha256": sha256(dataset_root / "meta/episodes.jsonl"),
    }
    if actual != expected:
        raise RuntimeError("Source dataset metadata changed after the manifest was built")
    episodes = [int(value) for value in manifest["kept_episodes"]]
    if len(episodes) != len(set(episodes)) or len(episodes) != manifest["kept_episode_count"]:
        raise RuntimeError("Manifest kept episode list is inconsistent")
    if not episodes:
        raise RuntimeError("Manifest selected zero episodes")
    return episodes, manifest


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {output_dir}. "
            "Choose a new path."
        )
    episodes, manifest = validated_episodes(dataset_root, manifest_path)

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    policy_config = ACTConfig(
        device=args.device,
        use_amp=args.use_amp,
        push_to_hub=False,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )
    pretrained_path = None
    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path.expanduser().resolve()
        if not (pretrained_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing pretrained model: {pretrained_path}")
        policy_config.pretrained_path = str(pretrained_path)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id or f"{dataset_root.name}_clean_v1",
            root=str(dataset_root),
            episodes=episodes,
            use_imagenet_stats=not args.no_imagenet_stats,
            video_backend=args.video_backend,
        ),
        policy=policy_config,
        output_dir=output_dir,
        job_name=args.job_name or output_dir.name,
        resume=False,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_checkpoint=not args.no_save_checkpoint,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=False),
    )
    run_provenance = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_dataset": str(dataset_root),
        "selected_episodes": len(episodes),
        "rejected_episodes": manifest["rejected_episode_count"],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "pretrained_path": str(pretrained_path) if pretrained_path else None,
        "drop_qf": args.drop_qf,
        "warnings": manifest["dataset_warnings"],
    }
    print(
        f"[CLEAN TRAIN] selected={len(episodes)} rejected={manifest['rejected_episode_count']} "
        f"seed={args.seed} steps={args.steps}",
        flush=True,
    )
    lerobot_train = _patch_lerobot_dataset_factory()
    patch_noncontiguous_episode_indexing(drop_qf=args.drop_qf)
    lerobot_train(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_subset_provenance.json").write_text(
        json.dumps(run_provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
