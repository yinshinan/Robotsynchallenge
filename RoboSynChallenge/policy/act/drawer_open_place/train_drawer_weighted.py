#!/usr/bin/env python
"""Fine-tune drawer ACT from checkpoint 080000 with phase/action weighted L1 loss.

This is an isolated training entry point.  It does not modify LeRobot's ACT
implementation or the shared RoboSynChallenge deployment code.  The weighted
loss is attached only to the in-memory policy used by this training process;
saved checkpoints remain ordinary ACT checkpoints.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F


ACTION = "action"
OBS_IMAGES = "observation.images"

LEGACY_FEATURE_ALIASES = {
    "observation.state": "observation.qpos",
    "observation.images.cam_high": "cam_high.color",
    "observation.images.cam_right_wrist": "cam_right_wrist.color",
    "observation.images.cam_left_wrist": "cam_left_wrist.color",
}


class _AliasedDatasetMetadata:
    """Expose the canonical ACT feature names without altering the dataset."""

    def __init__(self, base_meta, aliases):
        self._base_meta = base_meta
        self._aliases = aliases
        alias_sources = set(aliases.values())

        self._features = {
            key: value
            for key, value in base_meta.features.items()
            if key not in alias_sources and value.get("dtype") not in {"image", "video"}
        }
        for alias_key, source_key in aliases.items():
            self._features[alias_key] = copy.deepcopy(base_meta.features[source_key])

        self._stats = dict(base_meta.stats)
        for alias_key, source_key in aliases.items():
            if source_key in base_meta.stats:
                self._stats[alias_key] = base_meta.stats[source_key]

    def __getattr__(self, name):
        return getattr(self._base_meta, name)

    @property
    def features(self):
        return self._features

    @property
    def stats(self):
        return self._stats

    @property
    def image_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self):
        return [key for key, ft in self.features.items() if ft["dtype"] in {"video", "image"}]


class _AliasedLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, aliases):
        self._base_dataset = base_dataset
        self._aliases = aliases
        self.meta = _AliasedDatasetMetadata(base_dataset.meta, aliases)

    def __getattr__(self, name):
        return getattr(self._base_dataset, name)

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx):
        item = self._base_dataset[idx]
        for alias_key, source_key in self._aliases.items():
            if source_key not in item:
                continue
            item[alias_key] = item[source_key]
            source_pad_key = f"{source_key}_is_pad"
            if source_pad_key in item:
                item[f"{alias_key}_is_pad"] = item[source_pad_key]
        return item


def _legacy_aliases_for_dataset(dataset):
    features = dataset.meta.features
    aliases = {
        alias_key: source_key
        for alias_key, source_key in LEGACY_FEATURE_ALIASES.items()
        if alias_key not in features and source_key in features
    }
    if "observation.state" not in aliases and "observation.state" not in features:
        return {}
    return aliases


def _weighted_forward(
    self,
    batch,
    *,
    phase_start: int,
    phase_end: int,
    phase_weight: float,
    left_arm_dims: int,
    left_arm_weight: float,
):
    """ACT forward with per-sample phase weight and per-action-dimension weight."""
    if "frame_index" not in batch:
        raise KeyError("Dataset batch has no 'frame_index'; phase weighting cannot be applied.")

    frame_index = batch["frame_index"].reshape(-1)
    batch = self.normalize_inputs(batch)
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    batch = self.normalize_targets(batch)
    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
    target = batch[ACTION]

    if target.shape[-1] != 14:
        raise ValueError(f"Expected a 14-D action, got shape {tuple(target.shape)}")
    if left_arm_dims != 6:
        raise ValueError(f"This experiment requires action[..., 0:6], got {left_arm_dims=}")

    action_weights = torch.ones(target.shape[-1], device=target.device, dtype=target.dtype)
    action_weights[:left_arm_dims] = left_arm_weight
    valid = (~batch["action_is_pad"]).unsqueeze(-1)
    element_loss = F.l1_loss(target, actions_hat, reduction="none")
    weighted_element_loss = element_loss * action_weights.view(1, 1, -1) * valid

    # Preserve LeRobot ACT's original mean convention (padded entries contribute zero),
    # then apply the phase multiplier independently to every sample in the batch.
    per_sample_l1 = weighted_element_loss.mean(dim=(1, 2))
    sample_phase_weights = torch.where(
        (frame_index >= phase_start) & (frame_index <= phase_end),
        torch.as_tensor(phase_weight, device=target.device, dtype=target.dtype),
        torch.as_tensor(1.0, device=target.device, dtype=target.dtype),
    )
    weighted_l1 = (per_sample_l1 * sample_phase_weights).mean()

    loss_dict = {
        "l1_loss": weighted_l1.item(),
        "phase_fraction": (sample_phase_weights > 1.0).float().mean().item(),
    }
    if self.config.use_vae:
        mean_kld = (
            -0.5
            * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
        ).sum(-1).mean()
        loss_dict["kld_loss"] = mean_kld.item()
        loss = weighted_l1 + mean_kld * self.config.kl_weight
    else:
        loss = weighted_l1

    if not getattr(self, "_drawer_weighting_reported", False):
        print("[drawer weighted] First batch verified:")
        print(f"  action shape: {tuple(target.shape)}")
        print(f"  frame range: {int(frame_index.min())}..{int(frame_index.max())}")
        print(f"  phase samples: {int((sample_phase_weights > 1.0).sum())}/{frame_index.numel()}")
        print(f"  weighted L1: {weighted_l1.item():.6f}")
        if self.config.use_vae:
            print(f"  KL: {mean_kld.item():.6f}, kl_weight: {self.config.kl_weight}")
        self._drawer_weighting_reported = True

    return loss, loss_dict


def _patch_training_factories(args):
    """Patch factories only inside this Python process."""
    import lerobot.scripts.train as train_module

    original_make_dataset = train_module.make_dataset
    original_make_policy = train_module.make_policy

    def make_dataset_with_aliases(cfg):
        dataset = original_make_dataset(cfg)
        aliases = _legacy_aliases_for_dataset(dataset)
        if aliases:
            print("[drawer weighted] Applying in-memory dataset aliases:")
            for alias_key, source_key in aliases.items():
                print(f"  {source_key} -> {alias_key}")
            dataset = _AliasedLeRobotDataset(dataset, aliases)
        return dataset

    def make_weighted_policy(*factory_args, **factory_kwargs):
        policy = original_make_policy(*factory_args, **factory_kwargs)

        def forward(instance, batch):
            return _weighted_forward(
                instance,
                batch,
                phase_start=args.phase_start,
                phase_end=args.phase_end,
                phase_weight=args.phase_weight,
                left_arm_dims=6,
                left_arm_weight=args.left_arm_weight,
            )

        policy.forward = MethodType(forward, policy)
        print("[drawer weighted] Attached training-only weighted loss to ACTPolicy.")
        return policy

    train_module.make_dataset = make_dataset_with_aliases
    train_module.make_policy = make_weighted_policy
    return train_module.train


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args():
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "outputs/act_drawer_open_place_official1000_bs8_v0/checkpoints/080000/pretrained_model",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root
        / "lerobot_dataset/drawer_open_place/cobotmagic_Sim_drawer_open_place_official1000_custom_v0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/act_drawer_weighted_ft_from080000_lr1e6",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--phase-start", type=int, default=50)
    parser.add_argument("--phase-end", type=int, default=150)
    parser.add_argument("--phase-weight", type=float, default=3.0)
    parser.add_argument("--left-arm-weight", type=float, default=2.0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(args.checkpoint / "config.json")
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "model.safetensors")
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(args.dataset_root)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new path; this script will not overwrite it."
        )
    if args.phase_start > args.phase_end:
        raise ValueError("phase-start must be <= phase-end")
    if args.lr <= 0 or args.phase_weight <= 0 or args.left_arm_weight <= 0:
        raise ValueError("lr and loss weights must be positive")


def main():
    args = parse_args()
    _validate_args(args)

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.utils import init_logging

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if policy_cfg.type != "act":
        raise TypeError(f"Expected an ACT checkpoint, got {policy_cfg.type!r}")

    # Hard requirements for this drawer experiment.
    if policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 10:
        raise ValueError(
            "Checkpoint must have chunk_size=50 and n_action_steps=10; "
            f"got {policy_cfg.chunk_size=} and {policy_cfg.n_action_steps=}"
        )
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp
    policy_cfg.push_to_hub = False
    policy_cfg.pretrained_path = args.checkpoint
    policy_cfg.kl_weight = 10.0
    policy_cfg.optimizer_lr = args.lr
    policy_cfg.optimizer_lr_backbone = args.lr

    print("[drawer weighted] Experiment configuration:")
    print(f"  source checkpoint: {args.checkpoint}")
    print(f"  dataset: {args.dataset_root}")
    print(f"  output: {args.output_dir}")
    print("  chunk_size=50, n_action_steps=10")
    print(f"  batch_size={args.batch_size}, lr={args.lr:g}, steps={args.steps}")
    print(
        f"  frame {args.phase_start}..{args.phase_end}: x{args.phase_weight}; "
        f"action[..., 0:6]: x{args.left_arm_weight}; KL x10"
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
        job_name="drawer_weighted_ft_from080000",
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

    init_logging()
    lerobot_train = _patch_training_factories(args)
    lerobot_train(cfg)


if __name__ == "__main__":
    main()
