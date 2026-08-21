#!/usr/bin/env python
"""Third-stage ACT fine-tuning for strict drawer completion.

Starts from the selected drawer dual-phase 005000 checkpoint.  It retains the
grasp and placement/release losses and adds a 320--425 closure phase that
emphasizes right-arm action[7:13] plus genuine right-gripper-open targets.
This is training-only and does not modify shared ACT or deployment code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from policy.act.drawer_open_place.train_drawer_weighted import (
    ACTION,
    OBS_IMAGES,
    _AliasedLeRobotDataset,
    _legacy_aliases_for_dataset,
)


def _strict_completion_forward(
    self,
    batch,
    *,
    grasp_start: int,
    grasp_end: int,
    grasp_weight: float,
    placement_start: int,
    placement_end: int,
    placement_weight: float,
    closure_start: int,
    closure_end: int,
    closure_weight: float,
    left_arm_weight: float,
    release_gripper_weight: float,
    release_target_threshold: float,
    right_arm_closure_weight: float,
    right_gripper_closure_weight: float,
    right_open_target_threshold: float,
):
    if "frame_index" not in batch or ACTION not in batch:
        raise KeyError("Batch must contain frame_index and action")
    frame_index = batch["frame_index"].reshape(-1)
    raw_target = batch[ACTION].clone()

    batch = self.normalize_inputs(batch)
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
    batch = self.normalize_targets(batch)
    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
    target = batch[ACTION]
    if target.shape[-1] != 14 or raw_target.shape != target.shape:
        raise ValueError(
            f"Expected matching [B,T,14] targets, got {raw_target.shape=} {target.shape=}"
        )

    frame_index = frame_index.to(target.device)
    grasp_mask = (frame_index >= grasp_start) & (frame_index <= grasp_end)
    placement_mask = (frame_index >= placement_start) & (frame_index <= placement_end)
    closure_mask = (frame_index >= closure_start) & (frame_index <= closure_end)
    # The requested windows share boundary frame 320.  Use the larger phase
    # weight there instead of multiplying weights at one arbitrary frame.
    sample_weights = torch.ones_like(frame_index, dtype=target.dtype)
    for mask, weight in (
        (grasp_mask, grasp_weight),
        (placement_mask, placement_weight),
        (closure_mask, closure_weight),
    ):
        phase_weight = torch.as_tensor(weight, device=target.device, dtype=target.dtype)
        sample_weights = torch.where(mask, torch.maximum(sample_weights, phase_weight), sample_weights)

    valid = (~batch["action_is_pad"]).unsqueeze(-1)
    raw_target = raw_target.to(device=target.device, dtype=target.dtype)
    element_weights = torch.ones_like(target)
    # Retain the dual-phase left-arm emphasis.
    element_weights[..., :6] = left_arm_weight

    left_release_mask = (
        placement_mask[:, None]
        & (raw_target[..., 6] >= release_target_threshold)
        & (~batch["action_is_pad"])
    )
    element_weights[..., 6] = torch.where(
        left_release_mask,
        torch.as_tensor(release_gripper_weight, device=target.device, dtype=target.dtype),
        element_weights[..., 6],
    )

    right_arm_weight = torch.as_tensor(
        right_arm_closure_weight, device=target.device, dtype=target.dtype
    )
    element_weights[..., 7:13] = torch.where(
        closure_mask[:, None, None],
        right_arm_weight,
        element_weights[..., 7:13],
    )
    right_open_mask = (
        closure_mask[:, None]
        & (raw_target[..., 13] >= right_open_target_threshold)
        & (~batch["action_is_pad"])
    )
    element_weights[..., 13] = torch.where(
        right_open_mask,
        torch.as_tensor(
            right_gripper_closure_weight, device=target.device, dtype=target.dtype
        ),
        element_weights[..., 13],
    )

    element_l1 = F.l1_loss(target, actions_hat, reduction="none")
    per_sample_l1 = (element_l1 * element_weights * valid).mean(dim=(1, 2))
    weighted_l1 = (per_sample_l1 * sample_weights).mean()
    loss_dict = {
        "l1_loss": weighted_l1.item(),
        "grasp_fraction": grasp_mask.float().mean().item(),
        "placement_fraction": placement_mask.float().mean().item(),
        "closure_fraction": closure_mask.float().mean().item(),
        "left_release_target_fraction": left_release_mask.float().mean().item(),
        "right_open_target_fraction": right_open_mask.float().mean().item(),
    }
    if self.config.use_vae:
        mean_kld = (
            -0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
        ).sum(-1).mean()
        loss_dict["kld_loss"] = mean_kld.item()
        loss = weighted_l1 + mean_kld * self.config.kl_weight
    else:
        loss = weighted_l1

    if not getattr(self, "_drawer_strict_weighting_reported", False):
        print("[drawer strict weighted] First batch verified:")
        print(f"  action shape: {tuple(target.shape)}")
        print(f"  frame range: {int(frame_index.min())}..{int(frame_index.max())}")
        print(
            f"  phase samples: grasp={int(grasp_mask.sum())}, "
            f"placement={int(placement_mask.sum())}, closure={int(closure_mask.sum())}"
        )
        print(
            f"  target elements: left_release={int(left_release_mask.sum())}, "
            f"right_open={int(right_open_mask.sum())}"
        )
        print(f"  weighted L1: {weighted_l1.item():.6f}")
        if self.config.use_vae:
            print(f"  KL: {mean_kld.item():.6f}, kl_weight: {self.config.kl_weight}")
        self._drawer_strict_weighting_reported = True
    return loss, loss_dict


def _patch_training_factories(args):
    import lerobot.scripts.train as train_module

    original_make_dataset = train_module.make_dataset
    original_make_policy = train_module.make_policy

    def make_dataset_with_aliases(cfg):
        dataset = original_make_dataset(cfg)
        aliases = _legacy_aliases_for_dataset(dataset)
        if aliases:
            print("[drawer strict weighted] Applying in-memory dataset aliases:")
            for alias_key, source_key in aliases.items():
                print(f"  {source_key} -> {alias_key}")
            dataset = _AliasedLeRobotDataset(dataset, aliases)
        return dataset

    def make_weighted_policy(*factory_args, **factory_kwargs):
        policy = original_make_policy(*factory_args, **factory_kwargs)

        def forward(instance, batch):
            return _strict_completion_forward(
                instance,
                batch,
                grasp_start=args.grasp_start,
                grasp_end=args.grasp_end,
                grasp_weight=args.grasp_weight,
                placement_start=args.placement_start,
                placement_end=args.placement_end,
                placement_weight=args.placement_weight,
                closure_start=args.closure_start,
                closure_end=args.closure_end,
                closure_weight=args.closure_weight,
                left_arm_weight=args.left_arm_weight,
                release_gripper_weight=args.release_gripper_weight,
                release_target_threshold=args.release_target_threshold,
                right_arm_closure_weight=args.right_arm_closure_weight,
                right_gripper_closure_weight=args.right_gripper_closure_weight,
                right_open_target_threshold=args.right_open_target_threshold,
            )

        policy.forward = MethodType(forward, policy)
        print("[drawer strict weighted] Attached training-only three-phase loss.")
        return policy

    train_module.make_dataset = make_dataset_with_aliases
    train_module.make_policy = make_weighted_policy
    return train_module.train


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / (
            "outputs/act_drawer_dual_phase_ft_from020000_lr1e6_20260819/"
            "checkpoints/005000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=root / (
            "lerobot_dataset/drawer_open_place/"
            "cobotmagic_Sim_drawer_open_place_official1000_custom_v0"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/act_drawer_strict_completion_ft_from_dual005_lr1e6_20260819",
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
    parser.add_argument("--grasp-start", type=int, default=50)
    parser.add_argument("--grasp-end", type=int, default=150)
    parser.add_argument("--grasp-weight", type=float, default=2.0)
    parser.add_argument("--placement-start", type=int, default=200)
    parser.add_argument("--placement-end", type=int, default=320)
    parser.add_argument("--placement-weight", type=float, default=3.0)
    parser.add_argument("--closure-start", type=int, default=320)
    parser.add_argument("--closure-end", type=int, default=425)
    parser.add_argument("--closure-weight", type=float, default=3.0)
    parser.add_argument("--left-arm-weight", type=float, default=2.0)
    parser.add_argument("--release-gripper-weight", type=float, default=2.0)
    parser.add_argument("--release-target-threshold", type=float, default=0.9)
    parser.add_argument("--right-arm-closure-weight", type=float, default=2.0)
    parser.add_argument("--right-gripper-closure-weight", type=float, default=2.0)
    parser.add_argument("--right-open-target-threshold", type=float, default=0.9)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for filename in ("config.json", "model.safetensors"):
        if not (args.checkpoint / filename).is_file():
            raise FileNotFoundError(args.checkpoint / filename)
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(args.dataset_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    windows = (
        (args.grasp_start, args.grasp_end),
        (args.placement_start, args.placement_end),
        (args.closure_start, args.closure_end),
    )
    if any(start > end for start, end in windows):
        raise ValueError("Phase start must be <= phase end")
    if not (args.grasp_end < args.placement_start <= args.closure_start <= args.placement_end):
        raise ValueError("Expected ordered grasp, placement, and shared-boundary closure windows")
    weights = (
        args.lr, args.grasp_weight, args.placement_weight, args.closure_weight,
        args.left_arm_weight, args.release_gripper_weight,
        args.right_arm_closure_weight, args.right_gripper_closure_weight,
    )
    if any(value <= 0 for value in weights):
        raise ValueError("Learning rate and all loss weights must be positive")
    for value in (args.release_target_threshold, args.right_open_target_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Open-target thresholds must be in [0,1]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.utils import init_logging

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if policy_cfg.type != "act" or policy_cfg.chunk_size != 50 or policy_cfg.n_action_steps != 10:
        raise ValueError("Expected ACT with chunk_size=50 and n_action_steps=10")
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp
    policy_cfg.push_to_hub = False
    policy_cfg.pretrained_path = args.checkpoint
    policy_cfg.kl_weight = 10.0
    policy_cfg.optimizer_lr = args.lr
    policy_cfg.optimizer_lr_backbone = args.lr

    print("[drawer strict weighted] Experiment configuration:")
    print(f"  source checkpoint: {args.checkpoint}")
    print(f"  dataset: {args.dataset_root}")
    print(f"  output: {args.output_dir}")
    print(f"  steps={args.steps}, batch_size={args.batch_size}, lr={args.lr:g}")
    print(
        f"  grasp {args.grasp_start}..{args.grasp_end} x{args.grasp_weight}; "
        f"placement {args.placement_start}..{args.placement_end} x{args.placement_weight}; "
        f"closure {args.closure_start}..{args.closure_end} x{args.closure_weight}"
    )
    print(
        f"  left arm x{args.left_arm_weight}, left release x{args.release_gripper_weight}; "
        f"closure right arm x{args.right_arm_closure_weight}, "
        f"right open x{args.right_gripper_closure_weight}; KL x10"
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
        job_name="drawer_strict_completion_ft_from_dual005",
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
    _patch_training_factories(args)(cfg)


if __name__ == "__main__":
    main()
