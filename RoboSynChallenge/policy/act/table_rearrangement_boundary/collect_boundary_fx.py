#!/usr/bin/env python
"""Collect strict-success table demos at low spoon-y and positive top-camera fx.

The official gym/action configs and task implementation are read-only.  This
entry point writes a temporary patched gym config, saves to an isolated dataset
root, rejects failed expert rollouts, and records reset provenance per attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")

# Some planner paths return CUDA tensors but call Tensor.numpy() directly.
_tensor_numpy = torch.Tensor.numpy


def _numpy_with_cuda_transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any):
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return _tensor_numpy(tensor, *args, **kwargs)


torch.Tensor.numpy = _numpy_with_cuda_transfer

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
for search_path in (REPO_ROOT, REPO_ROOT / "policy", WORKSPACE_ROOT / "EmbodiChain"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import robosynchallenge  # noqa: E402,F401
import scripts.run_env  # noqa: E402,F401
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


# EmbodiChain's current recorder still uses the old metadata_buffer_size name;
# LeRobot 0.3.3 renamed the equivalent create-time batching parameter.
_lerobot_create = LeRobotDataset.create
_lerobot_add_frame = LeRobotDataset.add_frame


def _compatible_lerobot_create(*args: Any, **kwargs: Any):
    metadata_buffer_size = kwargs.pop("metadata_buffer_size", None)
    if metadata_buffer_size is not None:
        kwargs.setdefault("batch_encoding_size", metadata_buffer_size)
    # The recorder adds 600 RGB frames per episode.  Async image writers keep
    # simulator collection from blocking on every temporary PNG write.
    kwargs.setdefault("image_writer_threads", 8)
    return _lerobot_create(*args, **kwargs)


LeRobotDataset.create = _compatible_lerobot_create


def _compatible_add_frame(
    self, frame: dict[str, Any], task: str | None = None, timestamp: float | None = None
):
    frame = dict(frame)
    embedded_task = frame.pop("task", None)
    return _lerobot_add_frame(
        self,
        frame,
        task
        or embedded_task
        or "Pick the fork and the spoon, place them next to the plate.",
        timestamp,
    )


LeRobotDataset.add_frame = _compatible_add_frame

# LeRobot 0.3.3 persists metadata on every save_episode and no longer exposes
# finalize(); retain the old recorder contract as a local no-op.
if not hasattr(LeRobotDataset, "finalize"):
    LeRobotDataset.finalize = lambda self: None


SPOON_Y_RANGE = (-0.30, -0.29)
FOCAL_X_OFFSET_RANGE = (0.0, 50.0)
STRICT_XY_TOLERANCE_M = 0.035
STRICT_Z_TOLERANCE_M = 0.035
MIN_LIFT_M = 0.020
MIN_TARGET_STREAK = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--renderer", choices=["auto", "hybrid", "fast-rt", "rt"], default="auto"
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "lerobot_dataset/table_rearrangement_boundary_fx",
    )
    return parser.parse_args()


def load_patched_config(output_root: Path) -> dict[str, Any]:
    source = REPO_ROOT / "configs/table_rearrangement/random/gym_config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    events = config["env"]["events"]

    spoon_range = events["init_spoon_pose"]["params"]["position_range"]
    spoon_range[0][1], spoon_range[1][1] = SPOON_Y_RANGE
    events["random_camera_high_intrinsics"]["params"]["focal_x_range"] = list(
        FOCAL_X_OFFSET_RANGE
    )
    events["random_robot_qpos"]["params"]["joint_ids"] = list(range(12))

    # Match the reproducible evaluator: utensil poses must be current before
    # distractor avoidance sampling starts.
    distractor = events.pop("randomize_distractor_slots")
    events["randomize_distractor_slots"] = distractor

    dataset_params = config["env"]["dataset"]["lerobot"]["params"]
    dataset_params["save_path"] = str(output_root.resolve())
    dataset_params["extra"] = dict(dataset_params.get("extra", {}))
    dataset_params["extra"]["task_description"] = (
        "table_rearrangement_boundary_y_fxpos"
    )
    return config


def recorder(env):
    manager = env.get_wrapper_attr("dataset_manager")
    for configs in manager._mode_functor_cfgs.values():
        for config in configs:
            functor = config.func
            if hasattr(functor, "curr_episode") and hasattr(functor, "dataset_path"):
                return functor
    raise RuntimeError("No LeRobot recorder found")


def pose_xyz(env, uid: str) -> torch.Tensor:
    obj = env.unwrapped.sim.get_rigid_object(uid)
    return obj.get_local_pose(to_matrix=True)[:, :3, 3]


def reset_with_seed(env, seed: int, save_data: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env.reset(seed=seed, options={"save_data": save_data})


def execute_and_validate(env) -> tuple[bool, dict[str, Any]]:
    fork_initial = pose_xyz(env, "fork").clone()
    spoon_initial = pose_xyz(env, "spoon").clone()
    plate_initial = pose_xyz(env, "plate").clone()
    intrinsics = env.unwrapped.sim.get_sensor("cam_high").get_intrinsics()[0]
    action_list = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
    if action_list is None or len(action_list) == 0:
        return False, {"failure_stage": "expert_action_generation"}

    max_fork_z = fork_initial[:, 2].clone()
    max_spoon_z = spoon_initial[:, 2].clone()
    target_streak = 0
    max_target_streak = 0
    terminated_early = False
    for action in action_list:
        _, _, terminated, truncated, _ = env.step(action)
        fork = pose_xyz(env, "fork")
        spoon = pose_xyz(env, "spoon")
        plate = pose_xyz(env, "plate")
        max_fork_z = torch.maximum(max_fork_z, fork[:, 2])
        max_spoon_z = torch.maximum(max_spoon_z, spoon[:, 2])
        fork_target = plate[:, :2].clone()
        spoon_target = plate[:, :2].clone()
        fork_target[:, 1] += 0.16
        spoon_target[:, 1] -= 0.16
        placement = bool(
            (
                (torch.linalg.vector_norm(fork[:, :2] - fork_target, dim=-1) <= STRICT_XY_TOLERANCE_M)
                & (torch.linalg.vector_norm(spoon[:, :2] - spoon_target, dim=-1) <= STRICT_XY_TOLERANCE_M)
                & (torch.abs(fork[:, 2] - plate[:, 2]) <= STRICT_Z_TOLERANCE_M)
                & (torch.abs(spoon[:, 2] - plate[:, 2]) <= STRICT_Z_TOLERANCE_M)
            )[0].item()
        )
        target_streak = target_streak + 1 if placement else 0
        max_target_streak = max(max_target_streak, target_streak)
        if bool(torch.as_tensor(terminated).any().item()) or bool(
            torch.as_tensor(truncated).any().item()
        ):
            terminated_early = True
            break

    fork_final = pose_xyz(env, "fork")
    spoon_final = pose_xyz(env, "spoon")
    plate_final = pose_xyz(env, "plate")
    fork_lift = float((max_fork_z - fork_initial[:, 2])[0].item())
    spoon_lift = float((max_spoon_z - spoon_initial[:, 2])[0].item())
    fork_target = plate_final[:, :2].clone()
    spoon_target = plate_final[:, :2].clone()
    fork_target[:, 1] += 0.16
    spoon_target[:, 1] -= 0.16
    fork_error = float(
        torch.linalg.vector_norm(fork_final[:, :2] - fork_target, dim=-1)[0].item()
    )
    spoon_error = float(
        torch.linalg.vector_norm(spoon_final[:, :2] - spoon_target, dim=-1)[0].item()
    )
    final_z_ok = bool(
        (
            (torch.abs(fork_final[:, 2] - plate_final[:, 2]) <= STRICT_Z_TOLERANCE_M)
            & (torch.abs(spoon_final[:, 2] - plate_final[:, 2]) <= STRICT_Z_TOLERANCE_M)
        )[0].item()
    )
    success = (
        not terminated_early
        and fork_lift >= MIN_LIFT_M
        and spoon_lift >= MIN_LIFT_M
        and fork_error <= STRICT_XY_TOLERANCE_M
        and spoon_error <= STRICT_XY_TOLERANCE_M
        and final_z_ok
        and max_target_streak >= MIN_TARGET_STREAK
    )
    metrics = {
        "action_steps": len(action_list),
        "initial_fork_xyz_m": fork_initial[0].detach().cpu().tolist(),
        "initial_spoon_xyz_m": spoon_initial[0].detach().cpu().tolist(),
        "initial_plate_xyz_m": plate_initial[0].detach().cpu().tolist(),
        "cam_high_fx": float(intrinsics[0, 0].item()),
        "cam_high_fy": float(intrinsics[1, 1].item()),
        "max_fork_lift_m": fork_lift,
        "max_spoon_lift_m": spoon_lift,
        "max_target_streak": max_target_streak,
        "final_fork_target_xy_error_m": fork_error,
        "final_spoon_target_xy_error_m": spoon_error,
        "final_z_ok": final_z_ok,
        "terminated_early": terminated_early,
        "failure_stage": "success" if success else "strict_validation",
    }
    return success, metrics


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    max_attempts = args.max_attempts or args.episodes * 3
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    patched = load_patched_config(args.output_root)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="table_boundary_fx_", delete=False
        ) as handle:
            json.dump(patched, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)

        launcher_args = argparse.Namespace(
            gym_config=str(temp_path),
            action_config=str(REPO_ROOT / "configs/table_rearrangement/action_config.json"),
            num_envs=1,
            device=args.device,
            headless=args.headless,
            renderer=args.renderer,
            gpu_id=args.gpu_id,
            arena_space=5.0,
            max_episodes=args.episodes,
            filter_visual_rand=False,
            filter_dataset_saving=False,
            preview=False,
        )
        env_cfg, gym_config, action_kwargs = build_env_cfg_from_args(launcher_args)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_kwargs)
    rec = recorder(env)
    dataset_path = Path(rec.dataset_path)
    seed_rng = np.random.RandomState(args.seed)
    attempts: list[dict[str, Any]] = []
    saved: list[dict[str, Any]] = []
    attempt = 0
    try:
        current_seed = int(seed_rng.randint(0, 2**31 - 1))
        reset_with_seed(env, current_seed, save_data=False)
        while rec.curr_episode < args.episodes:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Reached max attempts {max_attempts} with "
                    f"{rec.curr_episode}/{args.episodes} saved"
                )
            success, metrics = execute_and_validate(env)
            record = {
                "attempt": attempt,
                "seed": current_seed,
                "saved": success,
                "saved_episode": rec.curr_episode if success else None,
                **metrics,
            }
            attempts.append(record)
            if success:
                saved.append(record)
            print(
                "[BOUNDARY COLLECT]",
                f"attempt={attempt + 1}",
                f"saved={len(saved)}/{args.episodes}",
                f"seed={current_seed}",
                f"spoon_y={record.get('initial_spoon_xyz_m', [None, None])[1]}",
                f"fx={record.get('cam_high_fx')}",
                f"strict_success={success}",
                flush=True,
            )
            attempt += 1
            next_seed = int(seed_rng.randint(0, 2**31 - 1))
            reset_with_seed(env, next_seed, save_data=success)
            current_seed = next_seed
    finally:
        env.close()

    provenance = {
        "schema_version": 1,
        "source_gym_config": str(
            REPO_ROOT / "configs/table_rearrangement/random/gym_config.json"
        ),
        "source_action_config": str(
            REPO_ROOT / "configs/table_rearrangement/action_config.json"
        ),
        "dataset": str(dataset_path),
        "requested_episodes": args.episodes,
        "saved_episodes": len(saved),
        "attempt_count": len(attempts),
        "seed": args.seed,
        "spoon_y_range_m": list(SPOON_Y_RANGE),
        "cam_high_focal_x_offset_range": list(FOCAL_X_OFFSET_RANGE),
        "cam_high_focal_y_offset_range": [-50.0, 50.0],
        "distractor_event_moved_after_utensil_init": True,
        "strict_thresholds": {
            "xy_tolerance_m": STRICT_XY_TOLERANCE_M,
            "z_tolerance_m": STRICT_Z_TOLERANCE_M,
            "min_lift_m": MIN_LIFT_M,
            "min_target_streak": MIN_TARGET_STREAK,
        },
        "saved_attempts": saved,
        "attempts": attempts,
    }
    dataset_path.mkdir(parents=True, exist_ok=True)
    provenance_path = dataset_path / "boundary_fx_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[BOUNDARY COMPLETE] dataset={dataset_path} "
        f"episodes={len(saved)} attempts={len(attempts)} "
        f"provenance={provenance_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
