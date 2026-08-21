#!/usr/bin/env python
"""Collect non-destructive sample_loading v2 pilots with varied timing.

This is a separate collection entry point. It monkeypatches the task class only
inside this process; the original task, action bank, configs, and launcher stay
unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


# Preserve Python exceptions and provenance writes during simulator teardown.
os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")

# Current simulator planners may return CUDA tensors while the stock action
# linker calls Tensor.numpy() directly. Keep the compatibility local to this
# standalone process and preserve normal CPU Tensor.numpy() behavior.
_torch_tensor_numpy = torch.Tensor.numpy


def _numpy_with_cuda_transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any):
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return _torch_tensor_numpy(tensor, *args, **kwargs)


torch.Tensor.numpy = _numpy_with_cuda_transfer


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, REPO_ROOT / "policy", WORKSPACE_ROOT / "EmbodiChain"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import robosynchallenge  # noqa: F401,E402
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args  # noqa: E402
import scripts.run_env  # noqa: F401,E402
from embodichain.lab.scripts.run_env import generate_and_execute_action_list  # noqa: E402
from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg  # noqa: E402
from embodichain.lab.gym.envs.managers.observations import get_rigid_object_pose  # noqa: E402
from robosynchallenge.tasks.sample_loading.sample_loading import SampleLoadingEnv  # noqa: E402


XY_THRESHOLD = 0.035
MIN_Z_ABOVE_RACK = 0.040
MAX_Z_ABOVE_RACK = 0.075
VELOCITY_THRESHOLD = 0.050
UPRIGHT_ANGLE_THRESHOLD = np.deg2rad(10.0)
GRIPPER_OPEN_THRESHOLD = 0.040
RELEASE_DISTANCE = 0.220

_base_create_demo = SampleLoadingEnv.create_demo_action_list
_base_initialize_episode = SampleLoadingEnv._initialize_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=["clear", "random"], required=True)
    parser.add_argument(
        "--mode",
        choices=["nominal", "recover_grasp", "recover_handoff", "recover_release", "mixed"],
        default="nominal",
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--v2-seed", type=int, default=1000)
    parser.add_argument("--stable-tail-min", type=int, default=25)
    parser.add_argument("--stable-tail-max", type=int, default=30)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", choices=["auto", "hybrid", "fast-rt", "rt"], default="auto")
    parser.add_argument("--filter-visual-rand", action="store_true")
    return parser.parse_args()


def edge_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: data
        for scope_edges in config["edge"].values()
        for edge in scope_edges
        for name, data in edge.items()
    }


def randomize_action_config(base: dict[str, Any], rng: np.random.Generator) -> tuple[dict, dict]:
    config = copy.deepcopy(base)
    edges = edge_map(config)
    ranges = {
        "init_to_pre1": (30, 42),
        "pre1_to_grasp": (28, 42),
        "rclose0": (18, 30),
        "grasp_to_up1": (28, 45),
        "up1_to_up2": (30, 48),
        "left_init_to_takeover_pre": (44, 65),
        "left_pre_to_takeover": (18, 30),
        "close0": (18, 30),
        "ropen1": (12, 24),
        "left_takeover_to_pre1": (30, 46),
        "left_pre1_to_pre2": (13, 23),
        "left_pre2_to_pre": (13, 23),
        "left_pre_to_place": (13, 23),
        "open1": (12, 24),
        # Ensure enough real simulation after release for a 25-30 frame tail.
        "left_place_to_pre": (28, 38),
        "left_pre_to_pre1": (30, 42),
    }
    durations = {}
    for name, (low, high) in ranges.items():
        if name not in edges:
            raise KeyError(f"Required sample_loading edge is missing: {name}")
        duration = int(rng.integers(low, high + 1))
        edges[name]["duration"] = duration
        durations[name] = duration
    return config, durations


def transition_index(values: torch.Tensor, direction: str) -> int | None:
    if direction == "close":
        hits = torch.nonzero((values[:-1] >= 0.5) & (values[1:] < 0.5), as_tuple=False)
    else:
        hits = torch.nonzero((values[:-1] < 0.5) & (values[1:] >= 0.5), as_tuple=False)
    return int(hits[0, 0].item() + 1) if len(hits) else None


def insert_gripper_retry(
    actions: torch.Tensor, joint_index: int, transition: str, mode: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    values = actions[:, 0, joint_index]
    indices = []
    for index in range(1, len(values)):
        if transition == "close" and values[index - 1] >= 0.5 and values[index] < 0.5:
            indices.append(index)
        if transition == "open" and values[index - 1] < 0.5 and values[index] >= 0.5:
            indices.append(index)
    if not indices:
        raise RuntimeError(f"Cannot find {transition} transition for {mode}")
    insert_at = indices[-1] if transition == "open" else indices[0]
    base = actions[max(0, insert_at - 1)].clone()
    if transition == "close":
        # Partial close, release, then let the original full close retry.
        profile = torch.cat([
            torch.linspace(float(base[0, joint_index]), 0.25, 6),
            torch.full((5,), 0.25),
            torch.linspace(0.25, 1.0, 7),
            torch.ones(5),
        ])
    else:
        # Partial release, return to hold, then let the original full release retry.
        profile = torch.cat([
            torch.linspace(float(base[0, joint_index]), 0.65, 6),
            torch.full((5,), 0.65),
            torch.linspace(0.65, 0.0, 7),
            torch.zeros(5),
        ])
    retry = base.unsqueeze(0).repeat(len(profile), 1, 1)
    retry[:, 0, joint_index] = profile.to(device=actions.device, dtype=actions.dtype)
    output = torch.cat([actions[:insert_at], retry, actions[insert_at:]], dim=0)
    return output, {
        "recovery_mode": mode,
        "insert_at": insert_at,
        "inserted_frames": len(profile),
        "joint_index": joint_index,
    }


MIXED_MODE_CYCLE = (
    "nominal",
    "recover_grasp",
    "nominal",
    "recover_handoff",
    "nominal",
    "recover_release",
    "recover_grasp",
    "nominal",
    "recover_handoff",
    "nominal",
    "recover_grasp",
    "nominal",
    "recover_release",
    "nominal",
    "recover_handoff",
    "recover_grasp",
    "nominal",
    "recover_release",
    "recover_grasp",
    "recover_handoff",
)


def choose_mode(requested: str, saved_episode: int, seed_base: int) -> str:
    if requested != "mixed":
        return requested
    # Bind the requested recovery class to the episode being filled, rather
    # than to an attempt. Harder modes are retried until they pass instead of
    # being silently underrepresented in the saved dataset.
    slot = (saved_episode + seed_base) % len(MIXED_MODE_CYCLE)
    return MIXED_MODE_CYCLE[slot]


def randomized_create_demo(self: SampleLoadingEnv, *args: Any, **kwargs: Any):
    attempt = int(getattr(self, "_v2_attempt", 0))
    seed = int(self._v2_seed_base + attempt)
    rng = np.random.default_rng(seed)
    config, durations = randomize_action_config(self._v2_base_action_config, rng)
    self.action_config = config
    self._v2_stable_target = int(
        rng.integers(self._v2_stable_min, self._v2_stable_max + 1)
    )
    mode = choose_mode(
        self._v2_requested_mode,
        int(getattr(self, "_v2_target_episode", 0)),
        self._v2_seed_base,
    )
    spec = {
        "attempt": attempt,
        "seed": seed,
        "setting": self._v2_setting,
        "requested_mode": self._v2_requested_mode,
        "actual_mode": mode,
        "stable_target": self._v2_stable_target,
        "durations": durations,
    }
    self._v2_current_spec = spec
    self._v2_attempt = attempt + 1
    actions = _base_create_demo(self, *args, **kwargs)
    if actions is None:
        spec["action_frames"] = 0
        spec["generation_valid"] = False
        return None
    recovery = {"recovery_mode": "nominal", "inserted_frames": 0}
    if mode == "recover_grasp":
        actions, recovery = insert_gripper_retry(actions, 13, "close", mode)
    elif mode == "recover_handoff":
        actions, recovery = insert_gripper_retry(actions, 6, "close", mode)
    elif mode == "recover_release":
        actions, recovery = insert_gripper_retry(actions, 6, "open", mode)
    spec["action_frames"] = int(len(actions))
    spec["generation_valid"] = True
    spec["recovery"] = recovery
    print("[V2 SCHEDULE] " + json.dumps(spec, ensure_ascii=False), flush=True)
    return actions


def reset_v2_state(self: SampleLoadingEnv) -> None:
    self._v2_stable_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._v2_last_eval_step = -1
    self._v2_last_cube_position = None
    self._v2_fd_speed = torch.full(
        (self.num_envs,), float("inf"), dtype=torch.float32, device=self.device
    )
    self._v2_last_frame_ok = torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device
    )


def v2_initialize_episode(self: SampleLoadingEnv, *args: Any, **kwargs: Any) -> None:
    _base_initialize_episode(self, *args, **kwargs)
    reset_v2_state(self)


def v2_evaluate_task_state(self: SampleLoadingEnv):
    if not hasattr(self, "_v2_stable_count"):
        reset_v2_state(self)
    cube = self.sim.get_rigid_object("cube")
    rack = self.sim.get_rigid_object("rack")
    if cube is None or rack is None or cube.body_data is None:
        raise RuntimeError("sample_loading v2 requires cube/rack rigid-body data")
    cube_pose = cube.get_local_pose(to_matrix=True)
    rack_pose = rack.get_local_pose(to_matrix=True)
    cube_position = cube_pose[:, :3, 3]
    rack_position = rack_pose[:, :3, 3]
    velocity = cube.body_data.lin_vel
    if velocity is None:
        raise RuntimeError("sample_loading v2 requires cube linear velocity")
    left_pose = self.robot.get_link_pose("left_link6")
    right_pose = self.robot.get_link_pose("right_link6")
    if left_pose is None or right_pose is None:
        raise RuntimeError("sample_loading v2 requires both link6 poses")
    left_distance = torch.linalg.vector_norm(cube_position - left_pose[:, :3], dim=-1)
    right_distance = torch.linalg.vector_norm(cube_position - right_pose[:, :3], dim=-1)
    qpos = self.robot.get_qpos()
    left_ids = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
    right_ids = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)
    left_q = qpos[:, left_ids].mean(dim=-1)
    right_q = qpos[:, right_ids].mean(dim=-1)
    released = (
        ((left_q > GRIPPER_OPEN_THRESHOLD) | (left_distance > RELEASE_DISTANCE))
        & ((right_q > GRIPPER_OPEN_THRESHOLD) | (right_distance > RELEASE_DISTANCE))
    )
    rack_xy = torch.linalg.vector_norm(cube_position[:, :2] - rack_position[:, :2], dim=-1)
    z_above = cube_position[:, 2] - rack_position[:, 2]
    body_speed = torch.linalg.vector_norm(velocity, dim=-1)
    upright = torch.arccos(torch.clamp(cube_pose[:, 2, 2], -1.0, 1.0))
    elapsed = int(self._elapsed_steps[0].item())
    if elapsed != self._v2_last_eval_step:
        self._v2_last_eval_step = elapsed
        if self._v2_last_cube_position is None:
            fd_speed = torch.full_like(body_speed, float("inf"))
        else:
            # Match the offline manifest exactly: ||p[t] - p[t-1]|| * dataset FPS.
            # The simulator's reported rigid-body velocity can briefly disagree
            # with recorded pose deltas around release and previously allowed
            # episodes whose persisted tail was shorter than the online count.
            fd_speed = torch.linalg.vector_norm(
                cube_position - self._v2_last_cube_position, dim=-1
            ) * float(self._v2_dataset_fps)
        self._v2_last_cube_position = cube_position.clone()
        frame_ok = (
            (rack_xy < XY_THRESHOLD)
            & (z_above >= MIN_Z_ABOVE_RACK)
            & (z_above <= MAX_Z_ABOVE_RACK)
            & (body_speed < VELOCITY_THRESHOLD)
            & (fd_speed < VELOCITY_THRESHOLD)
            & (upright < UPRIGHT_ANGLE_THRESHOLD)
            & released
        )
        self._v2_fd_speed = fd_speed
        self._v2_last_frame_ok = frame_ok
        self._v2_stable_count = torch.where(
            frame_ok, self._v2_stable_count + 1, torch.zeros_like(self._v2_stable_count)
        )
    else:
        fd_speed = self._v2_fd_speed
        frame_ok = self._v2_last_frame_ok
    target = int(
        getattr(self, "_v2_stable_target", getattr(self, "_v2_stable_min", 25))
    )
    success = self._v2_stable_count >= target
    return success, {}, {
        "cube_xy_dist": rack_xy,
        "cube_z": cube_position[:, 2],
        "rack_z": rack_position[:, 2],
        "cube_lin_vel_norm": body_speed,
        "cube_fd_speed_norm": fd_speed,
        "cube_vertical_angle": upright,
        "strict_released": released,
        "place_stable_count": self._v2_stable_count,
        "v2_stable_target": torch.full_like(self._v2_stable_count, target),
        "placement_ok_single_frame": frame_ok,
    }


def recorder(env: Any):
    manager = env.get_wrapper_attr("dataset_manager")
    for configs in manager._mode_functor_cfgs.values():
        for config in configs:
            functor = config.func
            if hasattr(functor, "curr_episode") and hasattr(functor, "dataset_path"):
                return functor
    raise RuntimeError("No LeRobot recorder found")


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if (
        args.stable_tail_min < 20
        or args.stable_tail_max < args.stable_tail_min
        or args.stable_tail_max > 30
    ):
        raise ValueError("stable tail range must satisfy 20 <= min <= max <= 30")
    launcher_args = argparse.Namespace(
        gym_config=str(REPO_ROOT / f"configs/sample_loading/{args.setting}/gym_config.json"),
        action_config=str(REPO_ROOT / "configs/sample_loading/action_config.json"),
        num_envs=1,
        device=args.device,
        headless=args.headless,
        renderer=args.renderer,
        gpu_id=args.gpu_id,
        arena_space=5.0,
        max_episodes=args.episodes,
        filter_visual_rand=args.filter_visual_rand,
        filter_dataset_saving=False,
        preview=False,
    )
    env_cfg, gym_config, action_kwargs = build_env_cfg_from_args(launcher_args)
    # The stock clear config omits object poses, but strict offline auditing
    # requires them. Add them only to this process's parsed config.
    for observation_name, uid in (("rack_pose", "rack"), ("cube_pose", "cube")):
        if not hasattr(env_cfg.observations, observation_name):
            setattr(
                env_cfg.observations,
                observation_name,
                ObservationCfg(
                    func=get_rigid_object_pose,
                    mode="add",
                    name=observation_name,
                    params={"entity_cfg": SceneEntityCfg(uid=uid)},
                ),
            )
    output_root = REPO_ROOT / "lerobot_dataset/sample_loading_v2"
    dataset_cfg = env_cfg.dataset.lerobot
    dataset_cfg.params = dict(dataset_cfg.params)
    dataset_cfg.params["save_path"] = str(output_root)
    dataset_cfg.params["extra"] = dict(dataset_cfg.params.get("extra", {}))
    dataset_cfg.params["extra"]["task_description"] = (
        f"sample_loading_v2_{args.setting}_{args.mode}"
    )
    physics_config = gym_config.get("physics", {})
    if "enable_ccd" in physics_config:
        env_cfg.sim_cfg.physics_config.enable_ccd = bool(physics_config["enable_ccd"])

    SampleLoadingEnv.create_demo_action_list = randomized_create_demo
    SampleLoadingEnv._initialize_episode = v2_initialize_episode
    SampleLoadingEnv._evaluate_task_state = v2_evaluate_task_state
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_kwargs)
    task = env.unwrapped
    rec = recorder(env)
    task._v2_base_action_config = copy.deepcopy(action_kwargs["action_config"])
    task._v2_seed_base = args.v2_seed
    task._v2_attempt = 0
    task._v2_setting = args.setting
    task._v2_requested_mode = args.mode
    task._v2_stable_min = args.stable_tail_min
    task._v2_stable_max = args.stable_tail_max
    task._v2_dataset_fps = float(rec.dataset.meta.info["fps"])
    reset_v2_state(task)
    dataset_path = Path(rec.dataset_path)
    max_attempts = args.max_attempts or max(args.episodes * 20, args.episodes + 10)
    saved_specs = []
    attempt_records = []
    try:
        env.reset(options={"save_data": False})
        while rec.curr_episode < args.episodes:
            if task._v2_attempt >= max_attempts:
                raise RuntimeError(
                    f"Reached max attempts ({max_attempts}) with {rec.curr_episode}/{args.episodes} saved"
                )
            before = rec.curr_episode
            task._v2_target_episode = before
            valid = generate_and_execute_action_list(env, 0, False)
            if not valid:
                spec = dict(getattr(task, "_v2_current_spec", {}))
                spec["saved"] = False
                spec["failure_stage"] = "expert_action_generation"
                attempt_records.append(spec)
                env.reset(options={"save_data": False})
                continue
            before_reset = rec.curr_episode
            env.reset(options={"save_data": before_reset < args.episodes})
            after = rec.curr_episode
            spec = dict(getattr(task, "_v2_current_spec", {}))
            spec["saved"] = after > before
            spec["saved_episode"] = before if after > before else None
            attempt_records.append(spec)
            if after > before:
                saved_specs.append(spec)
                print(f"[V2 SAVED] {after}/{args.episodes}", flush=True)
        # Clear any post-success residual actions so finalize cannot append an invalid episode.
        env.reset(options={"save_data": False})
    finally:
        env.close()

    dataset_path.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "setting": args.setting,
        "requested_mode": args.mode,
        "v2_seed": args.v2_seed,
        "stable_tail_range": [args.stable_tail_min, args.stable_tail_max],
        "mixed_mode_cycle": list(MIXED_MODE_CYCLE) if args.mode == "mixed" else None,
        "requested_episodes": args.episodes,
        "saved_episodes": len(saved_specs),
        "attempts": len(attempt_records),
        "qf_policy": "recorded_for_compatibility_but_must_be_dropped_from_training",
        "audit_observations": ["cube_pose", "rack_pose"],
        "source_action_config": str(REPO_ROOT / "configs/sample_loading/action_config.json"),
        "saved_schedules": saved_specs,
        "attempt_records": attempt_records,
    }
    (dataset_path / "sample_loading_v2_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[V2 COMPLETE] dataset={dataset_path} episodes={len(saved_specs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
