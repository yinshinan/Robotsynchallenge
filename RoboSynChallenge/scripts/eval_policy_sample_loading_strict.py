#!/usr/bin/env python
"""Evaluate sample_loading with strict, fail-closed completion metrics.

This is a separate entry point. It monkeypatches the registered task class in
memory and does not modify the task or the unified evaluator source files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, REPO_ROOT / "policy", WORKSPACE_ROOT / "EmbodiChain"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import eval_policy as base_evaluator  # noqa: E402
from embodichain.utils import set_seed  # noqa: E402
from robosynchallenge.tasks.sample_loading.sample_loading import (  # noqa: E402
    SampleLoadingEnv,
)


XY_THRESHOLD = 0.035
MIN_CUBE_Z_ABOVE_RACK = 0.04
MAX_CUBE_Z_ABOVE_RACK = 0.075
VELOCITY_THRESHOLD = 0.05
RELEASE_DISTANCE = 0.22
GRIPPER_OPEN_THRESHOLD = 0.04
UPRIGHT_ANGLE_THRESHOLD = 0.1745
STABLE_STEPS = 8
APPROACH_DISTANCE = 0.20
HOLD_DISTANCE = 0.22


_base_reset = base_evaluator.RecordingEnvProxy.reset
_base_initialize_episode = SampleLoadingEnv._initialize_episode


def _reproducible_reset(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    seed = kwargs.get("seed")
    if seed is None and args:
        seed = args[0]
    if seed is not None:
        set_seed(int(seed))
    print(f"[STRICT RESET START] seed={seed}", flush=True)
    result = _base_reset(self, *args, **kwargs)
    print(f"[STRICT RESET DONE] seed={seed}", flush=True)
    return result


def _reset_strict_state(self: SampleLoadingEnv) -> None:
    device = self.device
    self._place_stable_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self._strict_last_eval_step = -1
    self._strict_initial_cube_z = None
    self._strict_max_cube_z = torch.full((self.num_envs,), -torch.inf, device=device)
    self._strict_min_rack_xy = torch.full((self.num_envs,), torch.inf, device=device)
    self._strict_min_left_distance = torch.full((self.num_envs,), torch.inf, device=device)
    self._strict_min_right_distance = torch.full((self.num_envs,), torch.inf, device=device)
    self._strict_right_approached = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self._strict_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self._strict_right_held_while_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self._strict_left_handoff_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self._strict_reached_rack = torch.zeros(self.num_envs, dtype=torch.bool, device=device)


def _strict_initialize_episode(self: SampleLoadingEnv, *args: Any, **kwargs: Any) -> None:
    # Base reset may evaluate task state while clearing/saving buffers, so the
    # strict fields must exist both before and after that call.
    print("[STRICT INIT START]", flush=True)
    _reset_strict_state(self)
    _base_initialize_episode(self, *args, **kwargs)
    _reset_strict_state(self)
    print("[STRICT INIT DONE]", flush=True)


def _strict_evaluate_task_state(
    self: SampleLoadingEnv,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not getattr(self, "_strict_eval_started", False):
        self._strict_eval_started = True
        print("[STRICT EVAL START]", flush=True)
    if not hasattr(self, "_strict_last_eval_step"):
        _reset_strict_state(self)
    cube = self.sim.get_rigid_object("cube")
    rack = self.sim.get_rigid_object("rack")
    if cube is None or rack is None or cube.body_data is None:
        raise RuntimeError("Strict sample_loading metrics require cube/rack rigid-body data.")

    cube_pose = cube.get_local_pose(to_matrix=True)
    rack_pose = rack.get_local_pose(to_matrix=True)
    cube_position = cube_pose[:, :3, 3]
    rack_position = rack_pose[:, :3, 3]
    cube_velocity = cube.body_data.lin_vel
    if cube_velocity is None:
        raise RuntimeError("Strict sample_loading metrics require cube linear velocity.")

    left_pose = self.robot.get_link_pose("left_link6")
    right_pose = self.robot.get_link_pose("right_link6")
    if left_pose is None or right_pose is None:
        raise RuntimeError("Strict sample_loading metrics require both EEF link poses.")
    left_position = left_pose[:, :3]
    right_position = right_pose[:, :3]

    cube_to_left = torch.linalg.vector_norm(cube_position - left_position, dim=-1)
    cube_to_right = torch.linalg.vector_norm(cube_position - right_position, dim=-1)
    qpos = self.robot.get_qpos()
    left_gripper_ids = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
    right_gripper_ids = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)
    if not left_gripper_ids or not right_gripper_ids:
        raise RuntimeError("Strict sample_loading metrics require both gripper joints.")
    left_gripper_q_mean = qpos[:, left_gripper_ids].mean(dim=-1)
    right_gripper_q_mean = qpos[:, right_gripper_ids].mean(dim=-1)
    left_released = (left_gripper_q_mean > GRIPPER_OPEN_THRESHOLD) | (
        cube_to_left > RELEASE_DISTANCE
    )
    right_released = (right_gripper_q_mean > GRIPPER_OPEN_THRESHOLD) | (
        cube_to_right > RELEASE_DISTANCE
    )
    cube_velocity_norm = torch.linalg.vector_norm(cube_velocity, dim=-1)
    rack_xy_distance = torch.linalg.vector_norm(
        cube_position[:, :2] - rack_position[:, :2], dim=-1
    )
    cube_z_above_rack = cube_position[:, 2] - rack_position[:, 2]
    cube_z_axis = cube_pose[:, :3, 2]
    upright_angle = torch.arccos(torch.clamp(cube_z_axis[:, 2], -1.0, 1.0))

    placement_frame = (
        (rack_xy_distance < XY_THRESHOLD)
        & (cube_z_above_rack >= MIN_CUBE_Z_ABOVE_RACK)
        & (cube_z_above_rack <= MAX_CUBE_Z_ABOVE_RACK)
        & (cube_velocity_norm < VELOCITY_THRESHOLD)
        & left_released
        & right_released
        & (upright_angle < UPRIGHT_ANGLE_THRESHOLD)
    )

    elapsed = int(self._elapsed_steps[0].item())
    if self._strict_initial_cube_z is None:
        self._strict_initial_cube_z = cube_position[:, 2].clone()
    if elapsed != self._strict_last_eval_step:
        self._strict_last_eval_step = elapsed
        self._place_stable_count = torch.where(
            placement_frame,
            self._place_stable_count + 1,
            torch.zeros_like(self._place_stable_count),
        )
        self._strict_max_cube_z = torch.maximum(self._strict_max_cube_z, cube_position[:, 2])
        self._strict_min_rack_xy = torch.minimum(self._strict_min_rack_xy, rack_xy_distance)
        self._strict_min_left_distance = torch.minimum(
            self._strict_min_left_distance, cube_to_left
        )
        self._strict_min_right_distance = torch.minimum(
            self._strict_min_right_distance, cube_to_right
        )
        lifted_now = cube_position[:, 2] >= self._strict_initial_cube_z + 0.03
        self._strict_right_approached |= cube_to_right < APPROACH_DISTANCE
        self._strict_lifted |= lifted_now
        self._strict_right_held_while_lifted |= lifted_now & (
            cube_to_right < HOLD_DISTANCE
        )
        self._strict_left_handoff_reached |= self._strict_lifted & (
            cube_to_left < HOLD_DISTANCE
        )
        self._strict_reached_rack |= rack_xy_distance < 0.05

    success = self._place_stable_count >= STABLE_STEPS
    metrics = {
        "cube_xy_dist": rack_xy_distance,
        "cube_z": cube_position[:, 2],
        "rack_z": rack_position[:, 2],
        "placement_ok_single_frame": placement_frame,
        "place_stable_count": self._place_stable_count,
        "cube_lin_vel_norm": cube_velocity_norm,
        "cube_to_left_eef_dist": cube_to_left,
        "cube_to_right_eef_dist": cube_to_right,
        "left_gripper_q_mean": left_gripper_q_mean,
        "right_gripper_q_mean": right_gripper_q_mean,
        "cube_vertical_angle": upright_angle,
        "strict_left_released": left_released,
        "strict_right_released": right_released,
        "strict_released": left_released & right_released,
        "strict_right_approached": self._strict_right_approached,
        "strict_lifted": self._strict_lifted,
        "strict_right_held_while_lifted": self._strict_right_held_while_lifted,
        "strict_left_handoff_reached": self._strict_left_handoff_reached,
        "strict_reached_rack": self._strict_reached_rack,
        "strict_max_cube_z": self._strict_max_cube_z,
        "strict_min_rack_xy": self._strict_min_rack_xy,
        "strict_min_left_eef_dist": self._strict_min_left_distance,
        "strict_min_right_eef_dist": self._strict_min_right_distance,
        "strict_metrics_active": torch.ones_like(success, dtype=torch.bool),
    }
    if not getattr(self, "_strict_metrics_announced", False):
        self._strict_metrics_announced = True
        print(
            "[STRICT METRICS ACTIVE]",
            f"velocity={cube_velocity_norm.detach().cpu().tolist()}",
            f"left_distance={cube_to_left.detach().cpu().tolist()}",
            f"right_distance={cube_to_right.detach().cpu().tolist()}",
            flush=True,
        )
    return success, {}, metrics


base_evaluator.RecordingEnvProxy.reset = _reproducible_reset
SampleLoadingEnv._initialize_episode = _strict_initialize_episode
SampleLoadingEnv._evaluate_task_state = _strict_evaluate_task_state


if __name__ == "__main__":
    base_evaluator.main()
