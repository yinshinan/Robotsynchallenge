#!/usr/bin/env python
"""Reproducible evaluator with per-frame sample_loading JSONL traces.

The existing evaluator and ACT adapter are imported without modification.  Set
``SAMPLE_LOADING_TRACE_PATH`` to choose the output JSONL file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, REPO_ROOT / "policy", WORKSPACE_ROOT / "EmbodiChain"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import eval_policy as base_evaluator  # noqa: E402
from embodichain.utils import set_seed  # noqa: E402


TRACE_PATH = Path(
    os.environ.get(
        "SAMPLE_LOADING_TRACE_PATH",
        REPO_ROOT / "eval_result/sample_loading/sample_loading_trace.jsonl",
    )
)
TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
TRACE_FILE = TRACE_PATH.open("w", encoding="utf-8")
TRACE_STATE: dict[str, Any] = {"model": None, "episode": -1, "step": 0}


def _as_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 1:
        array = array[0]
    return [float(item) for item in array.reshape(-1)]


def _position(entity: Any) -> list[float]:
    pose = entity.get_local_pose(to_matrix=False)
    return _as_list(pose)[:3]


def _link_position(robot: Any, link_name: str) -> list[float]:
    return _as_list(robot.get_link_pose(link_name))[:3]


def _distance(first: list[float], second: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def _write_trace(
    env_proxy: Any,
    obs: dict[str, Any],
    event: str,
    action: Any | None = None,
) -> None:
    try:
        _write_trace_impl(env_proxy, obs, event, action)
    except Exception as exc:
        TRACE_FILE.write(
            json.dumps(
                {
                    "event": "trace_error",
                    "episode": TRACE_STATE["episode"],
                    "step": TRACE_STATE["step"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        TRACE_FILE.flush()
        raise


def _write_trace_impl(
    env_proxy: Any,
    obs: dict[str, Any],
    event: str,
    action: Any | None = None,
) -> None:
    env = env_proxy._env.unwrapped
    robot = env.robot
    cube = env.sim.get_rigid_object("cube")
    rack = env.sim.get_rigid_object("rack")

    qpos = _as_list(obs["robot"]["qpos"])
    qvel = _as_list(obs["robot"]["qvel"])
    qf = _as_list(obs["robot"]["qf"])
    cube_position = _position(cube)
    rack_position = _position(rack)
    left_eef_position = _link_position(robot, "left_link6")
    right_eef_position = _link_position(robot, "right_link6")

    model = TRACE_STATE["model"]
    queue_remaining = None
    replan_step = None
    n_action_steps = None
    if model is not None:
        n_action_steps = int(model.config.n_action_steps)
        action_queue = getattr(model, "_action_queue", None)
        if action_queue is not None:
            queue_remaining = len(action_queue)
            replan_step = queue_remaining == n_action_steps - 1

    action_values = None if action is None else _as_list(action)
    row = {
        "event": event,
        "episode": TRACE_STATE["episode"],
        "step": TRACE_STATE["step"],
        "n_action_steps": n_action_steps,
        "queue_remaining": queue_remaining,
        "replan_step": replan_step,
        "action": action_values,
        "qpos": qpos,
        "qvel": qvel,
        "qf": qf,
        "left_gripper_action": None if action_values is None else action_values[6],
        "right_gripper_action": None if action_values is None else action_values[13],
        "left_gripper_qpos": qpos[6],
        "right_gripper_qpos": qpos[13],
        "left_gripper_qf": qf[6],
        "right_gripper_qf": qf[13],
        "cube_position": cube_position,
        "rack_position": rack_position,
        "cube_linear_velocity": _as_list(cube.body_data.lin_vel)[:3],
        "left_eef_position": left_eef_position,
        "right_eef_position": right_eef_position,
        "cube_to_left_eef": _distance(cube_position, left_eef_position),
        "cube_to_right_eef": _distance(cube_position, right_eef_position),
        "cube_to_rack_xy": float(
            np.linalg.norm(
                np.asarray(cube_position[:2]) - np.asarray(rack_position[:2])
            )
        ),
    }
    if action_values is not None:
        row["left_arm_tracking_error"] = float(
            np.mean(np.abs(np.asarray(action_values[:6]) - np.asarray(qpos[:6])))
        )
        row["right_arm_tracking_error"] = float(
            np.mean(
                np.abs(np.asarray(action_values[7:13]) - np.asarray(qpos[7:13]))
            )
        )
    TRACE_FILE.write(json.dumps(row, ensure_ascii=False) + "\n")
    TRACE_FILE.flush()


_base_reset = base_evaluator.RecordingEnvProxy.reset
_base_step = base_evaluator.RecordingEnvProxy.step
_base_load_policy_adapter = base_evaluator.load_policy_adapter


def _traced_reset(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    seed = kwargs.get("seed")
    if seed is None and args:
        seed = args[0]
    if seed is not None:
        set_seed(int(seed))
    obs, info = _base_reset(self, *args, **kwargs)
    TRACE_STATE["episode"] += 1
    TRACE_STATE["step"] = 0
    _write_trace(self, obs, "reset")
    return obs, info


def _traced_step(self: Any, action: Any) -> tuple[Any, Any, Any, Any, Any]:
    obs, reward, terminated, truncated, info = _base_step(self, action)
    _write_trace(self, obs, "step", action)
    TRACE_STATE["step"] += 1
    return obs, reward, terminated, truncated, info


def _traced_load_policy_adapter(policy_name: str) -> Any:
    policy_package = _base_load_policy_adapter(policy_name)
    original_get_model = policy_package.get_model

    def get_model(config: dict[str, Any]) -> Any:
        model = original_get_model(config)
        TRACE_STATE["model"] = model
        return model

    policy_package.get_model = get_model
    return policy_package


base_evaluator.RecordingEnvProxy.reset = _traced_reset
base_evaluator.RecordingEnvProxy.step = _traced_step
base_evaluator.load_policy_adapter = _traced_load_policy_adapter


if __name__ == "__main__":
    try:
        base_evaluator.main()
    finally:
        TRACE_FILE.close()
