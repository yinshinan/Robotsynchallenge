"""ACT adapter for isolated table_rearrangement diagnostics.

Unlike policy/act/deploy_policy.py, this file treats gripper predictions as the
physical [0, 0.05] metre values present in the dataset.  The legacy transform is
still available through ``diag_gripper_mode: legacy_scaled`` for an A/B baseline.
"""

import json
from pathlib import Path

import numpy as np
import torch

from policy.act.deploy_policy import get_model as _get_base_model


GRIPPER_ACTION_IDS = (6, 13)


def get_model(usr_args):
    model = _get_base_model(usr_args)
    model.diag_gripper_mode = str(
        usr_args.get("diag_gripper_mode", "physical")
    )
    if model.diag_gripper_mode not in {"physical", "legacy_scaled"}:
        raise ValueError(
            "diag_gripper_mode must be 'physical' or 'legacy_scaled', got "
            f"{model.diag_gripper_mode!r}"
        )
    model.diag_log_every = max(1, int(usr_args.get("diag_log_every", 1)))
    model.diag_log_path = str(
        usr_args.get(
            "diag_log_path", "diagnostic_logs/table_rearrangement_trace.jsonl"
        )
    )
    log_path = Path(model.diag_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # One process corresponds to one A/B case; start with a clean trace.
    model._diag_log_handle = log_path.open("w", encoding="utf-8")
    model._diag_episode = -1
    model._diag_step = 0
    print(
        "[TABLE DIAG]",
        f"gripper_mode={model.diag_gripper_mode}",
        f"n_action_steps={model.config.n_action_steps}",
        f"trace={log_path}",
        flush=True,
    )
    return model


def _as_batch_tensor(value, device):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _extract_state(model, obs):
    state = obs
    for key in str(model.state_obs_path).split("/"):
        if key:
            state = state[key]
    return _as_batch_tensor(state, model.act_device)


def _build_batch(model, obs, state):
    qvel = _as_batch_tensor(obs["robot"]["qvel"], model.act_device)
    qf = _as_batch_tensor(obs["robot"]["qf"], model.act_device)
    if model.neutralize_qvel:
        mean = model.normalize_inputs.buffer_observation_qvel["mean"]
        qvel = mean.to(model.act_device, dtype=torch.float32).unsqueeze(0).expand_as(qvel)

    batch = {
        "observation.state": state,
        "observation.qvel": qvel,
        "observation.qf": qf,
    }
    for image_key in model.act_image_keys:
        camera_name = model.image_key_map.get(
            image_key, image_key.removeprefix("observation.images.")
        )
        image = _as_batch_tensor(obs["sensor"][camera_name]["color"], model.act_device)
        image = image[..., :3].permute(0, 3, 1, 2).contiguous()
        if torch.max(image) > 1.5:
            image = image / 255.0
        batch[image_key] = image
    return batch


def _prepare_env_action(env, model, raw_action):
    if raw_action.ndim == 1:
        raw_action = raw_action.unsqueeze(0)
    if raw_action.ndim != 2:
        raise ValueError(f"Expected policy action [B, D], got {tuple(raw_action.shape)}")

    env_dim = int(np.prod(env.unwrapped.single_action_space.shape))
    policy_dim = int(raw_action.shape[-1])
    if policy_dim != env_dim:
        message = f"Policy action has dim {policy_dim}, env expects {env_dim}."
        if model.strict_action_dim or policy_dim < env_dim:
            raise ValueError(message)
        raw_action = raw_action[:, :env_dim]

    env_action = raw_action.clone()
    action_low = torch.as_tensor(
        env.unwrapped.single_action_space.low,
        device=env_action.device,
        dtype=env_action.dtype,
    )
    action_high = torch.as_tensor(
        env.unwrapped.single_action_space.high,
        device=env_action.device,
        dtype=env_action.dtype,
    )
    ids = list(GRIPPER_ACTION_IDS)
    if model.diag_gripper_mode == "legacy_scaled":
        env_action[:, ids] = action_low[ids] + env_action[:, ids].clamp(0.0, 1.0) * (
            action_high[ids] - action_low[ids]
        )
    else:
        # Dataset labels already use the physical prismatic-joint range.
        env_action[:, ids] = torch.maximum(
            torch.minimum(env_action[:, ids], action_high[ids]), action_low[ids]
        )
    return env_action


def _physical_gripper_qpos(env):
    robot = env.unwrapped.robot
    qpos = robot.get_qpos()
    left_id = robot.get_joint_ids("left_eef", remove_mimic=True)[0]
    right_id = robot.get_joint_ids("right_eef", remove_mimic=True)[0]
    return float(qpos[0, left_id].item()), float(qpos[0, right_id].item())


def _xyz(value):
    return [float(x) for x in value[0].detach().cpu().tolist()]


def _write_trace(env, model, state, raw_action, env_action):
    if model._diag_step % model.diag_log_every:
        return
    left_qpos, right_qpos = _physical_gripper_qpos(env)
    row = {
        "episode": model._diag_episode,
        "step": model._diag_step,
        "gripper_mode": model.diag_gripper_mode,
        "raw_gripper_action_m": [
            float(raw_action[0, 6].item()),
            float(raw_action[0, 13].item()),
        ],
        "env_gripper_action_m": [
            float(env_action[0, 6].item()),
            float(env_action[0, 13].item()),
        ],
        "physical_gripper_qpos_m": [left_qpos, right_qpos],
        "normalized_gripper_observation": [
            float(state[0, 6].item()),
            float(state[0, 13].item()),
        ],
    }
    base_env = env.unwrapped
    if hasattr(base_env, "diagnostic_metrics"):
        metrics = base_env.diagnostic_metrics()
        robot = base_env.robot
        left_link6_xyz = robot.get_link_pose(
            "left_link6", to_matrix=True
        )[:, :3, 3]
        right_link6_xyz = robot.get_link_pose(
            "right_link6", to_matrix=True
        )[:, :3, 3]
        row.update(
            fork_xyz_m=_xyz(metrics["fork_xyz"]),
            spoon_xyz_m=_xyz(metrics["spoon_xyz"]),
            plate_xyz_m=_xyz(metrics["plate_xyz"]),
            left_link6_xyz_m=_xyz(left_link6_xyz),
            right_link6_xyz_m=_xyz(right_link6_xyz),
            left_link6_to_fork_m=float(
                torch.linalg.vector_norm(
                    left_link6_xyz - metrics["fork_xyz"], dim=-1
                )[0].item()
            ),
            right_link6_to_spoon_m=float(
                torch.linalg.vector_norm(
                    right_link6_xyz - metrics["spoon_xyz"], dim=-1
                )[0].item()
            ),
            fork_lift_m=float(metrics["fork_lift"][0].item()),
            spoon_lift_m=float(metrics["spoon_lift"][0].item()),
            fork_planar_displacement_m=float(
                metrics["fork_planar_displacement"][0].item()
            ),
            spoon_planar_displacement_m=float(
                metrics["spoon_planar_displacement"][0].item()
            ),
        )
    model._diag_log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    model._diag_log_handle.flush()
    if model._diag_step % 10 == 0:
        print(
            "[TABLE DIAG STEP]",
            f"step={model._diag_step}",
            f"raw_gripper={row['raw_gripper_action_m']}",
            f"env_gripper={row['env_gripper_action_m']}",
            f"physical_qpos={row['physical_gripper_qpos_m']}",
            flush=True,
        )


def _is_done(value):
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def eval(env, model, obs):
    final_obs = obs
    info = {}
    truncated = False
    executed_steps = 0

    if not getattr(model, "_diag_env_limits_printed", False):
        wrapper = getattr(env, "_env", env)
        wrapper_limits = []
        while wrapper is not None:
            wrapper_limits.append(
                [type(wrapper).__name__, getattr(wrapper, "_max_episode_steps", None)]
            )
            wrapper = getattr(wrapper, "env", None)
        print(
            "[TABLE DIAG LIMITS]",
            f"base_max_steps={getattr(env.unwrapped, 'max_episode_steps', None)}",
            f"wrappers={wrapper_limits}",
            flush=True,
        )
        model._diag_env_limits_printed = True

    for _ in range(model.act_step):
        state = _extract_state(model, final_obs)
        raw_action = model.select_action(_build_batch(model, final_obs, state))
        env_action = _prepare_env_action(env, model, raw_action)

        _write_trace(env, model, state, raw_action, env_action)
        action_tensor = env_action.detach().to(
            device=env.unwrapped.device, dtype=torch.float32
        )
        final_obs, _, terminated, truncated_value, info = env.step(action_tensor)
        executed_steps += 1
        model._diag_step += 1
        if hasattr(env.unwrapped, "update_diagnostic_state"):
            env.unwrapped.update_diagnostic_state()
        info["_policy_action_steps"] = executed_steps
        if _is_done(terminated) or _is_done(truncated_value):
            elapsed = getattr(env.unwrapped, "elapsed_steps", None)
            print(
                "[TABLE DIAG DONE]",
                f"terminated={terminated}",
                f"truncated={truncated_value}",
                f"elapsed_steps={elapsed}",
                flush=True,
            )
            truncated = True
            break

    strict_success = env.unwrapped.is_task_success()
    print(
        "[TABLE DIAG RETURN]",
        f"executed_steps={executed_steps}",
        f"strict_success={strict_success}",
        f"truncated={truncated}",
        flush=True,
    )
    return final_obs, info, truncated


def reset_model(model):
    model.reset()
    model._diag_episode += 1
    model._diag_step = 0
    model._trajectory_debug_count = 0
    model._diag_env_limits_printed = False
