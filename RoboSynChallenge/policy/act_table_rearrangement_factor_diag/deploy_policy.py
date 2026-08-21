"""ACT adapter with full state, camera, image-hash, and distractor tracing."""

import hashlib
import json

import numpy as np
import torch

from policy.act_table_rearrangement_diag import deploy_policy as base


def get_model(usr_args):
    model = base.get_model(usr_args)
    model.diag_factor = str(usr_args.get("diag_factor", "baseline"))
    model.diag_image_hash_every = max(
        0, int(usr_args.get("diag_image_hash_every", 10))
    )
    print(
        "[TABLE FACTOR TRACE]",
        f"factor={model.diag_factor}",
        f"image_hash_every={model.diag_image_hash_every}",
        flush=True,
    )
    return model


def _list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    return value


def _first(value):
    return _list(value[0])


def _pose(env, uid):
    obj = env.unwrapped.sim.get_rigid_object(uid)
    return _first(obj.get_local_pose(to_matrix=False))


def _camera_state(env, uid):
    camera = env.unwrapped.sim.get_sensor(uid)
    return {
        "intrinsics_3x3": _first(camera.get_intrinsics()),
        "local_pose_xyzw_or_wxyz": _first(camera.get_local_pose(to_matrix=False)),
        "arena_pose_xyzw_or_wxyz": _first(camera.get_arena_pose(to_matrix=False)),
    }


def _active_distractors(env):
    rows = []
    sim = env.unwrapped.sim
    for uid in sorted(sim.get_rigid_object_uid_list()):
        if not uid.startswith("distractor_"):
            continue
        obj = sim.get_rigid_object(uid)
        pose = _first(obj.get_local_pose(to_matrix=False))
        if float(pose[2]) <= -1.0:
            continue
        rows.append({"uid": uid, "pose_xyz_qwxyz": pose})
    return rows


def _image_hashes(model, obs):
    hashes = {}
    for image_key in model.act_image_keys:
        camera_name = model.image_key_map.get(
            image_key, image_key.removeprefix("observation.images.")
        )
        value = obs["sensor"][camera_name]["color"]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
        hashes[camera_name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest.hexdigest(),
        }
    return hashes


def _write_trace(env, model, obs, state, raw_action, env_action):
    if model._diag_step % model.diag_log_every:
        return

    base_env = env.unwrapped
    robot = base_env.robot
    qpos = robot.get_qpos()
    qvel = robot.get_qvel()
    left_gripper, right_gripper = base._physical_gripper_qpos(env)
    fork_pose = _pose(env, "fork")
    spoon_pose = _pose(env, "spoon")
    plate_pose = _pose(env, "plate")
    left_link6 = robot.get_link_pose("left_link6", to_matrix=True)[:, :3, 3]
    right_link6 = robot.get_link_pose("right_link6", to_matrix=True)[:, :3, 3]
    spoon_xyz = torch.as_tensor(spoon_pose[:3], device=right_link6.device).unsqueeze(0)
    fork_xyz = torch.as_tensor(fork_pose[:3], device=left_link6.device).unsqueeze(0)

    row = {
        "episode": model._diag_episode,
        "step": model._diag_step,
        "factor": model.diag_factor,
        "raw_action_14d": _first(raw_action),
        "env_action_14d": _first(env_action),
        "robot_qpos_16d": _first(qpos),
        "robot_qvel_16d": _first(qvel),
        "physical_gripper_qpos_m": [left_gripper, right_gripper],
        "normalized_robot_state": _first(state),
        "fork_pose_xyz_qwxyz": fork_pose,
        "spoon_pose_xyz_qwxyz": spoon_pose,
        "plate_pose_xyz_qwxyz": plate_pose,
        "left_link6_xyz_m": _first(left_link6),
        "right_link6_xyz_m": _first(right_link6),
        "fork_minus_left_link6_xyz_m": _first(fork_xyz - left_link6),
        "spoon_minus_right_link6_xyz_m": _first(spoon_xyz - right_link6),
    }

    if hasattr(base_env, "diagnostic_metrics"):
        metrics = base_env.diagnostic_metrics()
        row.update(
            fork_lift_m=float(metrics["fork_lift"][0].item()),
            spoon_lift_m=float(metrics["spoon_lift"][0].item()),
            fork_planar_displacement_m=float(
                metrics["fork_planar_displacement"][0].item()
            ),
            spoon_planar_displacement_m=float(
                metrics["spoon_planar_displacement"][0].item()
            ),
        )

    if model._diag_step == 0:
        row["joint_groups"] = {
            name: robot.get_joint_ids(name)
            for name in ("left_arm", "right_arm", "left_eef", "right_eef")
        }
        row["cameras"] = {
            uid: _camera_state(env, uid)
            for uid in ("cam_high", "cam_right_wrist", "cam_left_wrist")
        }
        row["active_distractors"] = _active_distractors(env)

    if model.diag_image_hash_every and (
        model._diag_step % model.diag_image_hash_every == 0
    ):
        row["image_hashes"] = _image_hashes(model, obs)

    model._diag_log_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    model._diag_log_handle.flush()


def eval(env, model, obs):
    final_obs = obs
    info = {}
    truncated = False
    executed_steps = 0

    for _ in range(model.act_step):
        state = base._extract_state(model, final_obs)
        raw_action = model.select_action(base._build_batch(model, final_obs, state))
        env_action = base._prepare_env_action(env, model, raw_action)
        _write_trace(env, model, final_obs, state, raw_action, env_action)

        action_tensor = env_action.detach().to(
            device=env.unwrapped.device, dtype=torch.float32
        )
        final_obs, _, terminated, truncated_value, info = env.step(action_tensor)
        executed_steps += 1
        model._diag_step += 1
        if hasattr(env.unwrapped, "update_diagnostic_state"):
            env.unwrapped.update_diagnostic_state()
        info["_policy_action_steps"] = executed_steps
        if base._is_done(terminated) or base._is_done(truncated_value):
            truncated = True
            break

    return final_obs, info, truncated


def reset_model(model):
    base.reset_model(model)
