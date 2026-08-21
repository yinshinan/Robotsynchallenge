#!/usr/bin/env python
"""Single-factor table-rearrangement ablations on copied random configs.

Every factor keeps the original reset event in place.  Ranges are collapsed or
appearance probabilities are changed so the event still performs the same
random draws in the same order.  Official task/config files are never written.
"""

from copy import deepcopy

import eval_table_rearrangement_reproducible_diag as reproducible_eval


base_eval = reproducible_eval.base_eval
_original_find_gym_config = base_eval.find_gym_config


FACTORS = {
    "baseline",
    "camera_fx_zero",
    "camera_fy_zero",
    "camera_intrinsics_zero",
    "camera_extrinsics_zero",
    "spoon_y_center",
    "spoon_yaw_zero",
    "robot_init_zero",
    "distractors_hidden",
}


def _zero_vector_range(params, key, dimensions):
    if key not in params:
        raise RuntimeError(f"Expected randomization range was not found: {key}")
    params[key] = [[0.0] * dimensions, [0.0] * dimensions]


def find_gym_config(config):
    gym_config = deepcopy(_original_find_gym_config(config))
    factor = str(config.get("diag_factor", "baseline"))
    if factor not in FACTORS:
        raise ValueError(f"Unknown diag_factor={factor!r}; expected one of {sorted(FACTORS)}")

    events = gym_config["env"]["events"]
    if factor in {"camera_fx_zero", "camera_fy_zero", "camera_intrinsics_zero"}:
        params = events["random_camera_high_intrinsics"]["params"]
        if factor in {"camera_fx_zero", "camera_intrinsics_zero"}:
            params["focal_x_range"] = [0.0, 0.0]
        if factor in {"camera_fy_zero", "camera_intrinsics_zero"}:
            params["focal_y_range"] = [0.0, 0.0]
    elif factor == "camera_extrinsics_zero":
        params = events["random_camera_high_extrinsics"]["params"]
        _zero_vector_range(params, "pos_range", 3)
        _zero_vector_range(params, "euler_range", 3)
    elif factor == "spoon_y_center":
        center = float(config.get("diag_spoon_y_center_m", -0.225))
        position_range = events["init_spoon_pose"]["params"]["position_range"]
        position_range[0][1] = center
        position_range[1][1] = center
    elif factor == "spoon_yaw_zero":
        events["init_spoon_pose"]["params"]["rotation_range"] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    elif factor == "robot_init_zero":
        eef_params = events["random_robot_init_eef_pose"]["params"]
        _zero_vector_range(eef_params, "position_range", 3)
        qpos_params = events["random_robot_qpos"]["params"]
        qpos_params["qpos_range"] = [[0.0] * 12, [0.0] * 12]
    elif factor == "distractors_hidden":
        # torch.rand, position sampling, yaw sampling, and Python random.sample
        # still execute; only the final visibility/pose selection changes.
        events["randomize_distractor_slots"]["params"]["appear_probs"] = [0.0, 0.0]

    print(
        "[TABLE FACTOR DIAG]",
        f"factor={factor}",
        "events_preserved=True",
        flush=True,
    )
    return gym_config


base_eval.find_gym_config = find_gym_config


if __name__ == "__main__":
    base_eval.main()
