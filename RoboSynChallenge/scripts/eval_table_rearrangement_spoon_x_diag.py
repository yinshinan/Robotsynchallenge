#!/usr/bin/env python
"""Evaluate copied random config with a diagnostic spoon-x interval."""

from copy import deepcopy

import eval_table_rearrangement_reproducible_diag as reproducible_eval


base_eval = reproducible_eval.base_eval
_original_find_gym_config = base_eval.find_gym_config


def find_gym_config(config):
    gym_config = deepcopy(_original_find_gym_config(config))
    x_min = float(config["diag_spoon_x_min_m"])
    x_max = float(config["diag_spoon_x_max_m"])
    if not 0.4 <= x_min < x_max <= 0.65:
        raise ValueError(f"Invalid spoon x range: [{x_min}, {x_max}]")
    position_range = gym_config["env"]["events"]["init_spoon_pose"]["params"][
        "position_range"
    ]
    position_range[0][0] = x_min
    position_range[1][0] = x_max
    print(f"[TABLE SPOON X DIAG] x_range_m=[{x_min}, {x_max}]", flush=True)
    return gym_config


base_eval.find_gym_config = find_gym_config


if __name__ == "__main__":
    base_eval.main()
