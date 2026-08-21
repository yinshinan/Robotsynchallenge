#!/usr/bin/env python
"""Run table_rearrangement diagnostics without editing official configs.

This wrapper delegates the evaluation loop to scripts/eval_policy.py, but patches
copies of the loaded JSON dictionaries in memory.  Source gym/action JSON files
are never written.
"""

from copy import deepcopy

import eval_policy as base_eval
from policy.act_table_rearrangement_diag.config_patch import (
    patch_grasp_z,
    patch_random_arm_joint_ids,
)


_original_find_gym_config = base_eval.find_gym_config
_original_find_action_config = base_eval.find_action_config


def find_gym_config(config):
    gym_config = deepcopy(_original_find_gym_config(config))
    gym_config["id"] = "TableRearrangementDiagnostic"
    joint_ids_changed = False
    if bool(config.get("diag_correct_random_joint_ids", True)):
        joint_ids_changed = patch_random_arm_joint_ids(gym_config)
    print(
        "[TABLE DIAG CONFIG]",
        "environment=TableRearrangementDiagnostic",
        f"random_joint_ids_corrected={joint_ids_changed}",
        flush=True,
    )
    return gym_config


def find_action_config(config):
    action_config = deepcopy(_original_find_action_config(config))
    target_offset = float(config.get("diag_grasp_z_offset_m", 0.0))
    changed = patch_grasp_z(action_config, target_offset)
    if changed != 2:
        raise RuntimeError(
            f"Expected to patch fork and spoon grasp z offsets, patched {changed}."
        )
    print(
        "[TABLE DIAG CONFIG]",
        f"grasp_z_offset_m={target_offset}",
        f"patched_entries={changed}",
        flush=True,
    )
    return action_config


base_eval.find_gym_config = find_gym_config
base_eval.find_action_config = find_action_config


if __name__ == "__main__":
    base_eval.main()
