#!/usr/bin/env python
"""Evaluate random table_rearrangement with an isolated post-sync gripper reset.

The official task/config files are not modified.  The reset hook runs after the
evaluator's reset synchronization so all four physical gripper joints start at
zero position, target, velocity, and force in the observation given to ACT.
"""

from copy import deepcopy

import eval_policy as base_eval
import torch
from policy.act_table_rearrangement_diag.config_patch import (
    patch_grasp_z,
    patch_random_arm_joint_ids,
)


GRIPPER_PHYSICAL_JOINT_IDS = [12, 13, 14, 15]

_original_find_gym_config = base_eval.find_gym_config
_original_find_action_config = base_eval.find_action_config


class PostSyncGripperResetProxy(base_eval.RecordingEnvProxy):
    """Reset physical gripper state after the standard observation sync."""

    def reset(self, *args, **kwargs):
        _, _ = super().reset(*args, **kwargs)
        base_env = self._env.unwrapped
        robot = base_env.robot
        env_ids = torch.arange(
            base_env.num_envs, dtype=torch.int32, device=base_env.device
        )
        zeros = torch.zeros(
            (base_env.num_envs, len(GRIPPER_PHYSICAL_JOINT_IDS)),
            dtype=torch.float32,
            device=base_env.device,
        )
        for target in (False, True):
            robot.set_qpos(
                zeros,
                joint_ids=GRIPPER_PHYSICAL_JOINT_IDS,
                env_ids=env_ids,
                target=target,
            )
            robot.set_qvel(
                zeros,
                joint_ids=GRIPPER_PHYSICAL_JOINT_IDS,
                env_ids=env_ids,
                target=target,
            )
        robot.set_qf(
            zeros,
            joint_ids=GRIPPER_PHYSICAL_JOINT_IDS,
            env_ids=env_ids,
        )

        current = robot.get_qpos()[0, GRIPPER_PHYSICAL_JOINT_IDS]
        target = robot.get_qpos(target=True)[0, GRIPPER_PHYSICAL_JOINT_IDS]
        print(
            "[TABLE POST-SYNC GRIPPER RESET]",
            f"current={current.detach().cpu().tolist()}",
            f"target={target.detach().cpu().tolist()}",
            flush=True,
        )
        return base_env.get_obs(), base_env.get_info()


def find_gym_config(config):
    gym_config = deepcopy(_original_find_gym_config(config))
    gym_config["id"] = "TableRearrangementDiagnostic"
    joint_ids_changed = patch_random_arm_joint_ids(gym_config)
    print(
        "[TABLE RANDOM RESET DIAG]",
        "environment=TableRearrangementDiagnostic",
        f"random_joint_ids_corrected={joint_ids_changed}",
        f"gripper_reset_joint_ids={GRIPPER_PHYSICAL_JOINT_IDS}",
        flush=True,
    )
    return gym_config


def find_action_config(config):
    action_config = deepcopy(_original_find_action_config(config))
    target_offset = float(config.get("diag_grasp_z_offset_m", -0.008))
    changed = patch_grasp_z(action_config, target_offset)
    if changed != 2:
        raise RuntimeError(
            f"Expected to patch fork and spoon grasp z offsets, patched {changed}."
        )
    return action_config


base_eval.find_gym_config = find_gym_config
base_eval.find_action_config = find_action_config
base_eval.RecordingEnvProxy = PostSyncGripperResetProxy


if __name__ == "__main__":
    base_eval.main()
