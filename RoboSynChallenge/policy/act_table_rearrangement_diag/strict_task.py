"""A stricter table-rearrangement environment used only by diagnostics.

The official task class remains untouched. This subclass requires evidence that
both utensils were lifted, checks full placement position, and requires five
consecutive control frames inside the target region before reporting success.
"""

import torch

from embodichain.lab.gym.utils.registration import register_env
from robosynchallenge.tasks.table_rearrangement.table_rearrangement import (
    TableRearrangementEnv,
)


@register_env("TableRearrangementDiagnostic", max_episode_steps=1000)
class TableRearrangementDiagnosticEnv(TableRearrangementEnv):
    """TableRearrangement with a false-positive-resistant success check."""

    XY_TOLERANCE_M = 0.035
    Z_TOLERANCE_M = 0.035
    MIN_LIFT_M = 0.020
    MIN_TARGET_STREAK_STEPS = 5

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        fork_pose, spoon_pose, _ = self._object_poses()
        self._diag_initial_fork_xyz = fork_pose[:, :3, 3].clone()
        self._diag_initial_spoon_xyz = spoon_pose[:, :3, 3].clone()
        self._diag_max_fork_z = self._diag_initial_fork_xyz[:, 2].clone()
        self._diag_max_spoon_z = self._diag_initial_spoon_xyz[:, 2].clone()
        self._diag_target_streak = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        return result

    def _object_poses(self):
        fork = self.sim.get_rigid_object("fork")
        spoon = self.sim.get_rigid_object("spoon")
        plate = self.sim.get_rigid_object("plate")
        return (
            fork.get_local_pose(to_matrix=True),
            spoon.get_local_pose(to_matrix=True),
            plate.get_local_pose(to_matrix=True),
        )

    def update_diagnostic_state(self):
        """Track peak lift at every policy step so a short lift is not missed."""
        if not hasattr(self, "_diag_max_fork_z"):
            return
        fork_pose, spoon_pose, _ = self._object_poses()
        self._diag_max_fork_z = torch.maximum(
            self._diag_max_fork_z, fork_pose[:, 2, 3]
        )
        self._diag_max_spoon_z = torch.maximum(
            self._diag_max_spoon_z, spoon_pose[:, 2, 3]
        )
        plate_pose = self.sim.get_rigid_object("plate").get_local_pose(to_matrix=True)
        fork_xyz = fork_pose[:, :3, 3]
        spoon_xyz = spoon_pose[:, :3, 3]
        plate_xyz = plate_pose[:, :3, 3]
        fork_target_xy = plate_xyz[:, :2].clone()
        spoon_target_xy = plate_xyz[:, :2].clone()
        fork_target_xy[:, 1] += 0.16
        spoon_target_xy[:, 1] -= 0.16
        placement_now = (
            torch.linalg.vector_norm(fork_xyz[:, :2] - fork_target_xy, dim=-1)
            <= self.XY_TOLERANCE_M
        ) & (
            torch.linalg.vector_norm(spoon_xyz[:, :2] - spoon_target_xy, dim=-1)
            <= self.XY_TOLERANCE_M
        ) & (
            torch.abs(fork_xyz[:, 2] - plate_xyz[:, 2]) <= self.Z_TOLERANCE_M
        ) & (
            torch.abs(spoon_xyz[:, 2] - plate_xyz[:, 2]) <= self.Z_TOLERANCE_M
        )
        self._diag_target_streak = torch.where(
            placement_now,
            self._diag_target_streak + 1,
            torch.zeros_like(self._diag_target_streak),
        )

    def diagnostic_metrics(self):
        """Return per-environment placement metrics for trace logging."""
        fork_pose, spoon_pose, plate_pose = self._object_poses()
        fork_xyz = fork_pose[:, :3, 3]
        spoon_xyz = spoon_pose[:, :3, 3]
        plate_xyz = plate_pose[:, :3, 3]

        metrics = {
            "fork_xyz": fork_xyz,
            "spoon_xyz": spoon_xyz,
            "plate_xyz": plate_xyz,
        }
        if hasattr(self, "_diag_initial_fork_xyz"):
            metrics.update(
                fork_planar_displacement=torch.linalg.vector_norm(
                    fork_xyz[:, :2] - self._diag_initial_fork_xyz[:, :2], dim=-1
                ),
                spoon_planar_displacement=torch.linalg.vector_norm(
                    spoon_xyz[:, :2] - self._diag_initial_spoon_xyz[:, :2], dim=-1
                ),
                fork_lift=self._diag_max_fork_z - self._diag_initial_fork_xyz[:, 2],
                spoon_lift=self._diag_max_spoon_z - self._diag_initial_spoon_xyz[:, 2],
            )
        return metrics

    def is_task_success(self, *args, **kwargs):
        # Base reset asks for success before the first diagnostic state exists.
        if not hasattr(self, "_diag_initial_fork_xyz"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        metrics = self.diagnostic_metrics()
        fork_xyz = metrics["fork_xyz"]
        spoon_xyz = metrics["spoon_xyz"]
        plate_xyz = metrics["plate_xyz"]

        fork_target = plate_xyz.clone()
        spoon_target = plate_xyz.clone()
        fork_target[:, 1] += 0.16
        spoon_target[:, 1] -= 0.16

        fork_xy_ok = torch.linalg.vector_norm(
            fork_xyz[:, :2] - fork_target[:, :2], dim=-1
        ) <= self.XY_TOLERANCE_M
        spoon_xy_ok = torch.linalg.vector_norm(
            spoon_xyz[:, :2] - spoon_target[:, :2], dim=-1
        ) <= self.XY_TOLERANCE_M
        z_ok = (
            (torch.abs(fork_xyz[:, 2] - plate_xyz[:, 2]) <= self.Z_TOLERANCE_M)
            & (torch.abs(spoon_xyz[:, 2] - plate_xyz[:, 2]) <= self.Z_TOLERANCE_M)
        )
        lifted_ok = (
            (metrics["fork_lift"] >= self.MIN_LIFT_M)
            & (metrics["spoon_lift"] >= self.MIN_LIFT_M)
        )
        stable_ok = self._diag_target_streak >= self.MIN_TARGET_STREAK_STEPS
        return fork_xy_ok & spoon_xy_ok & z_ok & lifted_ok & stable_ok
