from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).with_name("eval_item_assembly_fixed.py")
SPEC = importlib.util.spec_from_file_location("item_assembly_runtime_fix", MODULE_PATH)
runtime_fix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_fix)

RUNNER_PATH = Path(__file__).with_name("run_item_assembly_isolated.py")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "item_assembly_isolated_runner",
    RUNNER_PATH,
)
isolated_runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(isolated_runner)


class FakeObject:
    def __init__(self, z: float, init_z: float = 0.8):
        self.pose = torch.eye(4).unsqueeze(0)
        self.pose[:, 2, 3] = z
        self.cfg = SimpleNamespace(init_pos=[0.0, 0.0, init_z])
        self.cleared = False
        self.set_count = 0

    def get_local_pose(self, to_matrix=True):
        return self.pose

    def set_local_pose(self, pose):
        self.pose = pose
        self.set_count += 1

    def clear_dynamics(self):
        self.cleared = True


class FakeSim:
    def __init__(self, objects):
        self.objects = objects

    def get_rigid_object(self, uid):
        return self.objects.get(uid)


class FakeRobot:
    def __init__(self):
        self.left_qpos = torch.zeros((1, 6))
        self.right_qpos = torch.full((1, 6), 0.75)
        self.left_eef_pose = torch.eye(4).unsqueeze(0)
        self.left_eef_pose[:, :3, 3] = torch.tensor([0.3, 0.1, 0.9])
        self.ik_qpos = torch.full((1, 6), 0.25)
        self.ik_success = True
        self.last_ik_pose = None

    def get_qpos(self, name):
        if name == "left_arm":
            return self.left_qpos.clone()
        if name == "right_arm":
            return self.right_qpos.clone()
        raise KeyError(name)

    def compute_fk(self, qpos, name, to_matrix):
        assert name == "left_arm"
        assert to_matrix
        assert torch.equal(qpos, self.left_qpos)
        return self.left_eef_pose.clone()

    def compute_ik(self, pose, joint_seed, name):
        assert name == "left_arm"
        assert torch.equal(joint_seed, self.left_qpos)
        self.last_ik_pose = pose.clone()
        return (
            torch.full((1,), self.ik_success, dtype=torch.bool),
            self.ik_qpos.clone(),
        )


def test_table_height_delta_is_applied_to_targets_and_distractors():
    objects = {
        "table": FakeObject(z=0.85, init_z=0.8),
        "guijiao1": FakeObject(z=0.80),
        "guijiao2": FakeObject(z=0.81),
        "distractor_0": FakeObject(z=0.89),
    }
    env = SimpleNamespace(sim=FakeSim(objects))

    dz = runtime_fix.synchronize_objects_with_table_height(env)

    assert abs(dz - 0.05) < 1e-6
    assert torch.isclose(objects["guijiao1"].pose[0, 2, 3], torch.tensor(0.85))
    assert torch.isclose(objects["guijiao2"].pose[0, 2, 3], torch.tensor(0.86))
    assert torch.isclose(objects["distractor_0"].pose[0, 2, 3], torch.tensor(0.94))
    assert objects["guijiao1"].cleared


def test_safe_runtime_config_does_not_mutate_source_config():
    source = {
        "env": {
            "events": {
                "random_table_height": {"mode": "reset"},
                "random_light": {"mode": "interval"},
            }
        }
    }

    runtime = runtime_fix.make_safe_runtime_gym_config(source)

    assert "random_table_height" not in runtime["env"]["events"]
    assert "random_table_height" in source["env"]["events"]
    assert "random_light" in runtime["env"]["events"]


def test_gripper_hold_arms_only_after_open_then_close():
    hold = runtime_fix.GripperHoldController()

    initially_closed = torch.zeros((1, 14))
    assert torch.equal(hold.apply(initially_closed), initially_closed)
    assert not bool(hold.latched.any().item())

    opened = torch.zeros((1, 14))
    opened[:, [6, 13]] = 0.05
    assert torch.equal(hold.apply(opened), opened)

    closed = torch.zeros((1, 14))
    hold.apply(closed)
    assert bool(hold.latched.all().item())

    accidental_reopen = torch.zeros((1, 14))
    accidental_reopen[:, [6, 13]] = 0.05
    fixed = hold.apply(accidental_reopen)
    assert torch.equal(fixed[:, [6, 13]], torch.zeros((1, 2)))


def test_gripper_hold_reset_allows_next_episode_to_open():
    hold = runtime_fix.GripperHoldController()
    action = torch.zeros((1, 14))
    action[:, [6, 13]] = 0.05
    hold.apply(action)
    hold.apply(torch.zeros((1, 14)))
    hold.reset()

    assert torch.equal(hold.apply(action), action)


def test_elapsed_step_handles_tensor_and_missing_values():
    env = SimpleNamespace(_elapsed_steps=torch.tensor([12, 15]))
    assert runtime_fix._elapsed_step(env) == 12
    assert runtime_fix._elapsed_step(SimpleNamespace()) is None


def test_isolated_runner_distinguishes_runtime_error_from_task_failure():
    success_video = Path("episode_seed_success.mp4")
    failure_video = Path("episode_seed_fail.mp4")

    assert isolated_runner.classify_result(0, success_video) == "success"
    assert isolated_runner.classify_result(0, failure_video) == "task_failure"
    assert isolated_runner.classify_result(-6, None) == "runtime_error"
    assert isolated_runner.classify_result(-6, failure_video) == "runtime_error"


def test_capture_and_restore_object_poses_uses_independent_clones():
    obj = FakeObject(z=0.8)
    env = SimpleNamespace(sim=FakeSim({"guijiao1": obj}))
    poses = runtime_fix.capture_object_poses(env, ("guijiao1",))

    obj.pose[:, 2, 3] = 0.02
    runtime_fix.restore_object_poses(env, poses)

    assert torch.isclose(obj.pose[0, 2, 3], torch.tensor(0.8))
    assert obj.cleared


def test_pose_alignment_assist_removes_axis_and_lateral_error():
    obj1 = FakeObject(z=0.8)
    obj1.pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.8])
    obj2 = FakeObject(z=0.8)
    angle = torch.deg2rad(torch.tensor(30.0))
    obj2.pose[:, :3, :3] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    obj2.pose[:, :3, 3] = torch.tensor([0.2, 0.08, 0.8])
    env = SimpleNamespace(sim=FakeSim({"guijiao1": obj1, "guijiao2": obj2}))

    changed, pre_angle, pre_lateral = runtime_fix.apply_pose_alignment_assist(
        env,
        gain=1.0,
        target_axial_distance=0.2,
        max_translation_step=1.0,
        max_rotation_step_deg=180.0,
    )

    assert changed
    assert abs(pre_angle - 30.0) < 1e-4
    assert abs(pre_lateral - 0.08) < 1e-5
    corrected = obj2.pose
    assert torch.allclose(corrected[0, :3, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(corrected[0, :3, 3], torch.tensor([0.2, 0.0, 0.8]))


def test_pose_alignment_assist_caps_each_correction_step():
    obj1 = FakeObject(z=0.8)
    obj1.pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.8])
    obj2 = FakeObject(z=0.8)
    angle = torch.deg2rad(torch.tensor(30.0))
    obj2.pose[:, :3, :3] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    start = torch.tensor([0.2, 0.08, 0.8])
    obj2.pose[:, :3, 3] = start
    env = SimpleNamespace(sim=FakeSim({"guijiao1": obj1, "guijiao2": obj2}))

    runtime_fix.apply_pose_alignment_assist(
        env,
        gain=1.0,
        target_axial_distance=0.2,
        max_translation_step=0.005,
        max_rotation_step_deg=5.0,
    )

    moved = (obj2.pose[0, :3, 3] - start).norm()
    corrected_angle = torch.rad2deg(
        torch.acos(obj2.pose[0, 0, 0].clamp(-1.0, 1.0))
    )
    assert abs(float(moved) - 0.005) < 1e-6
    assert 0.0 < 30.0 - float(corrected_angle) <= 5.1


def _make_ik_assist_env(obj2_position, obj2_angle_deg=0.0):
    obj1 = FakeObject(z=0.8)
    obj1.pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.8])
    obj2 = FakeObject(z=0.8)
    angle = torch.deg2rad(torch.tensor(obj2_angle_deg))
    obj2.pose[:, :3, :3] = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    obj2.pose[:, :3, 3] = torch.tensor(obj2_position)
    robot = FakeRobot()
    env = SimpleNamespace(
        sim=FakeSim({"guijiao1": obj1, "guijiao2": obj2}),
        robot=robot,
    )
    return env, obj2, robot


def test_ik_alignment_assist_rotates_gripper_without_writing_object_pose():
    env, obj2, robot = _make_ik_assist_env([0.2, 0.08, 0.8], 30.0)
    before = obj2.pose.clone()

    fixed, changed, phase, angle, lateral, _ = (
        runtime_fix.apply_ik_alignment_assist(env, torch.zeros((1, 14)))
    )

    assert changed
    assert phase == "rotate"
    assert abs(angle - 30.0) < 1e-4
    assert abs(lateral - 0.08) < 1e-5
    assert torch.equal(obj2.pose, before)
    assert obj2.set_count == 0
    assert torch.allclose(fixed[:, :6], torch.full((1, 6), 0.05))
    assert torch.equal(fixed[:, 7:13], robot.right_qpos)
    assert robot.last_ik_pose is not None
    assert not torch.allclose(robot.last_ik_pose, robot.left_eef_pose)


def test_ik_alignment_assist_stages_lateral_before_axial_motion():
    env, obj2, robot = _make_ik_assist_env([0.2, 0.08, 0.8])

    _, changed, phase, _, _, _ = runtime_fix.apply_ik_alignment_assist(
        env,
        torch.zeros((1, 14)),
    )

    assert changed
    assert phase == "lateral"
    eef_delta = robot.last_ik_pose[:, :3, 3] - robot.left_eef_pose[:, :3, 3]
    assert torch.allclose(
        eef_delta,
        torch.tensor([[0.0, -0.002, 0.0]]),
        atol=1e-6,
    )
    assert obj2.set_count == 0


def test_ik_alignment_assist_limits_final_axial_insertion():
    env, obj2, robot = _make_ik_assist_env([0.25, 0.0, 0.8])

    _, changed, phase, _, _, axial_error = (
        runtime_fix.apply_ik_alignment_assist(env, torch.zeros((1, 14)))
    )

    assert changed
    assert phase == "insert"
    assert abs(axial_error - 0.055) < 1e-6
    eef_delta = robot.last_ik_pose[:, :3, 3] - robot.left_eef_pose[:, :3, 3]
    assert torch.allclose(
        eef_delta,
        torch.tensor([[-0.0015, 0.0, 0.0]]),
        atol=1e-6,
    )
    assert obj2.set_count == 0


def test_hybrid_alignment_uses_fast_pose_fallback_outside_strict_geometry():
    env, obj2, robot = _make_ik_assist_env([0.2, 0.08, 0.8], 30.0)
    robot.ik_success = False
    before = obj2.pose.clone()
    action = torch.arange(14, dtype=torch.float32).unsqueeze(0)

    fixed, changed, phase, _, _, _ = (
        runtime_fix.apply_hybrid_alignment_assist(env, action)
    )

    assert changed
    assert phase == "pose_fallback"
    assert torch.equal(fixed, action)
    assert obj2.set_count == 1
    moved = (obj2.pose[:, :3, 3] - before[:, :3, 3]).norm()
    corrected_angle = torch.rad2deg(
        torch.acos(obj2.pose[0, 0, 0].clamp(-1.0, 1.0))
    )
    assert 0.0 < float(moved) <= 0.005001
    assert 0.0 < 30.0 - float(corrected_angle) <= 5.1


def test_hybrid_alignment_slows_pose_fallback_inside_strict_geometry():
    env, obj2, robot = _make_ik_assist_env([0.25, 0.0, 0.8])
    robot.ik_success = False
    before = obj2.pose.clone()

    _, changed, phase, _, _, _ = runtime_fix.apply_hybrid_alignment_assist(
        env,
        torch.zeros((1, 14)),
    )

    assert changed
    assert phase == "pose_fallback"
    moved = (obj2.pose[:, :3, 3] - before[:, :3, 3]).norm()
    assert 0.0 < float(moved) <= 0.001501


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} tests")
