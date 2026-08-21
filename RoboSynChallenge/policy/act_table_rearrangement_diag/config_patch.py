"""In-memory config patches shared by diagnostic entry points."""


# Runtime CobotMagic ordering is left_arm=0..5, right_arm=6..11,
# left_eef=12/13, right_eef=14/15.  Keep the reset event on arm joints only.
CORRECT_ARM_JOINT_IDS = list(range(12))


def patch_random_arm_joint_ids(gym_config):
    events = gym_config.get("env", {}).get("events", {})
    event = events.get("random_robot_qpos")
    if not event:
        return False
    params = event.setdefault("params", {})
    params["joint_ids"] = list(CORRECT_ARM_JOINT_IDS)
    return True


def patch_grasp_z(value, target_offset):
    """Replace only the two fork/spoon grasp offsets in a copied config."""
    changed = 0
    if isinstance(value, dict):
        if (
            value.get("direction") == "z"
            and value.get("mode") == "extrinsic"
            and value.get("offset_value") == -0.008
        ):
            value["offset_value"] = float(target_offset)
            changed += 1
        for child in value.values():
            changed += patch_grasp_z(child, target_offset)
    elif isinstance(value, list):
        for child in value:
            changed += patch_grasp_z(child, target_offset)
    return changed
