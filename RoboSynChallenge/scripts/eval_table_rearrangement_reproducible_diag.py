#!/usr/bin/env python
"""Table-rearrangement diagnostic with reset-order-independent seeding.

Only copied configuration dictionaries are changed.  Official task and config
files remain untouched.
"""

import random
from copy import deepcopy

import numpy as np

import eval_table_rearrangement_diag as diagnostic_eval


base_eval = diagnostic_eval.base_eval
_original_find_gym_config = base_eval.find_gym_config
_original_recording_proxy = base_eval.RecordingEnvProxy


def find_gym_config(config):
    gym_config = deepcopy(_original_find_gym_config(config))
    events = gym_config["env"]["events"]
    distractor = events.pop("randomize_distractor_slots", None)
    if distractor is None:
        raise RuntimeError("randomize_distractor_slots event was not found")
    # Dict order is reset execution order.  Sampling distractors after the
    # utensils prevents old-episode utensil poses from changing RNG draw count.
    events["randomize_distractor_slots"] = distractor
    print(
        "[TABLE REPRO RESET]",
        "python_numpy_seeded=True",
        "distractor_event_moved_after_utensil_init=True",
        flush=True,
    )
    return gym_config


class FullySeededRecordingEnvProxy(_original_recording_proxy):
    def reset(self, *args, **kwargs):
        seed = kwargs.get("seed")
        if seed is None and args:
            seed = args[0]
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
        return super().reset(*args, **kwargs)


base_eval.find_gym_config = find_gym_config
base_eval.RecordingEnvProxy = FullySeededRecordingEnvProxy


if __name__ == "__main__":
    base_eval.main()
