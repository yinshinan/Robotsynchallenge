#!/usr/bin/env python
"""Run the table diagnostic while recording only selected episode indices."""

import eval_table_rearrangement_diag as diagnostic_eval


base_eval = diagnostic_eval.base_eval
_original_create_video_recorder = base_eval.create_video_recorder


class SelectiveRecorder:
    """Keep full sequential evaluation but encode video only for chosen episodes."""

    def __init__(self, recorder, selected_episodes):
        self._recorder = recorder
        self._selected_episodes = set(selected_episodes)
        self._active = False
        self.save_dir = recorder.save_dir

    def start_episode(self, episode_idx, seed):
        self._active = int(episode_idx) in self._selected_episodes
        if self._active:
            self._recorder.start_episode(episode_idx, seed)

    def record(self, obs):
        if self._active:
            self._recorder.record(obs)

    def close_episode(self, success=None):
        if self._active:
            self._recorder.close_episode(success=success)
        self._active = False


def create_video_recorder(config):
    recorder = _original_create_video_recorder(config)
    if recorder is None:
        return None
    raw = config.get("diag_video_episode_indices", (0, 1, 5))
    if isinstance(raw, (list, tuple)):
        selected = [int(value) for value in raw]
    else:
        selected = [
            int(value.strip()) for value in str(raw).split(",") if value.strip()
        ]
    print(f"[TABLE SELECTIVE VIDEO] episode_indices={selected}", flush=True)
    return SelectiveRecorder(recorder, selected)


base_eval.create_video_recorder = create_video_recorder


if __name__ == "__main__":
    base_eval.main()
