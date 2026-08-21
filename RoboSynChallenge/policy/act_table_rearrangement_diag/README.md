# table_rearrangement diagnostic adapter

This directory is an isolated diagnostic path. It does not modify the official
ACT adapter, task class, gym config, or action config.

It provides:

- physical gripper commands (`0..0.05 m`) with a legacy A/B mode;
- configurable `n_action_steps`;
- in-memory fork/spoon grasp-height overrides;
- corrected random-scene arm joint IDs (`0..5` and `8..13`);
- JSONL traces of raw action, environment action, normalized observation, and
  physical gripper qpos, plus wrist-link-to-utensil distances;
- a strict success condition requiring full XY/Z placement, verified lift, and
  five consecutive control frames inside the target region.

The runner executes four policy cases (`legacy/physical`, then `n=50/5/1`) and
three expert-replay grasp-height cases (`-0.008/0/+0.005 m`). It also writes
`diagnostic_logs/table_rearrangement_ab_summary.csv` automatically. Grasp height
is intentionally tested with expert replay because ACT inference does not read
the action-generation config. Expert results are written to
`diagnostic_logs/table_rearrangement_expert_z_summary.csv`.

Run the four-case controlled comparison:

```bash
bash policy/act_table_rearrangement_diag/eval_ab.sh \
  outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/080000/pretrained_model \
  0 924231285
```

Run one custom case:

```bash
policy/act/.venv/bin/python scripts/eval_table_rearrangement_diag.py \
  --config policy/act_table_rearrangement_diag/deploy_policy.yml \
  --overrides \
  --checkpoint_path outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/080000/pretrained_model \
  --n_action_steps 1 \
  --diag_grasp_z_offset_m 0.005 \
  --diag_log_path diagnostic_logs/n1_zplus005.jsonl
```
