#!/usr/bin/env python
"""Compare corrected n=50 failures with fixed-seed n=5 reruns."""

import csv
from pathlib import Path

from policy.act_table_rearrangement_diag.analyze_multiseed import (
    load_console_results,
    load_episodes,
    summarize_episode,
    write_csv,
)


SEEDS = (209652396, 398764591, 1537364731)


def main():
    baseline_path = Path(
        "diagnostic_logs/table_rearrangement_multiseed_original_ids.csv"
    )
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    rows = [
        row
        for row in baseline
        if row["setting"] == "random" and int(row["seed"]) in SEEDS
    ]
    for row in rows:
        row["n_action_steps"] = 50

    for seed in SEEDS:
        stem = f"diagnostic_logs/random_original_ids_seed{seed}_physical_n5"
        episodes = load_episodes(Path(stem + ".jsonl"))
        results = load_console_results(Path(stem + ".console.log"))
        if list(episodes) != [0] or len(results) != 1:
            raise RuntimeError(f"Expected exactly one episode for seed {seed}")
        row = summarize_episode("random", 0, seed, episodes[0], results[0])
        row["n_action_steps"] = 5
        rows.append(row)

    field_order = ["n_action_steps"] + [key for key in rows[0] if key != "n_action_steps"]
    normalized = [{key: row.get(key, "") for key in field_order} for row in rows]
    output = Path("diagnostic_logs/table_rearrangement_failed_seeds_n50_n5.csv")
    write_csv(output, normalized)
    for seed in SEEDS:
        selected = sorted(
            (row for row in normalized if int(row["seed"]) == seed),
            key=lambda row: int(row["n_action_steps"]),
            reverse=True,
        )
        for row in selected:
            print(
                f"seed={seed} n={row['n_action_steps']} success={row['success']} "
                f"stage={row['failure_stage']} "
                f"fork_xy={float(row['final_fork_target_xy_error_m']):.4f} "
                f"spoon_xy={float(row['final_spoon_target_xy_error_m']):.4f}"
            )
    print(f"comparison_csv={output}")


if __name__ == "__main__":
    main()
