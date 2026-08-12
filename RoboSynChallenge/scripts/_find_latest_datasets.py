import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Find the latest generated datasets for a specific task.")
    parser.add_argument("task_name", type=str, help="The name of the task to search datasets for")
    parser.add_argument("--count", type=int, default=2, help="Number of latest datasets to find")
    args = parser.parse_args()

    task_name = args.task_name
    count = args.count

    # Common roots where datasets might be saved
    possible_roots = [
        Path(f"lerobot_dataset/{task_name}")
    ]

    all_datasets = []
    for root in possible_roots:
        if root.exists():
            # Look for typical dataset subdirectories directly under task_name
            for d in root.iterdir():
                if d.is_dir() and (d / "meta" / "info.json").exists():
                    all_datasets.append(d)

    # Sort by creation / modification time (most recent first)
    all_datasets.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    if len(all_datasets) >= count:
        # We want the repo_id relative to HF_LEROBOT_HOME.
        # HF_LEROBOT_HOME is lerobot_dataset/
        # So repo_id is task_name/dataset_folder

        # Take the most recent `count` datasets
        selected = all_datasets[:count]
        # Reverse them so the older ones (e.g. clear) are printed before newer ones (e.g. random)
        selected.reverse()

        for d in selected:
            print(f"{task_name}/{d.name}")
    else:
        print("ERROR_NOT_ENOUGH_DATASETS")
        sys.exit(1)

if __name__ == "__main__":
    main()
