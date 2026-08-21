#!/usr/bin/env python
"""Create a v2.1 training copy while leaving a v3.0 source dataset untouched."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_converter_path = REPO_ROOT / "scripts/convert_lerobot3.0_to_2.1.py"
_converter_spec = importlib.util.spec_from_file_location(
    "robosyn_lerobot_v30_to_v21", _converter_path
)
if _converter_spec is None or _converter_spec.loader is None:
    raise ImportError(f"Cannot load converter: {_converter_path}")
converter = importlib.util.module_from_spec(_converter_spec)
_converter_spec.loader.exec_module(converter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    converter.validate_local_dataset_version(source)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")
    destination.mkdir(parents=True)
    try:
        episodes = converter.load_episode_records(source)
        info = converter.load_info(source)
        chunks_size = int(info.get("chunks_size", 1000))
        video_keys = [
            key for key, feature in info["features"].items()
            if feature.get("dtype") == "video"
        ]
        converter.convert_info(source, destination, episodes, video_keys)
        converter.copy_global_stats(source, destination)
        converter.convert_tasks(source, destination)
        converter.convert_data(source, destination, episodes, chunks_size)
        converter.convert_videos(source, destination, episodes, video_keys, chunks_size)
        converter.convert_episodes_metadata(destination, episodes)
        converter.copy_ancillary_directories(source, destination)
        provenance = source / "sample_loading_v2_provenance.json"
        if provenance.is_file():
            shutil.copy2(provenance, destination / provenance.name)
    except Exception:
        # Destination is newly created by this command; remove only this partial copy.
        shutil.rmtree(destination)
        raise
    print(f"[V2 READONLY CONVERT] source={source}")
    print(f"[V2 READONLY CONVERT] destination={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
