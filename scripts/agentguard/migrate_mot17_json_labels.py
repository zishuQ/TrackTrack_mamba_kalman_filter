#!/usr/bin/env python3
"""Migrate the retained MOT17 v3 JSON labels to compact label arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentguard.data.compact_iwg_labels import (
    convert_json_label_directory_to_compact,
)
from agentguard.datasets.iwg_rg_cma_dataset import MOT17_FRCNN_ALL_SEQUENCES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MOT17 candidate-A JSON labels to compact arrays."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--event-cache-root",
        default="outputs/agentguard/event_cache_v3_iwg_v2",
    )
    parser.add_argument(
        "--detection-cache-root",
        default="outputs/agentguard/detection_cache",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    summary = convert_json_label_directory_to_compact(
        dataset="MOT17",
        source_root=(root / args.source_dir).resolve()
        if not Path(args.source_dir).is_absolute()
        else args.source_dir,
        output_root=(root / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else args.output_dir,
        sequences=list(MOT17_FRCNN_ALL_SEQUENCES),
        event_cache_root=(root / args.event_cache_root).resolve()
        if not Path(args.event_cache_root).is_absolute()
        else args.event_cache_root,
        detection_cache_root=(root / args.detection_cache_root).resolve()
        if not Path(args.detection_cache_root).is_absolute()
        else args.detection_cache_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
