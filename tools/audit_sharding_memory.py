#!/usr/bin/env python3
"""Estimate shard staging RAM using real sample indices, without loading ReID matrices."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as training


def host_memory():
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] in ("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"):
                values[fields[0][:-1]] = int(fields[1]) * 1024
    return values


def selected_bytes(source, count):
    return int(count) * int(np.prod(source.shape[1:])) * source.dtype.itemsize + source.shape[0] * 8


def estimate(data, rows, group):
    events = np.unique(np.maximum(np.concatenate((
        data.small["window_indices"][rows].reshape(-1),
        data.small["current_event_index"][rows],
    )), 0))
    detection_ids = np.asarray(data.event_det[events], dtype=np.int64)
    if group == "appearance_row_column":
        detection_ids = np.concatenate((detection_ids, data.small["column_det"][rows]))
    detection_ids = np.unique(detection_ids[detection_ids >= 0])
    rank_ids = data.rank_pos[rows]
    rank_ids = np.unique(rank_ids[rank_ids >= 0])
    track = selected_bytes(data.event_track, len(events))
    detection = selected_bytes(data.cache.features, len(detection_ids))
    rank = sum(selected_bytes(array, len(rank_ids)) for array in data._rank_arrays)
    return {
        "samples": len(rows), "events": len(events), "detections": len(detection_ids),
        "rank_rows": len(rank_ids), "track_bytes": track,
        "detection_bytes": detection, "rank_bytes": rank,
        "staging_bytes": track + detection + rank,
    }


def audit(config_path):
    config = json.loads(config_path.read_text())
    datasets = {}
    labels_bytes = 0
    source_bytes = 0
    result = {"dataset": config["dataset"], "config": str(config_path), "cases": []}
    for name in config["sequence_names"]:
        split = config["split"]
        if config["dataset"] not in ("MOT17", "MOT20") and split == "trainval":
            matches = [s for s in ("train", "val")
                       if (training.CACHE / config["dataset"] / s / name / "manifest.json").is_file()]
            if len(matches) != 1:
                raise ValueError(f"{name}: cannot resolve original train/val cache")
            split = matches[0]
        data = training.SequenceData(name, split, Path(config["data_root"]), config["dataset"])
        for key in ("rank_template0", "rank_detection", "rank_positive", "rank_competitor", "rank_score"):
            if data.samples.zip.getinfo(key + ".npy").compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{name}/{key}: compressed member; skip to avoid allocating full features")
        data.open_cache()
        data.load_rank_arrays()
        datasets[name] = data
        labels_bytes += sum(array.nbytes for array in data.small.values()) + data.rank_pos.nbytes + 14 * data.n
        source_bytes += data.event_track.nbytes + data.cache.features.nbytes + sum(a.nbytes for a in data._rank_arrays)
    try:
        for shards in sorted(set((int(config["shards"]), 1)), reverse=True):
            sequences = {}
            for shard in range(shards):
                order, plans, _, _ = training.plan_phase(
                    datasets, shard, shards, config["sequence_sampling"],
                    config["sequence_sampling_power"], int(config["seed"]), shard + 1,
                )
                for name in order:
                    entry = estimate(datasets[name], plans[name], config["group"])
                    if name not in sequences or entry["staging_bytes"] > sequences[name]["staging_bytes"]:
                        sequences[name] = entry
                del plans
                gc.collect()
            ordered = sorted(sequences.items(), key=lambda x: x[1]["staging_bytes"], reverse=True)
            result["cases"].append({
                "shards": shards, "largest_sequence": ordered[0][0],
                "max_staging_gib": ordered[0][1]["staging_bytes"] / 2**30,
                "sequences": {name: {**entry, "staging_gib": entry["staging_bytes"] / 2**30}
                              for name, entry in sequences.items()},
            })
        result["preloaded_labels_estimate_gib"] = labels_bytes / 2**30
        result["source_feature_working_set_gib"] = source_bytes / 2**30
        result["samples"] = sum(data.n for data in datasets.values())
        result["sequence_count"] = len(datasets)
        return result
    finally:
        for data in datasets.values():
            data.close_cache()
            data.samples.close()
        datasets.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = {"method": "real sampler and unique indices; feature values never staged",
              "host_memory_before": host_memory(), "datasets": []}
    for config in args.config:
        item = audit(config)
        report["datasets"].append(item)
        print(json.dumps({"dataset": item["dataset"], "samples": item["samples"],
                          "labels_gib": item["preloaded_labels_estimate_gib"],
                          "source_features_gib": item["source_feature_working_set_gib"],
                          "cases": [{k: x[k] for k in ("shards", "largest_sequence", "max_staging_gib")}
                                    for x in item["cases"]]}), flush=True)
    report["host_memory_after"] = host_memory()
    report["audit_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "peak_rss_gib": report["audit_peak_rss_gib"],
                      "host_memory": report["host_memory_after"]}), flush=True)


if __name__ == "__main__":
    main()
