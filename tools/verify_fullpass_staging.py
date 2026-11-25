#!/usr/bin/env python3
"""Verify full-pass CPU staging on the largest MOT20 sequence; no training."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import audit_sharding_memory as memory_audit
import train_supervision_groups as training
from shutrack.training.feature_staging import staged_sequence_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = json.loads(args.audit.read_text())
    record = next(item for item in audit["datasets"] if item["config"] == str(args.config))
    case = next(item for item in record["cases"] if item["shards"] == 1)
    available = memory_audit.host_memory()["MemAvailable"]
    required = (case["max_staging_gib"] + 6.0) * 2**30
    if available < required:
        print(json.dumps({"status": "deferred", "available_gib": available / 2**30,
                          "required_gib": required / 2**30}), flush=True)
        return
    datasets = {}
    for name in config["sequence_names"]:
        data = training.SequenceData(name, config["split"], Path(config["data_root"]), config["dataset"])
        datasets[name] = data
    _, plans, _, _ = training.plan_phase(
        datasets, 0, 1, config["sequence_sampling"], config["sequence_sampling_power"],
        config["seed"], 1,
    )
    name = case["largest_sequence"]
    data = datasets[name]
    rows = plans[name]
    data.open_cache()
    data.load_rank_arrays()
    targets = training.derive_targets(data.samples, config["taus"])
    sample = rows[:16]
    reference = training.batch_tensors(data, sample, targets, config["group"], compact_reid=True)
    print(json.dumps({"status": "loading", "sequence": name, "samples": len(rows),
                      "estimated_staging_gib": case["max_staging_gib"],
                      "available_before_gib": available / 2**30}), flush=True)
    started = time.perf_counter()
    try:
        with staged_sequence_rows(data, rows, config["group"]) as staged:
            actual = training.batch_tensors(data, sample, targets, config["group"], compact_reid=True)
            for key in reference:
                np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
                assert actual[key].dtype == reference[key].dtype
            full_batch = training.batch_tensors(
                data, rows[:config["batch_size"]], targets, config["group"], compact_reid=True)
            report = {"status": "passed", "dataset": config["dataset"], "sequence": name,
                      "shards": 1, "samples": len(rows), "stage_gib": staged["bytes"] / 2**30,
                      "stage_seconds": staged["seconds"], "batch_values_equal": True,
                      "batch_size": len(rows[:config["batch_size"]]),
                      "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                      "memory_during": memory_audit.host_memory()}
            del actual, full_batch
        report["seconds"] = time.perf_counter() - started
        report["memory_after"] = memory_audit.host_memory()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        for value in datasets.values():
            value.close_cache()
            value.samples.close()
        gc.collect()


if __name__ == "__main__":
    main()
