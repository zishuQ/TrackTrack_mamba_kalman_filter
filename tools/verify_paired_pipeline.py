#!/usr/bin/env python3
"""Compare the paired GPU pipeline against a preserved trainer source.

Runs short, isolated updates only. It does not write training checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools", ROOT / "agentguard/src", ROOT / "3. Tracker"):
    sys.path.insert(0, str(path))

import train_supervision_groups as current
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.training.loss import appearance_column_loss, appearance_rank_loss, motion_box_diou_loss
from shutrack.training.paired_pipeline import DeferredMetrics, prepared_batches


def max_tensor_difference(left, right):
    if torch.is_tensor(left):
        return float((left.detach().cpu() - right.detach().cpu()).abs().max())
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((max_tensor_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        return max((max_tensor_difference(a, b) for a, b in zip(left, right)), default=0.0)
    assert left == right
    return 0.0


def run_case(module, data, targets, rows, group, model_config, batch_size, warmup, optimized):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = ShutrackV2Model(ShutrackV2Config(**model_config)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    metrics = DeferredMetrics(torch.device("cuda"))
    if optimized:
        context = prepared_batches(
            rows, batch_size,
            lambda selected: module.batch_tensors(data, selected, targets, group,
                                                 compact_reid=True),
            module.PAIRED_TENSOR_KEYS, prefetch=1, pin_memory=True)
    else:
        context = nullcontext((
            (rows[start:start + batch_size],
             module.batch_tensors(data, rows[start:start + batch_size], targets, group))
            for start in range(0, len(rows), batch_size)))
    elapsed = None
    torch.cuda.reset_peak_memory_stats()
    with context as batches:
        for index, (selected, cpu_batch) in enumerate(batches):
            batch = module.to_device(cpu_batch, torch.device("cuda"))
            outputs = model(**module.model_inputs(batch))
            motion, _ = motion_box_diou_loss(outputs, batch)
            appearance, components = appearance_rank_loss(
                outputs, batch, alpha=0.95, tau=0.1, rank_weight=18)
            if group == "appearance_row_column":
                stats = {}
                column = appearance_column_loss(
                    outputs, batch, alpha=0.95, tau=0.1, components=stats,
                    tensor_statistics=optimized)
                appearance = appearance + 50 * column
                if optimized:
                    metrics.add("column", column, len(selected))
                    metrics.add("column_valid", stats["valid"])
                    metrics.add("column_active", stats["active"])
                else:
                    float(column.detach())
                    int(stats["valid"])
                    int(stats["active"])
            if optimized:
                metrics.add("rank_mean", components["rank_mean"], len(selected))
                metrics.add("fallback_mean", components["fallback_mean"], len(selected))
            else:
                float(components["rank_mean"])
                float(components["fallback_mean"])
            loss = motion + appearance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            gate = outputs["gate_logits"].detach().sigmoid().mean(0)
            if optimized:
                for key, value in (("loss", loss), ("motion", motion), ("appearance", appearance),
                                   ("gate_motion", gate[0]), ("gate_appearance", gate[1])):
                    metrics.add(key, value, len(selected))
            else:
                float(loss.detach())
                float(motion.detach())
                float(appearance.detach())
                float(gate[0])
                float(gate[1])
            if index + 1 == warmup:
                torch.cuda.synchronize()
                started = time.perf_counter()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if optimized:
        metrics.summary()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    for item in optimizer_state["state"].values():
        for key, value in item.items():
            if torch.is_tensor(value):
                item[key] = value.detach().cpu().clone()
    result = {
        "group": group, "pipeline": "optimized" if optimized else "reference",
        "measured_steps": len(rows) // batch_size - warmup,
        "seconds": elapsed, "ms_per_step": elapsed * 1000 / (len(rows) // batch_size - warmup),
        "samples_per_second": (len(rows) - warmup * batch_size) / elapsed,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
        "last_loss": float(loss.detach()),
    }
    del model, optimizer, batch, outputs, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result, state, optimizer_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=("MOT20", "MOT17", "SportsMOT"))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the isolated throughput check")
    reference = types.ModuleType("paired_reference")
    reference.__file__ = str(ROOT / "tools/train_supervision_groups.py")
    source = Path(args.reference).read_text()
    exec(compile(source, reference.__file__, "exec"), reference.__dict__)
    root = Path(args.data_root)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taus = payload["training_config"]["taus"]
    sequences = [
        module.SequenceData(args.sequence, "all" if args.dataset != "SportsMOT" else "train",
                            root, args.dataset)
        for module in (reference, current)
    ]
    for data in sequences:
        data.open_cache()
        data.input_statistics = (np.asarray(payload["normalization_mean"]),
                                 np.asarray(payload["normalization_std"]))
    print("sequence handles and normalization ready", flush=True)
    targets = current.derive_targets(sequences[0].samples, taus)
    print("targets ready", flush=True)
    n = sequences[0].n
    seed = 42 + 1000 + int(hashlib.sha1(args.sequence.encode()).hexdigest()[:8], 16) % 100000
    order = np.random.RandomState(seed).permutation(max(1, n // 4))
    rows = np.resize(order, (args.warmup + args.steps) * args.batch_size)
    checks = {}
    reports = []
    try:
        for group in ("appearance_rank", "appearance_row_column"):
            equal = True
            for start in range(0, min(3 * args.batch_size, len(rows)), args.batch_size):
                selected = rows[start:start + args.batch_size]
                left = reference.to_device(
                    reference.batch_tensors(sequences[0], selected, targets, group),
                    torch.device("cpu"))
                right = current.to_device(
                    current.batch_tensors(sequences[1], selected, targets, group,
                                          compact_reid=True), torch.device("cpu"))
                equal = equal and left.keys() == right.keys() and all(
                    torch.equal(left[key], right[key]) if torch.is_tensor(left[key])
                    else np.array_equal(left[key], right[key]) for key in left)
                del left, right
            if not equal:
                raise AssertionError(f"{group}: real batch mismatch")
            print(json.dumps({"real_batches_equal": equal, "group": group}), flush=True)
            old, old_state, old_optimizer = run_case(
                reference, sequences[0], targets, rows, group, payload["model_config"],
                args.batch_size, args.warmup, False)
            print(json.dumps(old), flush=True)
            new, new_state, new_optimizer = run_case(
                current, sequences[1], targets, rows, group, payload["model_config"],
                args.batch_size, args.warmup, True)
            print(json.dumps(new), flush=True)
            parameter_diff = max_tensor_difference(old_state, new_state)
            optimizer_diff = max_tensor_difference(old_optimizer, new_optimizer)
            checks[group] = {
                "batch_equal": equal, "parameter_max_abs_diff": parameter_diff,
                "optimizer_max_abs_diff": optimizer_diff,
                "last_loss_abs_diff": abs(old["last_loss"] - new["last_loss"]),
                "speedup": old["ms_per_step"] / new["ms_per_step"],
            }
            if parameter_diff > 1e-6 or optimizer_diff > 1e-6:
                raise AssertionError(f"{group}: short-training state mismatch: {checks[group]}")
            reports.extend([old, new])
            del old_state, new_state, old_optimizer, new_optimizer
        report = {"dataset": args.dataset, "sequence": args.sequence,
                  "batch_size": args.batch_size, "warmup": args.warmup,
                  "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "checks": checks, "timings": reports,
                  "verifier_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"ok": True, "checks": checks, "out": args.out}), flush=True)
    finally:
        for data in sequences:
            data.close_cache()


if __name__ == "__main__":
    main()
