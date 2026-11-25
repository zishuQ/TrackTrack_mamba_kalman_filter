#!/usr/bin/env python3
"""From-scratch preflight for the SportsMOT cost / cost+rank experiment.

Checks:
  - gate-target / rank-label / source-row fingerprints and checksums, with
    keep/suppress/gain/damage recomputed exactly;
  - seed-42 initialization is reproducible (two fresh constructions hash equal);
  - all 89 parameter tensors require grad and are covered by the optimizer;
  - the full training loss on a real batch actually contains the motion cost
    term, the motion rank term, appearance rank and weighted column losses;
  - full-parameter A/B benchmark on the same real batch after warm-up:
    per-step time, CUDA peak memory, RSS.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from train_motion_cost_bce import cost_motion_loss
from train_motion_rank import pairwise_rank_loss
from shutrack.training.loss import appearance_column_loss, appearance_rank_loss
from shutrack.training.motion_targets import (
    GRID, SOURCE_KEYS, TIE_TOLERANCE, numpy_diou, source_fingerprint,
)


def rss_mib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def state_hash(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_model(config, reid_dim, seed):
    trainer._seed_everything(int(seed))
    return trainer.ShutrackV2Model(trainer.ShutrackV2Config(
        reid_dim=int(reid_dim),
        appearance_proj_dim=int(config["appearance_proj_dim"]),
        architecture_variant=config["architecture_variant"],
    ))


def full_loss(outputs, batch, config, rank_weight, generator):
    motion, _ = cost_motion_loss(outputs, batch, 0.25)
    rank, _ = pairwise_rank_loss(outputs, batch, rank_weight, generator)
    appearance, _ = appearance_rank_loss(
        outputs, batch, alpha=float(config["rank_alpha"]), tau=float(config["rank_tau"]),
        rank_weight=float(config["rank_weight"]))
    column_components = {}
    column = appearance_column_loss(
        outputs, batch, alpha=float(config["rank_alpha"]), tau=float(config["rank_tau"]),
        hinge=bool(config["column_hinge"]), components=column_components,
        tensor_statistics=True)
    weighted_column = float(config["column_weight"]) * column
    total = motion + rank + appearance + weighted_column
    components = {
        "motion": float(motion.detach()), "rank": float(rank.detach()),
        "appearance": float(appearance.detach()), "column": float(column.detach()),
        "column_weighted": float(weighted_column.detach()), "total": float(total.detach()),
        "column_valid": int(column_components.get("valid", 0)),
        "column_active": int(column_components.get("active", 0)),
    }
    return total, components


def benchmark(checkpoint_meta, config, reid_dim, batch, steps, warmup, rank_weight,
              seed, device):
    model = build_model(config, reid_dim, seed)
    model.to(device)
    model.train()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(20261122)

    def compute():
        outputs = model(**trainer.model_inputs(batch))
        return full_loss(outputs, batch, config, rank_weight, generator)

    with torch.no_grad():
        initial, components = compute()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    result = {
        "initial_loss": float(initial.detach()),
        "initial_components": components,
        "step_seconds_mean": float(np.mean(times)),
        "step_seconds_median": float(np.median(times)),
        "step_seconds_min": float(np.min(times)),
        "step_seconds_max": float(np.max(times)),
        "cuda_peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                    if device.type == "cuda" else None),
        "cuda_peak_reserved_mib": (torch.cuda.max_memory_reserved() / 2 ** 20
                                   if device.type == "cuda" else None),
        "process_rss_mib_after": rss_mib(),
    }
    del optimizer, parameters, model, initial
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--rank-root", required=True)
    parser.add_argument("--norm-checkpoint", required=True)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if not config.get("from_scratch") or config.get("init") is not None:
        raise ValueError("This preflight requires from_scratch=true and init=null")
    data_root = Path(config["data_root"])
    sequence = args.sequence or config["sequence_names"][0]
    split = config.get("split", "train")
    dataset = config.get("dataset", "SportsMOT")
    group = config["group"]
    if split not in ("train", "val", "all"):
        # The trainval data root carries split="trainval" in summary.json,
        # while the detection cache is stored under train/ or val/.
        split = ("val" if sequence in trainer.official_sequences("val", dataset)
                 else "train")

    seq = trainer.SequenceData(sequence, split, data_root, dataset)
    target_dir = Path(args.target_root) / sequence
    rank_dir = Path(args.rank_root) / sequence
    target_meta = json.loads((target_dir / "metadata.json").read_text())
    rank_meta = json.loads((rank_dir / "metadata.json").read_text())
    source = {key: seq.small[key] if key in seq.small else seq.samples[key]
              for key in SOURCE_KEYS}
    fingerprint_ok = target_meta["source_sha256"] == source_fingerprint(source)
    target = np.load(target_dir / "motion_gate_target.npy", allow_pickle=False)
    target_checksum_ok = target_meta["target_sha256"] == hashlib.sha256(target.tobytes()).hexdigest()
    rank_binding_ok = (rank_meta["source_sha256"] == target_meta["source_sha256"]
                       and rank_meta["target_sha256"] == target_meta["target_sha256"])
    label = np.load(rank_dir / "motion_rank_label.npy", allow_pickle=False)
    gain = np.load(rank_dir / "motion_rank_gain.npy", allow_pickle=False)
    damage = np.load(rank_dir / "motion_rank_damage.npy", allow_pickle=False)
    checksums_ok = all(
        rank_meta[key + "_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
        for key, array in (("label", label), ("gain", gain), ("damage", damage)))
    shapes_ok = all(array.shape == (seq.n,) for array in (label, gain, damage))

    rows = np.arange(min(args.rows, seq.n), dtype=np.int64)
    seq.open_cache()
    norm_meta = torch.load(args.norm_checkpoint, map_location="cpu", weights_only=False)
    seq.input_statistics = (
        np.asarray(norm_meta["normalization_mean"], dtype=np.float32),
        np.asarray(norm_meta["normalization_std"], dtype=np.float32),
    )
    taus = trainer.load_global_taus([sequence], data_root)
    targets = trainer.derive_targets(seq.samples, taus)
    batch_np = trainer.batch_tensors(seq, rows, targets, group, compact_reid=True)
    batch_np["motion_gate_target"] = target[rows]
    batch_np["motion_rank_label"] = label[rows]
    batch_np["motion_rank_gain"] = gain[rows]
    batch_np["motion_rank_damage"] = damage[rows]
    valid_np = batch_np["motion_gt_valid"].astype(bool)
    g0 = batch_np["pred_box_g0"].astype(np.float64)
    g1 = batch_np["pred_box_g1"].astype(np.float64)
    gt = batch_np["gt_box_t1"].astype(np.float64)
    star = np.asarray(batch_np["motion_gate_target"], dtype=np.float64)
    loss1 = 1.0 - numpy_diou(g1, gt)
    loss_half = 1.0 - numpy_diou(g0 + 0.5 * (g1 - g0), gt)
    loss_best = 1.0 - numpy_diou(g0 + star[:, None] * (g1 - g0), gt)
    keep_ref = valid_np & (star >= 1.0 - 1e-6) & (loss_half > loss1 + 1e-6)
    suppress_ref = valid_np & (star < 1.0 - 1e-6) & (loss1 > loss_best + 1e-6)
    stored_label = np.asarray(batch_np["motion_rank_label"])
    binding = {
        "fingerprint_ok": bool(fingerprint_ok),
        "target_checksum_ok": bool(target_checksum_ok),
        "rank_binding_ok": bool(rank_binding_ok),
        "rank_checksums_ok": bool(checksums_ok),
        "rank_shapes_ok": bool(shapes_ok),
        "keep_rows_match": bool(np.array_equal(keep_ref, stored_label == 1)),
        "suppress_rows_match": bool(np.array_equal(suppress_ref, stored_label == 2)),
        "gain_match": bool(np.allclose(np.asarray(batch_np["motion_rank_gain"]),
                                       np.where(suppress_ref, loss1 - loss_best, 0.0), atol=1e-6)),
        "damage_match": bool(np.allclose(np.asarray(batch_np["motion_rank_damage"]),
                                         np.where(keep_ref, loss_half - loss1, 0.0), atol=1e-6)),
    }
    assert all(binding.values())

    reid_dim = int(seq.event_track.shape[1])
    model_a = build_model(config, reid_dim, args.seed)
    hash_a = state_hash(model_a)
    model_b = build_model(config, reid_dim, args.seed)
    hash_b = state_hash(model_b)
    init_equal = hash_a == hash_b
    total = int(sum(p.numel() for p in model_a.parameters()))
    tensors = sum(1 for _ in model_a.parameters())
    all_trainable = all(p.requires_grad for p in model_a.parameters())
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-4, weight_decay=1e-4)
    covered = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [name for name, p in model_a.named_parameters() if id(p) not in covered]
    del optimizer, model_a, model_b
    gc.collect()

    device = torch.device(args.device)
    batch = trainer.to_device(batch_np, device)
    for key in ("motion_gate_target", "motion_rank_label",
                "motion_rank_gain", "motion_rank_damage"):
        batch[key] = torch.from_numpy(np.asarray(batch_np[key])).to(device)

    benchmark_result = {}
    for name, weight in (("cost", 0.0), ("cost_rank", 1.0)):
        benchmark_result[name] = benchmark(
            norm_meta, config, reid_dim, batch, args.steps, args.warmup,
            weight, args.seed, device)
    benchmark_result["time_ratio_cost_over_cost_rank"] = (
        benchmark_result["cost"]["step_seconds_mean"]
        / max(1e-12, benchmark_result["cost_rank"]["step_seconds_mean"]))

    report = {
        "sequence": sequence, "rows": int(len(rows)), "split": split,
        "device": str(device), "seed": args.seed,
        "init": {"hash": hash_a, "hash_prefix": hash_a[:16], "equal": init_equal,
                 "parameter_tensors": tensors, "model_total": total,
                 "all_parameters_requires_grad": all_trainable,
                 "optimizer_parameter_tensors": len(covered),
                 "optimizer_covers_all": not missing, "optimizer_missing": missing},
        "binding": binding,
        "counts": {"valid": int(valid_np.sum()), "keep": int(keep_ref.sum()),
                   "suppress": int(suppress_ref.sum())},
        "loss_on_real_batch": benchmark_result["cost_rank"]["initial_components"],
        "benchmark": benchmark_result,
        "policy": {"base_lambda": 0.25, "rank_weight_full": 1.0,
                   "appearance_rank_weight": config["rank_weight"],
                   "column_weight": config["column_weight"],
                   "grid": GRID.tolist(), "tie_tolerance": TIE_TOLERANCE},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    seq.close_cache()
    seq.samples.close()


if __name__ == "__main__":
    main()
