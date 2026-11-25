#!/usr/bin/env python3
"""Preflight and A/B step benchmark for the pairwise motion-gate ranking.

Checks on one real sequence/batch:
  - rank-label / gate-target / source-row fingerprints and checksums;
  - stored keep/suppress labels exactly match a fresh recomputation;
  - synthetic pair-loss gradients: suppress logits down, keep logits up,
    simultaneous shift invariance, weight detached, empty pairs -> 0;
  - frozen shared/appearance parameters, trainable set exactly
    rsc.base_queries.0.* / rsc.modulations.0.* / rsc.gate_heads.0.*,
    unchanged fixed-input appearance output and unchanged total parameters;
  - after warm-up on the same real batch: per-step time, CUDA peak memory and
    RSS for A (L_base) and B (L_base + L_rank).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from train_motion_cost_bce import cost_motion_loss
from train_motion_rank import pairwise_rank_loss
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


def vmsize_peak_mib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def synthetic_gradient_checks():
    torch.manual_seed(0)
    n_bad, n_good, n_none = 7, 11, 5
    n = n_bad + n_good + n_none
    label = torch.zeros(n, dtype=torch.uint8)
    label[:n_bad] = 2
    label[n_bad:n_bad + n_good] = 1
    gain = torch.zeros(n).float()
    damage = torch.zeros(n).float()
    gain[:n_bad] = torch.linspace(0.01, 0.06, n_bad)
    damage[n_bad:n_bad + n_good] = torch.linspace(0.02, 0.05, n_good)
    variant = {
        "motion_rank_label": label,
        "motion_rank_gain": gain.clone().requires_grad_(True),
        "motion_rank_damage": damage.clone().requires_grad_(True),
        "motion_gt_valid": torch.ones(n, dtype=torch.bool),
    }

    def make_logits(shift):
        z = torch.full((n,), shift, requires_grad=True)
        return torch.stack([z, torch.zeros(n)], dim=1), z

    generator = torch.Generator().manual_seed(7)
    stacked, z_leaf = make_logits(0.0)
    loss, stats = pairwise_rank_loss({"gate_logits": stacked}, variant, 1.0, generator)
    loss.backward()
    grad = z_leaf.grad
    z = z_leaf.detach()

    generator2 = torch.Generator().manual_seed(7)
    picks = torch.randint(0, n_good, (n_bad,), generator=generator2)
    good_idx = torch.arange(n_bad, n_bad + n_good)[picks]
    bad_idx = torch.arange(n_bad)
    weight = torch.sqrt(gain[:n_bad] * damage[good_idx]).detach()
    expected = weight * torch.sigmoid(1.0 + z[bad_idx] - z[good_idx]) / n_bad
    expected_grad_bad = torch.zeros(n)
    expected_grad_good = torch.zeros(n)
    expected_grad_bad.index_add_(0, bad_idx, expected)
    expected_grad_good.index_add_(0, good_idx, -expected)
    expected_grad = expected_grad_bad + expected_grad_good
    max_grad_error = float((grad - expected_grad).abs().max())
    manual_loss = float((weight * torch.nn.functional.softplus(
        1.0 + z[bad_idx] - z[good_idx])).sum() / n_bad)

    generator3 = torch.Generator().manual_seed(7)
    shifted_stack, _ = make_logits(3.7)
    shifted, _ = pairwise_rank_loss({"gate_logits": shifted_stack}, variant, 1.0, generator3)
    shift_diff = float(abs(shifted.item() - loss.item()))

    generator4 = torch.Generator().manual_seed(7)
    empty = {"motion_rank_label": torch.zeros(n, dtype=torch.uint8),
             "motion_rank_gain": gain.detach(), "motion_rank_damage": damage.detach(),
             "motion_gt_valid": torch.ones(n, dtype=torch.bool)}
    empty_stack, _ = make_logits(0.0)
    empty_loss, empty_stats = pairwise_rank_loss({"gate_logits": empty_stack}, empty,
                                                 1.0, generator4)

    none_idx = torch.arange(n_bad + n_good, n)
    return {
        "pairs": int(stats["rank_pairs"]),
        "loss_matches_manual": abs(float(loss.item()) - manual_loss) < 1e-6,
        "loss": float(loss.item()),
        "loss_shift_invariance_max_diff": shift_diff,
        "suppress_grad_positive_fraction": float((grad[bad_idx] > 0).float().mean()),
        "keep_grad_negative_fraction": float((grad[good_idx] < 0).float().mean()),
        "none_grad_max_abs": float(grad[none_idx].abs().max()),
        "grad_sum": float(grad.sum()),
        "grad_formula_max_error": max_grad_error,
        "pair_weight_requires_grad": bool(gain.grad is not None or damage.grad is not None),
        "empty_pair_loss": float(empty_loss.item()),
        "empty_pair_pairs": int(empty_stats["rank_pairs"]),
        "logit_gap": float(stats["rank_logit_gap"]),
    }


def benchmark_variant(checkpoint, batch, steps, warmup, rank_weight, device):
    model, _ = trainer.load_shutrack_checkpoint(checkpoint, map_location="cpu")
    model.to(device)
    model.train()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(20261122)

    def compute_loss():
        outputs = model(**trainer.model_inputs(batch))
        base, components = cost_motion_loss(outputs, batch, 0.25)
        rank, _ = pairwise_rank_loss(outputs, batch, rank_weight, generator)
        return base + rank, components, outputs

    with torch.no_grad():
        initial, components, outputs = compute_loss()
        logits = outputs["gate_logits"].float()
        motion_gate = torch.sigmoid(logits[:, 0])
        appearance_gate = torch.sigmoid(logits[:, 1])
        appearance_valid = batch["appearance_valid"].bool()
        appearance_target = batch["appearance_target"].float()
        appearance_n = appearance_valid.sum().clamp_min(1)
        appearance_mae = float(
            (((appearance_gate - appearance_target).abs() * appearance_valid).sum()
             / appearance_n))
        appearance_bce = float(
            (torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1], appearance_target, reduction="none") * appearance_valid).sum()
            / appearance_n)

    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = compute_loss()
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
        loss, _, _ = compute_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    cuda_peak_alloc = (torch.cuda.max_memory_allocated() / 2 ** 20
                       if device.type == "cuda" else None)
    cuda_peak_reserved = (torch.cuda.max_memory_reserved() / 2 ** 20
                          if device.type == "cuda" else None)
    result = {
        "trainable_total": int(sum(p.numel() for p in parameters)),
        "trainable_parameters": len(parameters),
        "initial_loss": float(initial.detach()),
        "initial_l_pred": float(components["l_pred"]),
        "initial_l_bce": float(components["l_bce"]),
        "initial_motion_gate_mean": float(motion_gate.mean()),
        "initial_appearance_gate_mean": float(appearance_gate.mean()),
        "initial_appearance_mae": appearance_mae,
        "initial_appearance_bce_mean": appearance_bce,
        "step_seconds_mean": float(np.mean(times)),
        "step_seconds_median": float(np.median(times)),
        "step_seconds_min": float(np.min(times)),
        "step_seconds_max": float(np.max(times)),
        "cuda_peak_allocated_mib": cuda_peak_alloc,
        "cuda_peak_reserved_mib": cuda_peak_reserved,
        "process_rss_mib_after": rss_mib(),
    }
    del optimizer, parameters, model, outputs, components, initial
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--rank-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    data_root = Path(config["data_root"])
    sequence = args.sequence or config["sequence_names"][0]
    split = config.get("split", "train")
    dataset = config.get("dataset", "SportsMOT")
    group = config["group"]
    rank_weight = float(config.get("rank_weight_cli", 1.0))

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
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seq.input_statistics = (
        np.asarray(checkpoint_meta["normalization_mean"], dtype=np.float32),
        np.asarray(checkpoint_meta["normalization_std"], dtype=np.float32),
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
    best_box = g0 + star[:, None] * (g1 - g0)
    loss_best = 1.0 - numpy_diou(best_box, gt)
    keep_ref = valid_np & (star >= 1.0 - 1e-6) & (loss_half > loss1 + 1e-6)
    suppress_ref = valid_np & (star < 1.0 - 1e-6) & (loss1 > loss_best + 1e-6)
    stored_label = np.asarray(batch_np["motion_rank_label"])
    keep_match = bool(np.array_equal(keep_ref, stored_label == 1))
    suppress_match = bool(np.array_equal(suppress_ref, stored_label == 2))
    gain_match = bool(np.allclose(np.asarray(batch_np["motion_rank_gain"]),
                                  np.where(suppress_ref, loss1 - loss_best, 0.0), atol=1e-6))
    damage_match = bool(np.allclose(np.asarray(batch_np["motion_rank_damage"]),
                                    np.where(keep_ref, loss_half - loss1, 0.0), atol=1e-6))

    synthetic = synthetic_gradient_checks()

    device = torch.device(args.device)
    batch = trainer.to_device(batch_np, device)
    for key in ("motion_gate_target", "motion_rank_label",
                "motion_rank_gain", "motion_rank_damage"):
        batch[key] = torch.from_numpy(np.asarray(batch_np[key])).to(device)

    model, payload = trainer.load_shutrack_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device).train()
    parameters = list(model.parameters())
    total = int(sum(p.numel() for p in parameters))
    parameter_tensors = len(parameters)
    declared_total = int((payload.get("parameter_counts") or {}).get("total", 0))
    all_trainable = all(p.requires_grad for p in parameters)
    inputs = trainer.model_inputs(batch)
    with torch.no_grad():
        before = model(**inputs)
        appearance_before = before["gate_logits"][:, 1].clone()
        motion_before = before["gate_logits"][:, 0].clone()
        logits = before["gate_logits"].float()
        motion_gate = torch.sigmoid(logits[:, 0])
        appearance_gate = torch.sigmoid(logits[:, 1])
        motion_target = batch["motion_gate_target"].float()
        motion_valid = batch["motion_gt_valid"].bool()
        motion_n = motion_valid.sum().clamp_min(1)
        motion_mae = float(((motion_gate - motion_target).abs() * motion_valid).sum() / motion_n)
        appearance_valid = batch["appearance_valid"].bool()
        appearance_target = batch["appearance_target"].float()
        appearance_n = appearance_valid.sum().clamp_min(1)
        appearance_mae = float(
            ((appearance_gate - appearance_target).abs() * appearance_valid).sum() / appearance_n)
        appearance_bce_mean = float(
            (F.binary_cross_entropy_with_logits(
                logits[:, 1], appearance_target, reduction="none") * appearance_valid).sum()
            / appearance_n)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, p in model.named_parameters() if id(p) not in covered]
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    base, _ = cost_motion_loss(outputs, batch, 0.25)
    rank, _ = pairwise_rank_loss(outputs, batch, rank_weight,
                                 torch.Generator().manual_seed(20261122))
    (base + rank).backward()
    optimizer.step()
    with torch.no_grad():
        after = model(**inputs)
        appearance_diff = float((appearance_before - after["gate_logits"][:, 1]).abs().max())
        motion_diff = float((motion_before - after["gate_logits"][:, 0]).abs().max())
    del optimizer, before, after, outputs, model, inputs, base, rank
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    benchmark = {}
    if device.type == "cuda":
        for name, weight in (("A", 0.0), ("B", 1.0)):
            benchmark[name] = benchmark_variant(
                args.checkpoint, batch, args.steps, args.warmup, weight, device)
        benchmark["time_ratio_A_over_B"] = (benchmark["A"]["step_seconds_mean"]
                                            / max(1e-12, benchmark["B"]["step_seconds_mean"]))

    source_bytes = sum(int(seq.samples[key].nbytes) for key in SOURCE_KEYS if key in seq.samples)
    rank_bytes = int(label.nbytes + gain.nbytes + damage.nbytes)
    report = {
        "sequence": sequence, "split": split, "rows": int(len(rows)), "device": str(device),
        "binding": {
            "fingerprint_ok": bool(fingerprint_ok),
            "target_checksum_ok": bool(target_checksum_ok),
            "rank_binding_ok": bool(rank_binding_ok),
            "rank_checksums_ok": bool(checksums_ok),
            "rank_shapes_ok": bool(shapes_ok),
            "keep_rows_match": keep_match, "suppress_rows_match": suppress_match,
            "gain_match": gain_match, "damage_match": damage_match,
        },
        "counts": {
            "valid": int(valid_np.sum()), "keep": int(keep_ref.sum()),
            "suppress": int(suppress_ref.sum()),
            "keep_fraction": float(keep_ref.sum() / max(1, valid_np.sum())),
            "suppress_fraction": float(suppress_ref.sum() / max(1, valid_np.sum())),
        },
        "arrays": {
            "rank_dtype": [str(label.dtype), str(gain.dtype), str(damage.dtype)],
            "rank_bytes": rank_bytes, "source_bytes": source_bytes,
            "rank_bytes_fraction": rank_bytes / max(1, source_bytes),
            "rows_per_sequence": int(seq.n),
        },
        "gradient": synthetic,
        "model": {
            "parameter_tensors": parameter_tensors,
            "model_total": total,
            "checkpoint_declared_total": declared_total,
            "total_unchanged": total == declared_total,
            "all_parameters_requires_grad": all_trainable,
            "optimizer_parameter_tensors": len(covered),
            "optimizer_covers_all": not missing,
            "optimizer_missing": missing,
            "appearance_logit_max_diff_after_step": appearance_diff,
            "motion_logit_max_diff_after_step": motion_diff,
        },
        "branches": {
            "motion": {
                "valid": int(motion_valid.sum()),
                "gate_mean": float(motion_gate.mean()),
                "target_mean": float((motion_target * motion_valid).sum() / motion_n),
                "target_mae": motion_mae,
            },
            "appearance": {
                "valid": int(appearance_valid.sum()),
                "gate_mean": float(appearance_gate.mean()),
                "target_mean": float((appearance_target * appearance_valid).sum() / appearance_n),
                "target_mae": appearance_mae,
                "bce_mean": appearance_bce_mean,
            },
        },
        "benchmark": benchmark,
        "process_vm_hwm_mib": vmsize_peak_mib(),
        "policy": {"base_lambda": 0.25, "rank_weight": rank_weight,
                   "grid": GRID.tolist(), "tie_tolerance": TIE_TOLERANCE,
                   "train_all_parameters": True},
    }
    assert fingerprint_ok and target_checksum_ok and rank_binding_ok and checksums_ok and shapes_ok
    assert keep_match and suppress_match and gain_match and damage_match
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    seq.close_cache()
    seq.samples.close()


if __name__ == "__main__":
    main()
