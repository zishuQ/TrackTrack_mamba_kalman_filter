#!/usr/bin/env python3
"""Data and real-batch validation for the B/C/D supervision groups.

B = appearance_rank (motion DIoU + rank/fallback appearance)
C = appearance_rank_speed4 (B + 4-D KF velocity L1)
D = appearance_row_column (B + next-frame appearance contention loss)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (
    str(WORKTREE / "tools"),
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_supervision_groups import (  # noqa: E402
    INIT_CKPT,
    SequenceData,
    batch_tensors,
    derive_targets,
    load_global_taus,
    model_inputs,
    to_device,
)
from sportsmot_supervision_pilot import DEFAULT_CACHE, iou_xyxy  # noqa: E402
from agentguard.data.detection_cache import SequenceDetectionCache  # noqa: E402
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss,
    appearance_rank_loss,
    motion_box_diou_loss,
    motion_speed4_l1_loss,
)

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_bcd_20261119"
DATA = ROOT / "data"
SEQUENCES = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
)
SAMPLE_COLUMNS = 200


def split_of(sequence: str) -> str:
    summary = json.loads((DATA / sequence / "summary.json").read_text())
    split = summary.get("split")
    if split in ("train", "val"):
        return split
    val_dir = MAIN / "outputs/agentguard/detection_cache/SportsMOT/val"
    return "val" if (val_dir / sequence).is_dir() else "train"


def data_checks() -> dict:
    sequences = sorted(path.name for path in DATA.iterdir() if (path / "samples.npz").is_file())
    totals = {
        "sequences": len(sequences),
        "samples": 0,
        "rank_valid": 0,
        "speed_valid": 0,
        "column_valid": 0,
        "column_sequences": 0,
        "protocol_schema3": 0,
        "protocol_column": 0,
        "rank_arrays": 0,
        "actual_mismatch": 0,
        "column_key_checked": 0,
        "column_key_mismatch": 0,
        "column_offset_min": None,
        "column_offset_max": None,
        "column_rival_min": None,
        "column_rival_max": None,
        "reasons": [],
    }
    for sequence in sequences:
        summary = json.loads((DATA / sequence / "summary.json").read_text())
        protocol = summary.get("protocol") or {}
        if protocol.get("schema") == 3:
            totals["protocol_schema3"] += 1
        if protocol.get("column"):
            totals["protocol_column"] += 1
        if protocol.get("rank_arrays"):
            totals["rank_arrays"] += 1
        totals["actual_mismatch"] += int(
            (summary.get("column_counts") or {}).get("actual_mismatch", 0)
        )
        samples = np.load(DATA / sequence / "samples.npz")
        count = int(samples["frame"].size)
        totals["samples"] += count
        totals["rank_valid"] += int(samples["rank_valid"].sum())
        totals["speed_valid"] += int(samples["size_adj_valid"].sum())
        column = samples["column_det"] >= 0
        totals["column_valid"] += int(column.sum())
        if column.any():
            totals["column_sequences"] += 1
        for key in ("column_det", "column_offset", "column_rival"):
            if samples[key].shape[0] != count:
                totals["reasons"].append(f"{sequence}: {key} length mismatch")
        offsets = samples["column_offset"][column]
        rivals = samples["column_rival"][column]
        if offsets.size:
            low, high = float(offsets.min()), float(offsets.max())
            totals["column_offset_min"] = low if totals["column_offset_min"] is None else min(totals["column_offset_min"], low)
            totals["column_offset_max"] = high if totals["column_offset_max"] is None else max(totals["column_offset_max"], high)
            low, high = float(rivals.min()), float(rivals.max())
            totals["column_rival_min"] = low if totals["column_rival_min"] is None else min(totals["column_rival_min"], low)
            totals["column_rival_max"] = high if totals["column_rival_max"] is None else max(totals["column_rival_max"], high)
        # Spot-check that the recorded column detection is the correct-GT
        # detection at frame+1 for the same event.
        split = summary.get("split") or split_of(sequence)
        rows = np.nonzero(column)[0]
        if rows.size > SAMPLE_COLUMNS:
            rows = rows[:: max(1, rows.size // SAMPLE_COLUMNS)][:SAMPLE_COLUMNS]
        gt = None
        cache = SequenceDetectionCache(DEFAULT_CACHE / "SportsMOT" / split / sequence)
        try:
            for row in rows:
                frame = int(samples["frame"][row]) + 1
                gt_id = int(samples["gt_id"][row])
                index = int(samples["column_det"][row])
                det = cache.get_detection(index)
                box = np.asarray(det["box"], dtype=np.float64)
                if gt is None:
                    from sportsmot_supervision_pilot import load_gt

                    gt = load_gt(sequence, split)
                boxes = gt.get(frame, {})
                scored = sorted(
                    ((iou_xyxy(box, gt_box), gid) for gid, gt_box in boxes.items()),
                    reverse=True,
                )
                totals["column_key_checked"] += 1
                if not scored:
                    totals["column_key_mismatch"] += 1
                    continue
                best_iou, best_id = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                if not (best_iou >= 0.5 and best_iou - second >= 0.1 and int(best_id) == gt_id):
                    totals["column_key_mismatch"] += 1
        finally:
            cache.close()
    # Reject empty data: require at least the expected sequences and samples
    EXPECTED_SEQUENCES = 90
    EXPECTED_MIN_SAMPLES = 500000  # 593,673 for the current e200 replay
    totals["data_ok"] = (
        totals["reasons"] == []
        and len(sequences) >= EXPECTED_SEQUENCES
        and totals["samples"] >= EXPECTED_MIN_SAMPLES
        and totals["protocol_schema3"] == len(sequences)
        and totals["protocol_column"] == len(sequences)
        and totals["rank_arrays"] == len(sequences)
        and totals["actual_mismatch"] == 0
        and totals["column_key_mismatch"] == 0
    )
    return totals


def grad_stats(model, inputs, loss_fn):
    model.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    loss, _ = loss_fn(outputs)
    loss.backward()
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm() ** 2)
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), total ** 0.5


def batch_checks() -> dict:
    torch.set_num_threads(4)
    model, _ = load_shutrack_checkpoint(INIT_CKPT, map_location="cpu")
    model.eval()
    taus = load_global_taus(list(SEQUENCES), DATA)
    c_stats = {"rows": 0, "rank_valid": 0, "speed_valid": 0}
    d_stats = {"rows": 0, "column_valid": 0}
    report = {"C": {}, "D": {}}
    for sequence in SEQUENCES:
        split = split_of(sequence)
        data = SequenceData(sequence, split, DATA)
        data.open_cache()
        targets = derive_targets(data.samples, taus)
        rows = np.arange(min(data.n, 4096), dtype=np.int64)

        batch_c = to_device(
            batch_tensors(data, rows, targets, "appearance_rank_speed4"),
            torch.device("cpu"),
        )
        inputs_c = model_inputs(batch_c)
        logits_c = model(**inputs_c)["gate_logits"].clone().requires_grad_(True)
        box_loss, _ = motion_box_diou_loss({"gate_logits": logits_c}, batch_c)
        speed_loss, speed_components = motion_speed4_l1_loss(
            {"gate_logits": logits_c}, batch_c, center_weight=50.0, size_weight=14.0
        )
        rank_loss, rank_components = appearance_rank_loss(
            {"gate_logits": logits_c}, batch_c, alpha=0.95, tau=0.1, rank_weight=18.0
        )
        box_grad_full = torch.autograd.grad(box_loss, logits_c, retain_graph=True)[0]
        speed_grad_full = torch.autograd.grad(speed_loss, logits_c, retain_graph=True)[0]
        rank_grad_full = torch.autograd.grad(rank_loss, logits_c, retain_graph=True)[0]
        box_grad = box_grad_full[:, 0]
        speed_grad = speed_grad_full[:, 0]
        rank_grad = rank_grad_full[:, 1]
        speed_mask = batch_c["size_adj_valid"]
        rank_mask = batch_c["rank_valid"] & batch_c["appearance_valid"]
        c_stats["rows"] += int(rows.size)
        c_stats["rank_valid"] += int(rank_mask.sum())
        c_stats["speed_valid"] += int(speed_mask.sum())
        report["C"][sequence] = {
            "rows": int(rows.size),
            "rank_valid": int(rank_mask.sum()),
            "speed_valid": int(speed_mask.sum()),
            "box_loss": float(box_loss),
            "speed_center_l1": float(speed_components["speed_center_l1"]),
            "speed_size_l1": float(speed_components["speed_size_l1"]),
            "rank_mean": float(rank_components["rank_mean"]),
            "fallback_mean": float(rank_components["fallback_mean"]),
            "box_grad_motion": float(box_grad.abs().mean()),
            "speed_grad_motion_valid": float(
                (speed_grad.abs() * speed_mask.float()).sum()
                / speed_mask.float().sum().clamp_min(1.0)
            ),
            "rank_grad_appearance_valid": float(
                (rank_grad.abs() * rank_mask.float()).sum()
                / rank_mask.float().sum().clamp_min(1.0)
            ),
            "rank_grad_nonzero_fraction": float(
                ((rank_grad.abs() > 0) & rank_mask).float().sum()
                / rank_mask.float().sum().clamp_min(1.0)
            ),
            "motion_leak_from_rank": float(rank_grad_full[:, 0].abs().max()),
            "appearance_leak_from_speed": float(speed_grad_full[:, 1].abs().max()),
            "template_alignment_cosine_distance": float(
                (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        batch_c["current_track"][rank_mask],
                        batch_c["rank_template0"][rank_mask],
                        dim=-1,
                    )
                ).mean()
            )
            if rank_mask.any()
            else None,
        }

        batch_d = to_device(
            batch_tensors(data, rows, targets, "appearance_row_column"),
            torch.device("cpu"),
        )
        inputs_d = model_inputs(batch_d)
        logits_d = model(**inputs_d)["gate_logits"].clone().requires_grad_(True)
        column_loss = appearance_column_loss(
            {"gate_logits": logits_d}, batch_d, alpha=0.95, tau=0.1
        )
        rank_loss_d, _ = appearance_rank_loss(
            {"gate_logits": logits_d}, batch_d, alpha=0.95, tau=0.1, rank_weight=18.0
        )
        column_grad_full = torch.autograd.grad(column_loss, logits_d, retain_graph=True)[0]
        column_grad = column_grad_full[:, 1]
        valid_d = batch_d["column_valid"]
        d_stats["rows"] += int(rows.size)
        d_stats["column_valid"] += int(valid_d.sum())
        report["D"][sequence] = {
            "rows": int(rows.size),
            "column_valid": int(valid_d.sum()),
            "column_loss": float(column_loss),
            "rank_loss": float(rank_loss_d),
            "column_grad_appearance_valid": float(
                (column_grad.abs() * valid_d.float()).sum()
                / valid_d.float().sum().clamp_min(1.0)
            )
            if valid_d.any()
            else None,
            "column_grad_nonzero_fraction": float(
                ((column_grad.abs() > 0) & valid_d).float().sum()
                / valid_d.float().sum().clamp_min(1.0)
            )
            if valid_d.any()
            else None,
            "motion_leak_from_column": float(column_grad_full[:, 0].abs().max()),
        }
        data.close_cache()
    c_report = {
        "rows": c_stats["rows"],
        "rank_valid": c_stats["rank_valid"],
        "speed_valid": c_stats["speed_valid"],
        "grad_motion_box": np.mean([v["box_grad_motion"] for v in report["C"].values()]),
        "grad_motion_speed": np.mean(
            [v["speed_grad_motion_valid"] for v in report["C"].values()]
        ),
        "grad_appearance_rank": np.mean(
            [v["rank_grad_appearance_valid"] for v in report["C"].values()]
        ),
        "finite": all(
            bool(np.isfinite(v[key]))
            for v in report["C"].values()
            for key in ("box_loss", "speed_center_l1", "speed_size_l1", "rank_mean")
        ),
    }
    grad_values = [
        v["column_grad_appearance_valid"]
        for v in report["D"].values()
        if v["column_grad_appearance_valid"] is not None
    ]
    nz_values = [
        v["column_grad_nonzero_fraction"]
        for v in report["D"].values()
        if v["column_grad_nonzero_fraction"] is not None
    ]
    d_report = {
        "rows": d_stats["rows"],
        "column_valid": d_stats["column_valid"],
        "column_losses": [v["column_loss"] for v in report["D"].values()],
        "rank_losses": [v["rank_loss"] for v in report["D"].values()],
        "grad_appearance_column": float(np.mean(grad_values)) if grad_values else None,
        "grad_appearance_column_weight50": float(np.mean(grad_values) * 50.0)
        if grad_values
        else None,
        "column_grad_nonzero_fraction": float(np.mean(nz_values)) if nz_values else None,
    }
    return {"per_sequence": report, "C": c_report, "D": d_report}


def main() -> None:
    data = data_checks()
    batch = batch_checks()
    report = {"data": data, "batch": batch}
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "bcd_validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("saved", out)
    if not data["data_ok"]:
        raise SystemExit("BCD data validation failed: " + "; ".join(data["reasons"] or ["see report"]))


if __name__ == "__main__":
    main()
