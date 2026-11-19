#!/usr/bin/env python3
"""Real-batch validation of the motion GT-box DIoU loss on pilot samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentguard.data.detection_cache import SequenceDetectionCache
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.training.loss import motion_box_diou_loss

ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_pilot_20261118"
CKPT = MAIN / "outputs/agentguard/experiments/shutrack_a_submit_20261117/SportsMOT/checkpoints/last.pt"
CACHE = MAIN / "outputs/agentguard/detection_cache"
SEQUENCES = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")


def load_sequence(sequence: str):
    data = np.load(ROOT / sequence / "samples.npz", allow_pickle=False)
    keys = json.loads(str(data["keys_json"]))
    cache = SequenceDetectionCache(CACHE / "SportsMOT" / "val" / sequence)
    return data, keys, cache


def detection_features(cache: SequenceDetectionCache, indices: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((indices.shape[0], indices.shape[1], dim), dtype=np.float32)
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            index = int(indices[row, col])
            if index >= 0:
                out[row, col] = cache.get_detection(index)["feature"].astype(np.float32)
    return out


def build_batch(names, limit_per_sequence=400):
    arrays = {}
    meta = []
    for sequence in names:
        data, keys, cache = load_sequence(sequence)
        n = min(len(keys), limit_per_sequence)
        dim = int(data["history_track"].shape[-1])
        history_indices = data["history_detection_indices"][:n]
        current_indices = data["current_detection_index"][:n]
        history_det = detection_features(cache, history_indices, dim)
        current_det = np.zeros((n, dim), dtype=np.float32)
        for row in range(n):
            index = int(current_indices[row])
            if index >= 0:
                current_det[row] = cache.get_detection(index)["feature"].astype(np.float32)
        def add(name, value):
            arrays.setdefault(name, []).append(value)
        add("history_evidence", data["history_evidence"][:n])
        add("history_track", data["history_track"][:n])
        add("history_detection", history_det)
        add("history_mask", data["history_mask"][:n])
        add("history_has_detection", data["history_has_detection"][:n])
        add("history_reset", data["history_reset"][:n])
        add("current_evidence", data["current_evidence"][:n])
        add("current_competition", data["current_competition"][:n])
        add("current_track", data["current_track"][:n])
        add("current_detection", current_det)
        add("current_has_detection", data["current_has_detection"][:n])
        add("safe_gate_target", np.stack([data["motion_safe_target"][:n], data["appearance_safe_target"][:n]], axis=1))
        add("valid_channels", np.stack([data["valid_motion"][:n], data["valid_appearance"][:n]], axis=1))
        add("sample_weight", data["sample_weight"][:n])
        add("pred_box_g0", data["pred_box_g0"][:n])
        add("pred_box_g1", data["pred_box_g1"][:n])
        add("gt_box_t1", data["gt_box_t1"][:n])
        add("motion_gt_valid", data["motion_gt_valid"][:n])
        meta.extend(
            {
                "sequence": sequence,
                "frame": keys[row]["frame"],
                "track_id": keys[row]["track_id"],
            }
            for row in range(n)
        )
        cache.close()
    batch = {name: torch.from_numpy(np.concatenate(values, axis=0)) for name, values in arrays.items()}
    return batch, meta


def main() -> None:
    torch.set_num_threads(4)
    model, _ = load_runtime_model(CKPT, device="cpu")
    batch, meta = build_batch(SEQUENCES)
    inputs = {
        key: batch[key]
        for key in (
            "history_evidence",
            "history_track",
            "history_detection",
            "history_mask",
            "history_has_detection",
            "history_reset",
            "current_evidence",
            "current_competition",
            "current_track",
            "current_detection",
            "current_has_detection",
        )
    }
    logits = model(**inputs)["gate_logits"]
    loss, components = motion_box_diou_loss({"gate_logits": logits}, batch)
    grad = torch.autograd.grad(loss, logits)[0][:, 0]
    gate = torch.sigmoid(logits.detach())[:, 0]
    box0 = batch["pred_box_g0"]
    box1 = batch["pred_box_g1"]
    gt = batch["gt_box_t1"]
    valid = batch["motion_gt_valid"].bool()

    def diou_at(g):
        box = box0 + g.unsqueeze(-1) * (box1 - box0)
        return 1.0 - _diou(box, gt)

    e0 = diou_at(torch.zeros_like(gate))
    e1 = diou_at(torch.ones_like(gate))
    ec = diou_at(gate)
    mask = valid.float()
    report = {
        "events": int(len(gate)),
        "valid_events": int(valid.sum()),
        "loss": float(loss),
        "valid_ratio": float(components["valid_ratio"]),
        "gradient_nonzero_fraction": float((grad.abs() > 0).float().mean()),
        "gradient_abs_mean": float(grad.abs().mean()),
        "gradient_abs_max": float(grad.abs().max()),
        "diou_g0_mean": float(((e0 * mask).sum() / mask.sum()).item()),
        "diou_gate_mean": float(((ec * mask).sum() / mask.sum()).item()),
        "diou_g1_mean": float(((e1 * mask).sum() / mask.sum()).item()),
        "gate_mean": float(gate.mean()),
        "gate_vs_pilot_max_abs_diff": None,
    }
    # Preferred write gate per event (which endpoint has lower DIoU).
    pref0 = ((e0 <= e1) & valid).sum().item()
    pref1 = ((e1 < e0) & valid).sum().item()
    report["prefers_g0"] = int(pref0)
    report["prefers_g1"] = int(pref1)
    report["gate_matches_best_endpoint_fraction"] = float(
        (
            ((gate < 0.5) & (e0 <= e1)) | ((gate >= 0.5) & (e1 < e0))
        )[valid].float().mean()
    )

    # Key-window event table: which endpoint does GT DIoU prefer?
    windows = {
        "v_9MHDmAMxO5I_c003": (100, 140),
        "v_00HRwkvvjtQ_c007": (560, 600),
        "v_00HRwkvvjtQ_c011": (380, 410),
        "v_00HRwkvvjtQ_c011_2": (680, 700),
    }
    rows = []
    for index, item in enumerate(meta):
        for name, (start, end) in windows.items():
            sequence = name.split("_c0")[0] if name.count("_c0") > 1 else name
            base = "v_00HRwkvvjtQ_c011" if name.startswith("v_00HRwkvvjtQ_c011") else name
            if item["sequence"] == base and start <= item["frame"] <= end and bool(valid[index]):
                rows.append(
                    {
                        "sequence": item["sequence"],
                        "frame": item["frame"],
                        "track_id": item["track_id"],
                        "gate": float(gate[index]),
                        "diou_g0": float(e0[index]),
                        "diou_gate": float(ec[index]),
                        "diou_g1": float(e1[index]),
                        "preferred": "g0" if e0[index] <= e1[index] else "g1",
                    }
                )
                break
    report["key_window_rows"] = rows[:40]

    # One-step optimization on a temporary model.
    temp, _ = load_runtime_model(CKPT, device="cpu")
    for parameter in temp.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(temp.rsc.evidence_head.parameters(), lr=1e-3)
    before_logits = temp(**inputs)["gate_logits"].detach()
    loss_temp, _ = motion_box_diou_loss({"gate_logits": temp(**inputs)["gate_logits"]}, batch)
    optimizer.zero_grad()
    loss_temp.backward()
    optimizer.step()
    after_logits = temp(**inputs)["gate_logits"].detach()
    report["one_step"] = {
        "loss_before": float(loss_temp),
        "gate_logits_mean_abs_change": float((after_logits - before_logits).abs().mean()),
        "gate_logits_max_abs_change": float((after_logits - before_logits).abs().max()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    (ROOT / "motion_diou_batch_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _diou(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter = (
        torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)
    ).clamp_min(0) * (
        torch.minimum(ay2, by2) - torch.maximum(ay1, by1)
    ).clamp_min(0)
    area_a = (ax2 - ax1).clamp_min(0) * (ay2 - ay1).clamp_min(0)
    area_b = (bx2 - bx1).clamp_min(0) * (by2 - by1).clamp_min(0)
    iou = inter / (area_a + area_b - inter).clamp_min(1e-9)
    cx1, cy1 = torch.minimum(ax1, bx1), torch.minimum(ay1, by1)
    cx2, cy2 = torch.maximum(ax2, bx2), torch.maximum(ay2, by2)
    c2 = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2).clamp_min(1e-9)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    return iou - ((acx - bcx) ** 2 + (acy - bcy) ** 2) / c2


if __name__ == "__main__":
    main()
