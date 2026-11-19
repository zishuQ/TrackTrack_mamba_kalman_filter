#!/usr/bin/env python3
"""Real-batch validation for the appearance-rank and motion-speed groups.

Checks on real samples from three sequences:
  * the differentiable appearance template matches the online numpy formula
    (keep / full EMA write / gate interpolation) at g = 0, 0.5, 1;
  * the identity ranking loss produces finite gradients on the appearance
    gate only, and reports how many samples actually generate gradient;
  * KF centre-velocity interpolation linearity (recorded at generation) and
    the speed-L1 / box-DIoU gradient magnitudes on the motion gate, which set
    the starting speed coefficient;
  * c003 contested GT3/GT4 events around the long swap window.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
for path in (str(WORKTREE / "tools"), str(WORKTREE), str(WORKTREE / "agentguard/src"), str(WORKTREE / "3. Tracker")):
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
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    appearance_rank_loss,
    appearance_soft_loss,
    gated_appearance_template,
    motion_box_diou_loss,
    motion_speed_l1_loss,
)

NEW_ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119"
DATA = NEW_ROOT / "data"
SEQUENCES = (
    "v_9MHDmAMxO5I_c003",
    "v_00HRwkvvjtQ_c007",
    "v_00HRwkvvjtQ_c011",
)
RANK_LIMIT = 1024


def numpy_template(t0, det, score, gate, alpha=0.95):
    beta = alpha + (1.0 - alpha) * (1.0 - score)
    full = beta * t0 + (1.0 - beta) * det
    if gate == 1.0:
        feat = full / np.linalg.norm(full)
    elif gate == 0.0:
        feat = t0.copy()
    else:
        blended = (1.0 - gate) * t0 + gate * full
        feat = blended / (np.linalg.norm(blended) + 1e-12)
    return feat


def split_of(sequence: str) -> str:
    summary = json.loads((DATA / sequence / "summary.json").read_text())
    split = summary.get("split")
    if split in ("train", "val"):
        return split
    val_dir = MAIN / "outputs/agentguard/detection_cache/SportsMOT/val"
    return "val" if (val_dir / sequence).is_dir() else "train"


def parameter_grad_norm(model, inputs, loss_fn):
    model.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    loss, _ = loss_fn(outputs)
    loss.backward()
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm() ** 2)
    model.zero_grad(set_to_none=True)
    return math.sqrt(total), float(loss.detach())


def main() -> None:
    torch.set_num_threads(4)
    model, _ = load_shutrack_checkpoint(INIT_CKPT, map_location="cpu")
    model.eval()
    taus = load_global_taus(list(SEQUENCES), DATA)
    report = {"taus": taus, "sequences": {}, "totals": {}}
    totals = {
        "rank_rows": 0,
        "rank_valid_appearance_rows": 0,
        "rank_grad_nonzero": 0,
        "rank_grad_nonfinite": 0,
        "rank_grad_abs_sum": 0.0,
        "rank_appearance_leak_max": 0.0,
        "rank_loss_sum": 0.0,
        "soft_grad_abs_sum": 0.0,
        "rank_param_norm_sum": 0.0,
        "soft_param_norm_sum": 0.0,
        "box_param_norm_sum": 0.0,
        "speed_param_norm_sum": 0.0,
        "fallback_rows": 0,
        "speed_rows": 0,
        "speed_grad_nonzero": 0,
        "speed_grad_nonfinite": 0,
        "speed_grad_abs_sum": 0.0,
        "box_grad_abs_sum": 0.0,
        "box_grad_nonzero": 0,
        "speed_motion_leak_max": 0.0,
        "template_parity_max": 0.0,
        "c003_rank_rows": 0,
        "c003_margin_g1_improved": 0,
        "c003_margin_g0_mean": 0.0,
        "c003_margin_g05_mean": 0.0,
        "c003_margin_g1_mean": 0.0,
    }
    for sequence in SEQUENCES:
        split = split_of(sequence)
        data = SequenceData(sequence, split, DATA)
        data.open_cache()
        targets = derive_targets(data.samples, taus)
        samples = data.samples
        summary = json.loads((DATA / sequence / "summary.json").read_text())
        entry: dict = {"split": split}

        rank_rows = samples["rank_rows"][:RANK_LIMIT]
        if rank_rows.size:
            batch = to_device(
                batch_tensors(data, rank_rows, targets, "appearance_rank"),
                torch.device("cpu"),
            )
            outputs = model(**model_inputs(batch))
            logits = outputs["gate_logits"].clone().requires_grad_(True)
            rank_loss, rank_components = appearance_rank_loss(
                {"gate_logits": logits}, batch, alpha=0.95, tau=0.1
            )
            grad = torch.autograd.grad(rank_loss, logits)[0]
            finite = bool(torch.isfinite(grad).all())
            rank_mask = batch["rank_valid"] & batch["appearance_valid"]
            appearance_grad = grad[:, 1]
            motion_grad = grad[:, 0]
            nonzero = float(((appearance_grad.abs() > 0) & rank_mask).float().sum())
            mean_abs = float(
                (appearance_grad.abs() * rank_mask.float()).sum()
                / rank_mask.float().sum().clamp_min(1.0)
            )
            entry["rank"] = {
                "rows": int(rank_rows.size),
                "rank_valid_appearance": int(rank_mask.sum()),
                "loss": float(rank_loss),
                "rank_mean": float(rank_components["rank_mean"]),
                "fallback_mean": float(rank_components["fallback_mean"]),
                "d_positive_mean": float(rank_components["d_positive_mean"]),
                "d_competitor_mean": float(rank_components["d_competitor_mean"]),
                "grad_finite": finite,
                "grad_nonzero_fraction": nonzero / max(float(rank_mask.sum()), 1.0),
                "grad_abs_mean": mean_abs,
                "motion_channel_leak_max": float(motion_grad.abs().max()),
            }
            totals["rank_rows"] += int(rank_rows.size)
            totals["rank_valid_appearance_rows"] += int(rank_mask.sum())
            totals["rank_grad_nonzero"] += int(nonzero)
            totals["rank_grad_nonfinite"] += 0 if finite else int(rank_rows.size)
            totals["rank_grad_abs_sum"] += mean_abs * int(rank_mask.sum())
            totals["rank_appearance_leak_max"] = max(
                totals["rank_appearance_leak_max"], float(motion_grad.abs().max())
            )
            totals["rank_loss_sum"] += float(rank_components["rank_mean"]) * int(rank_mask.sum())
            totals["fallback_rows"] += int(
                (batch["appearance_valid"] & ~batch["rank_valid"]).sum()
            )
            soft_loss, _ = appearance_soft_loss({"gate_logits": logits}, batch)
            soft_grad = torch.autograd.grad(soft_loss, logits)[0][:, 1]
            soft_mean = float(
                (soft_grad.abs() * rank_mask.float()).sum()
                / rank_mask.float().sum().clamp_min(1.0)
            )
            entry["rank"]["soft_bce_grad_abs_mean"] = soft_mean
            entry["rank"]["suggested_rank_weight"] = soft_mean / max(mean_abs, 1e-12)
            totals["soft_grad_abs_sum"] += soft_mean * int(rank_mask.sum())

            inputs = model_inputs(batch)
            rank_norm, _ = parameter_grad_norm(
                model,
                inputs,
                lambda out: appearance_rank_loss(
                    out, batch, alpha=0.95, tau=0.1, rank_weight=1.0
                ),
            )
            soft_norm, _ = parameter_grad_norm(
                model, inputs, lambda out: appearance_soft_loss(out, batch)
            )
            entry["rank"]["rank_param_grad_norm"] = rank_norm
            entry["rank"]["soft_param_grad_norm"] = soft_norm
            entry["rank"]["suggested_rank_weight_param_norm"] = soft_norm / max(
                rank_norm, 1e-12
            )
            totals["rank_param_norm_sum"] += rank_norm * int(rank_rows.size)
            totals["soft_param_norm_sum"] += soft_norm * int(rank_rows.size)

            # Template parity at g = 0, 0.5, 1 on the first valid rows.
            valid_positions = torch.nonzero(rank_mask).flatten()[:8]
            parity = 0.0
            if valid_positions.numel():
                t0 = batch["rank_template0"][valid_positions].float()
                det = batch["rank_detection"][valid_positions].float()
                score = batch["rank_score"][valid_positions].float()
                for gate_value in (0.0, 0.5, 1.0):
                    gate_tensor = torch.full(
                        (t0.shape[0],), gate_value, dtype=torch.float32
                    )
                    torch_value = gated_appearance_template(
                        t0, det, score, gate_tensor, alpha=0.95
                    ).detach().numpy()
                    for index in range(t0.shape[0]):
                        numpy_value = numpy_template(
                            t0[index].numpy().astype(np.float64),
                            det[index].numpy().astype(np.float64),
                            float(score[index]),
                            gate_value,
                            0.95,
                        )
                        parity = max(
                            parity,
                            float(np.max(np.abs(torch_value[index] - numpy_value))),
                        )
            entry["rank"]["template_parity_max"] = parity
            totals["template_parity_max"] = max(totals["template_parity_max"], parity)

        # GT3/GT4 swap window inside c003.
        if sequence == "v_9MHDmAMxO5I_c003" and rank_rows.size:
            frame = samples["frame"][rank_rows]
            gt_id = samples["gt_id"][rank_rows]
            window = (frame >= 100) & (frame <= 140) & np.isin(gt_id, (3, 4))
            if window.any():
                rows = rank_rows[window]
                batch = to_device(
                    batch_tensors(data, rows, targets, "appearance_rank"),
                    torch.device("cpu"),
                )
                t0 = batch["rank_template0"].float()
                det = batch["rank_detection"].float()
                score = batch["rank_score"].float()
                positive = batch["rank_positive"].float()
                competitor = batch["rank_competitor"].float()

                def unit(x):
                    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                positive = unit(positive)
                competitor = unit(competitor)
                margins = {}
                for gate_value in (0.0, 0.5, 1.0):
                    gate_tensor = torch.full((t0.shape[0],), gate_value)
                    updated = gated_appearance_template(t0, det, score, gate_tensor)
                    margin = (updated * competitor).sum(-1) - (updated * positive).sum(-1)
                    margins[gate_value] = margin.detach().numpy()
                margins = {key: value[~np.isnan(value)] for key, value in margins.items()}
                improved = int(np.sum(margins[1.0] > margins[0.0]))
                entry["c003_swap_window"] = {
                    "rows": int(len(rows)),
                    "margin_g0_mean": float(np.mean(margins[0.0])),
                    "margin_g05_mean": float(np.mean(margins[0.5])),
                    "margin_g1_mean": float(np.mean(margins[1.0])),
                    "margin_improved_g1_vs_g0": improved,
                    "fraction_margin_positive_g1": float(np.mean(margins[1.0] > 0)),
                }
                totals["c003_rank_rows"] += int(len(rows))
                totals["c003_margin_g0_mean"] += float(np.mean(margins[0.0])) * len(rows)
                totals["c003_margin_g05_mean"] += float(np.mean(margins[0.5])) * len(rows)
                totals["c003_margin_g1_mean"] += float(np.mean(margins[1.0])) * len(rows)
                totals["c003_margin_g1_improved"] += improved

        # Speed batch: use all rows of the sequence (sampled cap 4096).
        speed_rows = np.arange(min(data.n, 4096), dtype=np.int64)
        batch_speed = to_device(
            batch_tensors(data, speed_rows, targets, "motion_diou_speed"),
            torch.device("cpu"),
        )
        outputs = model(**model_inputs(batch_speed))
        logits = outputs["gate_logits"].clone().requires_grad_(True)
        box_loss, box_components = motion_box_diou_loss(
            {"gate_logits": logits}, batch_speed
        )
        speed_loss, speed_components = motion_speed_l1_loss(
            {"gate_logits": logits}, batch_speed, weight=1.0
        )
        appearance_loss, appearance_components = appearance_soft_loss(
            {"gate_logits": logits}, batch_speed
        )
        box_grad_full = torch.autograd.grad(box_loss, logits, retain_graph=True)[0]
        speed_grad_full = torch.autograd.grad(speed_loss, logits)[0]
        box_grad = box_grad_full[:, 0]
        speed_grad = speed_grad_full[:, 0]
        finite = bool(
            torch.isfinite(box_grad_full).all() and torch.isfinite(speed_grad_full).all()
        )
        speed_mask = batch_speed["speed_valid"]
        box_mask = batch_speed["motion_gt_valid"]
        speed_nonzero = float(((speed_grad.abs() > 0) & speed_mask).float().sum())
        speed_mean = float(
            (speed_grad.abs() * speed_mask.float()).sum()
            / speed_mask.float().sum().clamp_min(1.0)
        )
        box_mean = float(
            (box_grad.abs() * box_mask.float()).sum()
            / box_mask.float().sum().clamp_min(1.0)
        )
        entry["speed"] = {
            "rows": int(speed_rows.size),
            "speed_valid": int(speed_mask.sum()),
            "box_valid": int(box_mask.sum()),
            "box_loss": float(box_components["diou_loss"]),
            "speed_l1": float(speed_components["speed_l1"]),
            "speed_at_g0": float(speed_components["speed_at_g0"]),
            "speed_at_g1": float(speed_components["speed_at_g1"]),
            "speed_at_gate": float(speed_components["speed_at_gate"]),
            "appearance_loss": float(appearance_loss),
            "grad_finite": finite,
            "box_grad_abs_mean": box_mean,
            "speed_grad_abs_mean": speed_mean,
            "speed_to_box_grad_ratio": speed_mean / max(box_mean, 1e-12),
            "speed_grad_nonzero_fraction": speed_nonzero
            / max(float(speed_mask.sum()), 1.0),
            "suggested_speed_weight_25pct": 0.25 * box_mean / max(speed_mean, 1e-12),
            "appearance_channel_leak_max": float(
                torch.maximum(
                    box_grad_full[:, 1].abs().max(),
                    speed_grad_full[:, 1].abs().max(),
                )
            ),
            "kf_velocity_mid_error_max": summary.get("kf_velocity_mid_error_max"),
        }
        inputs = model_inputs(batch_speed)
        box_norm, _ = parameter_grad_norm(
            model, inputs, lambda out: motion_box_diou_loss(out, batch_speed)
        )
        speed_norm, _ = parameter_grad_norm(
            model,
            inputs,
            lambda out: motion_speed_l1_loss(out, batch_speed, weight=1.0),
        )
        entry["speed"]["box_param_grad_norm"] = box_norm
        entry["speed"]["speed_param_grad_norm"] = speed_norm
        entry["speed"]["suggested_speed_weight_param_norm"] = (
            0.25 * box_norm / max(speed_norm, 1e-12)
        )
        totals["box_param_norm_sum"] += box_norm * int(speed_rows.size)
        totals["speed_param_norm_sum"] += speed_norm * int(speed_rows.size)
        totals["speed_rows"] += int(speed_rows.size)
        totals["speed_grad_nonzero"] += int(speed_nonzero)
        totals["speed_grad_nonfinite"] += 0 if finite else int(speed_rows.size)
        totals["speed_grad_abs_sum"] += speed_mean * int(speed_mask.sum())
        totals["box_grad_abs_sum"] += box_mean * int(box_mask.sum())
        totals["box_grad_nonzero"] += int(((box_grad.abs() > 0) & box_mask).sum())
        report["sequences"][sequence] = entry
        data.close_cache()

    totals["rank_grad_abs_mean"] = totals["rank_grad_abs_sum"] / max(
        totals["rank_valid_appearance_rows"], 1
    )
    totals["rank_grad_nonzero_fraction"] = totals["rank_grad_nonzero"] / max(
        totals["rank_valid_appearance_rows"], 1
    )
    totals["rank_mean"] = totals["rank_loss_sum"] / max(
        totals["rank_valid_appearance_rows"], 1
    )
    totals["suggested_rank_weight"] = totals["soft_grad_abs_sum"] / max(
        totals["rank_grad_abs_sum"], 1e-12
    )
    totals["suggested_rank_weight_param_norm"] = totals["soft_param_norm_sum"] / max(
        totals["rank_param_norm_sum"], 1e-12
    )
    totals["speed_to_box_grad_ratio"] = (
        totals["speed_grad_abs_sum"] / max(totals["box_grad_abs_sum"], 1e-12)
    )
    totals["suggested_speed_weight_25pct"] = (
        0.25 * totals["box_grad_abs_sum"] / max(totals["speed_grad_abs_sum"], 1e-12)
    )
    totals["suggested_speed_weight_param_norm"] = (
        0.25 * totals["box_param_norm_sum"] / max(totals["speed_param_norm_sum"], 1e-12)
    )
    totals["speed_grad_nonzero_fraction"] = totals["speed_grad_nonzero"] / max(
        sum(report["sequences"][s]["speed"]["speed_valid"] for s in report["sequences"]),
        1,
    )
    if totals["c003_rank_rows"]:
        totals["c003_margin_g0_mean"] /= totals["c003_rank_rows"]
        totals["c003_margin_g05_mean"] /= totals["c003_rank_rows"]
        totals["c003_margin_g1_mean"] /= totals["c003_rank_rows"]
    totals["template_parity_equal"] = totals["template_parity_max"] <= 1e-5
    totals["rank_gradients_finite"] = totals["rank_grad_nonfinite"] == 0
    totals["speed_gradients_finite"] = totals["speed_grad_nonfinite"] == 0
    report["totals"] = totals
    out = NEW_ROOT / "rank_speed_validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("saved", out)


if __name__ == "__main__":
    main()
