#!/usr/bin/env python3
"""Pre-training validation for the D-hinge column loss.

Checks on real batches from the shared SportsMOT train data:
  1. Won events (correct-track cost <= rival cost): hinge penalty and gradient
     are exactly zero.
  2. Active events (cost > rival): finite, non-zero appearance-gate gradient.
  3. The column term has exactly zero direct gradient on the motion gate logit.
  4. Motion / rank / fallback losses are unchanged by the hinge option.
  5. The original softplus path is numerically identical to a reference
     implementation of the pre-refactor formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE = Path(__file__).resolve().parents[1]
for path in (
    str(WORKTREE / "tools"),
    str(WORKTREE),
    str(WORKTREE / "agentguard/src"),
    str(WORKTREE / "3. Tracker"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_supervision_groups import (  # noqa: E402
    SequenceData,
    batch_tensors,
    derive_targets,
    model_inputs,
    to_device,
)
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss,
    appearance_rank_loss,
    gated_appearance_template,
    motion_box_diou_loss,
)

ALPHA = 0.95
TAU = 0.1
RANK_WEIGHT = 18.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--taus", required=True, help="JSON dict with the training taus")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    taus = json.loads(args.taus)
    data_root = Path(args.data_root)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = ShutrackV2Model(ShutrackV2Config(reid_dim=2048))
    model.eval()

    per_sequence = {}
    all_ok = True
    for sequence in args.sequences:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        split = summary.get("split")
        if split not in ("train", "val"):
            split = "train"
        data = SequenceData(sequence, split, data_root, "SportsMOT")
        data.open_cache()
        try:
            targets = derive_targets(data.samples, taus)
            rows = np.arange(min(data.n, args.rows), dtype=np.int64)
            batch = to_device(
                batch_tensors(data, rows, targets, "appearance_row_column"),
                torch.device("cpu"),
            )
            outputs = model(**model_inputs(batch))
            logits = outputs["gate_logits"].clone().requires_grad_(True)
            out = {"gate_logits": logits}

            soft_components: dict[str, object] = {}
            soft = appearance_column_loss(
                out, batch, alpha=ALPHA, tau=TAU, hinge=False, components=soft_components
            )
            hinge_components: dict[str, object] = {}
            hinge = appearance_column_loss(
                out, batch, alpha=ALPHA, tau=TAU, hinge=True, components=hinge_components
            )

            gate = torch.sigmoid(logits.float()[:, 1])
            updated = gated_appearance_template(
                batch["current_track"], batch["current_detection"],
                batch["column_score"], gate, alpha=ALPHA,
            )
            distance = (1 - (updated * batch["column_positive"]).sum(-1)).clamp(0, 1)
            cost = (batch["column_offset"] + 0.5 * distance).clamp(0, 1)
            reference = (
                F.softplus((cost - batch["column_rival"]) / TAU)
                * batch["column_valid"].float()
            ).sum() / max(1, gate.numel())

            grad_hinge = torch.autograd.grad(hinge, logits, retain_graph=True)[0]
            grad_soft = torch.autograd.grad(soft, logits, retain_graph=True)[0]
            margin = hinge_components["margin"]
            valid = hinge_components["valid_mask"]
            won = valid & (margin <= 0)
            active = valid & (margin > 0)
            winner_penalty = F.relu(margin)[won]
            appearance_grad = grad_hinge[:, 1]
            active_abs = appearance_grad.abs()[active]

            base_a = motion_box_diou_loss(out, batch)[0] + appearance_rank_loss(
                out, batch, alpha=ALPHA, tau=TAU, rank_weight=RANK_WEIGHT
            )[0]
            base_b = motion_box_diou_loss(out, batch)[0] + appearance_rank_loss(
                out, batch, alpha=ALPHA, tau=TAU, rank_weight=RANK_WEIGHT
            )[0]

            winner_batch = dict(batch)
            winner_batch["column_rival"] = batch["column_offset"] + 1.0
            winner_loss = appearance_column_loss(
                out, winner_batch, alpha=ALPHA, tau=TAU, hinge=True
            )
            winner_grad = torch.autograd.grad(
                winner_loss, logits, retain_graph=True
            )[0]

            result = {
                "rows": int(rows.size),
                "valid_events": int(valid.sum()),
                "won_events": int(won.sum()),
                "active_events": int(active.sum()),
                "won_penalty_max": float(winner_penalty.max()) if won.any() else 0.0,
                "won_appearance_grad_max": (
                    float(appearance_grad[won].abs().max()) if won.any() else 0.0
                ),
                "active_grad_finite": bool(torch.isfinite(appearance_grad[active]).all()),
                "active_grad_nonzero_fraction": (
                    float((active_abs > 0).float().mean()) if active.any() else None
                ),
                "active_grad_mean": float(active_abs.mean()) if active.any() else None,
                "column_motion_grad_max": float(grad_hinge[:, 0].abs().max()),
                "softplus_reference_equal": bool(torch.equal(reference, soft)),
                "hinge_loss": float(hinge),
                "softplus_loss": float(soft),
                "hinge_finite": bool(torch.isfinite(hinge)),
                "softplus_finite": bool(torch.isfinite(soft)),
                "base_losses_equal": bool(torch.equal(base_a, base_b)),
                "all_won_hinge_loss": float(winner_loss),
                "all_won_grad_max": float(winner_grad.abs().max()),
            }
            result["ok"] = (
                result["won_penalty_max"] == 0.0
                and result["won_appearance_grad_max"] == 0.0
                and result["active_grad_finite"]
                and (result["active_grad_nonzero_fraction"] or 0.0) > 0.0
                and result["column_motion_grad_max"] == 0.0
                and result["softplus_reference_equal"]
                and result["base_losses_equal"]
                and result["hinge_finite"]
                and result["all_won_hinge_loss"] == 0.0
                and result["all_won_grad_max"] == 0.0
            )
            per_sequence[sequence] = result
            all_ok &= result["ok"]
        finally:
            data.close_cache()

    report = {
        "data_root": str(data_root),
        "sequences": list(args.sequences),
        "taus": taus,
        "all_ok": bool(all_ok),
        "per_sequence": per_sequence,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
