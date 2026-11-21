#!/usr/bin/env python3
"""Pre-training checks for the MOT20 D-256 appearance-projection experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    fit_input_statistics,
    load_global_taus,
    model_inputs,
    to_device,
)
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model  # noqa: E402
from shutrack.training.checkpoint import load_shutrack_checkpoint  # noqa: E402
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss,
    appearance_rank_loss,
    motion_box_diou_loss,
)

ALPHA = 0.95
TAU = 0.1
RANK_WEIGHT = 18.0
COLUMN_WEIGHT = 50.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--dataset", choices=("SportsMOT", "MOT17", "MOT20"), default="MOT20")
    parser.add_argument("--baseline-checkpoint", required=True, help="D-64 epoch_100.pt")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    torch.set_num_threads(4)
    data_root = Path(args.data_root)
    sequences = list(args.sequences)
    report: dict = {"data_root": str(data_root), "dataset": args.dataset,
                    "sequences": sequences, "proj_dim": args.proj_dim, "checks": {}}

    # Model structures and parameter counts.
    torch.manual_seed(42)
    model64 = ShutrackV2Model(ShutrackV2Config(reid_dim=2048, appearance_proj_dim=64))
    torch.manual_seed(42)
    model256 = ShutrackV2Model(ShutrackV2Config(reid_dim=2048, appearance_proj_dim=args.proj_dim))
    count64 = sum(p.numel() for p in model64.parameters())
    count256 = sum(p.numel() for p in model256.parameters())
    shapes = {
        "reid_proj": tuple(model256.appearance_encoder.reid_proj.weight.shape),
        "interaction_first": tuple(model256.appearance_encoder.interaction[0].weight.shape),
        "appearance_head_first": tuple(model256.rsc.appearance_head[1].weight.shape),
    }
    report["checks"]["parameter_count_64"] = int(count64)
    report["checks"]["parameter_count_256"] = int(count256)
    report["checks"]["shapes_256"] = shapes
    report["checks"]["shapes_ok"] = (
        shapes["reid_proj"] == (args.proj_dim, 2048)
        and shapes["interaction_first"] == (128, 4 * args.proj_dim)
        and shapes["appearance_head_first"] == (64, 128)
    )

    # Old 64-dim checkpoint still loads through the normal loader.
    baseline_model, baseline_payload = load_shutrack_checkpoint(
        args.baseline_checkpoint, map_location="cpu"
    )
    baseline_proj = int(baseline_payload.get("model_config", {}).get("appearance_proj_dim", 64))
    report["checks"]["baseline_proj_dim"] = baseline_proj
    report["checks"]["baseline_load_ok"] = baseline_proj == 64

    # Data, normalization and label thresholds must match D-64 exactly.
    taus = load_global_taus(sequences, data_root)
    report["checks"]["taus"] = taus
    baseline_cfg = baseline_payload.get("training_config") or {}
    report["checks"]["baseline_taus"] = baseline_cfg.get("taus")
    report["checks"]["taus_equal"] = taus == baseline_cfg.get("taus")

    datasets = {}
    for sequence in sequences:
        summary = json.loads((data_root / sequence / "summary.json").read_text())
        split = summary.get("split") if summary.get("split") in ("train", "val", "all") else "all"
        data = SequenceData(sequence, split, data_root, args.dataset)
        data.open_cache()
        datasets[sequence] = data
    mean, std = fit_input_statistics(datasets)
    base_mean = np.asarray(baseline_payload["normalization_mean"])
    base_std = np.asarray(baseline_payload["normalization_std"])
    report["checks"]["normalization_equal"] = bool(
        np.array_equal(mean, base_mean) and np.array_equal(std, base_std)
    )
    for data in datasets.values():
        data.input_statistics = (mean, std)

    # Real batch: forward/loss/backward with finite gradients.
    data = datasets[sequences[0]]
    targets = derive_targets(data.samples, taus)
    derive_targets_base = derive_targets(data.samples, baseline_cfg.get("taus"))
    report["checks"]["targets_equal"] = all(
        np.array_equal(np.asarray(targets[key]), np.asarray(derive_targets_base[key]))
        for key in targets
    )
    rows = np.arange(min(args.rows, data.n), dtype=np.int64)
    batch = to_device(
        batch_tensors(data, rows, targets, "appearance_row_column"),
        torch.device("cpu"),
    )
    model256.train()
    outputs = model256(**model_inputs(batch))
    motion_loss, _ = motion_box_diou_loss(outputs, batch)
    rank_loss, _ = appearance_rank_loss(
        outputs, batch, alpha=ALPHA, tau=TAU, rank_weight=RANK_WEIGHT
    )
    column_loss = appearance_column_loss(
        outputs, batch, alpha=ALPHA, tau=TAU, hinge=False
    )
    loss = motion_loss + rank_loss + COLUMN_WEIGHT * column_loss
    model256.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model256.parameters() if p.grad is not None]
    finite_grads = all(bool(torch.isfinite(g).all()) for g in grads)
    report["checks"].update(
        {
            "batch_rows": int(rows.size),
            "loss": float(loss.detach()),
            "loss_finite": bool(torch.isfinite(loss)),
            "grad_tensors": len(grads),
            "gradients_finite": finite_grads,
        }
    )
    for data in datasets.values():
        data.close_cache()

    report["all_ok"] = (
        report["checks"]["shapes_ok"]
        and report["checks"]["baseline_load_ok"]
        and report["checks"]["taus_equal"]
        and report["checks"]["normalization_equal"]
        and report["checks"]["targets_equal"]
        and report["checks"]["loss_finite"]
        and report["checks"]["gradients_finite"]
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
