"""Verify the new model against real paired batches without starting training.

Uses the existing optimized loader, D losses and runtime checkpoint loader.
No optimizer updates, data generation or tracking evaluation are performed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tools", ROOT / "agentguard/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from train_supervision_groups import (
    SequenceData, PAIRED_TENSOR_KEYS, batch_tensors, derive_targets,
    model_inputs, to_device,
)
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.training.checkpoint import build_checkpoint_payload, save_checkpoint_atomic
from shutrack.training.loss import (
    appearance_rank_loss, appearance_column_loss, motion_box_diou_loss,
)
from shutrack.training.paired_pipeline import prepared_batches


def verify(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline, reference = load_runtime_model(args.reference_checkpoint, device="cpu")
    assert baseline.config.appearance_proj_dim == 256
    assert baseline.config.architecture_variant == "legacy"
    baseline_parameters = baseline.parameter_report()["total"]
    del baseline
    model = ShutrackV2Model(ShutrackV2Config(
        reid_dim=reference["reid_dim"], appearance_proj_dim=256,
        architecture_variant="query_film_bilinear",
    )).to(device)
    model.train()
    data = SequenceData(args.sequence, "train", Path(args.data_root), dataset="MOT20")
    data.input_statistics = (
        np.asarray(reference["normalization_mean"], dtype=np.float32),
        np.asarray(reference["normalization_std"], dtype=np.float32),
    )
    taus = reference["training_config"]["taus"]
    targets = derive_targets(data.samples, taus)
    column_rows = np.flatnonzero(data.small["column_det"] >= 0)
    rank_rows = np.flatnonzero(data.rank_pos >= 0)
    fallback_rows = np.flatnonzero(data.rank_pos < 0)
    per_kind = max(1, args.batch_size // 3)
    rows = np.unique(np.concatenate([
        column_rows[:per_kind], rank_rows[:per_kind], fallback_rows[:per_kind],
    ]))
    data.open_cache()
    results = []
    try:
        with prepared_batches(
            rows, args.batch_size,
            lambda selected: batch_tensors(data, selected, targets,
                                           "appearance_row_column", compact_reid=True),
            PAIRED_TENSOR_KEYS, prefetch=1, pin_memory=device.type == "cuda",
        ) as batches:
            for _, host_batch in batches:
                batch = to_device(host_batch, device)
                outputs = model(**model_inputs(batch))
                box, _ = motion_box_diou_loss(outputs, batch)
                rank, _ = appearance_rank_loss(outputs, batch, rank_weight=18,
                                               tau=0.1, alpha=0.95)
                column = appearance_column_loss(outputs, batch, tau=0.1, alpha=0.95)
                loss = box + rank + 50 * column
                assert torch.isfinite(loss) and torch.isfinite(outputs["gates"]).all()
                loss.backward()
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None, f"Unused parameter: {name}"
                    assert torch.isfinite(parameter.grad).all(), f"Bad gradient: {name}"
                for index in range(2):
                    assert model.rsc.gate_heads[index].interaction.grad.abs().sum() > 0
                    assert model.rsc.modulations[index].weight.grad.abs().sum() > 0
                results.append({
                    "samples": len(batch["current_track"]),
                    "rank_valid": int(batch["rank_valid"].sum()),
                    "column_valid": int(batch["column_valid"].sum()),
                    "loss": float(loss.detach()), "box": float(box.detach()),
                    "rank": float(rank.detach()), "column": float(column.detach()),
                })
                model.zero_grad(set_to_none=True)
        assert results and sum(row["rank_valid"] for row in results) > 0
        assert sum(row["column_valid"] for row in results) > 0
        model.eval()
        with tempfile.TemporaryDirectory(prefix="shutrack-query-") as temporary:
            path = Path(temporary) / "new.pt"
            payload = build_checkpoint_payload(
                model=model, training_config={"architecture_variant": "query_film_bilinear"},
                dataset_descriptor=reference["dataset"], epoch=0, global_step=0,
                best_metric=float(loss.detach()),
                normalization_mean=reference["normalization_mean"],
                normalization_std=reference["normalization_std"], include_optimizer=False,
            )
            save_checkpoint_atomic(path, payload)
            restored, _ = load_runtime_model(path, device=args.device)
            with torch.no_grad():
                expected = model(**model_inputs(batch))["gate_logits"]
                actual = restored(**model_inputs(batch))["gate_logits"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return {
            "sequence": args.sequence, "device": str(device),
            "new_parameters": model.parameter_report(),
            "baseline_parameters": baseline_parameters,
            "batches": results, "runtime_roundtrip_max_error": float((actual - expected).abs().max()),
            "optimizer_steps": 0,
        }
    finally:
        data.close_cache()
        data.samples.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--sequence", default="MOT20-02")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 3:
        parser.error("--batch-size must be at least 3")
    print(json.dumps(verify(args), indent=2))


if __name__ == "__main__":
    main()
