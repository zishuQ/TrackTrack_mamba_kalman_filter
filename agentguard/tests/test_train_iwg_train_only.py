from __future__ import annotations

import json

import torch

from agentguard.models.iwg import IWG
from agentguard.training.train_iwg import train_iwg


class _ExplodingValidationLoader:
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        raise AssertionError("train-only mode must not iterate validation data")


def _batch() -> dict:
    generator = torch.Generator().manual_seed(41)
    batch_size, length, reid_dim = 2, 6, 4
    motion = torch.rand(batch_size, generator=generator)
    appearance = torch.rand(batch_size, generator=generator)
    motion_confidence = torch.rand(batch_size, generator=generator)
    appearance_confidence = torch.rand(batch_size, generator=generator)
    policy = torch.rand(batch_size, 5, generator=generator)
    policy /= policy.sum(dim=-1, keepdim=True)
    valid = torch.ones(batch_size, dtype=torch.bool)
    return {
        "track_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "det_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "scalar_feats": torch.randn(batch_size, length, 63, generator=generator),
        "mask": torch.zeros(batch_size, length, dtype=torch.bool),
        "targets": {
            "motion_target": motion,
            "appearance_target": appearance,
            "policy_soft_target": policy,
            "cue_target": torch.stack(
                [
                    motion_confidence,
                    appearance_confidence,
                    torch.maximum(motion_confidence, appearance_confidence),
                ],
                dim=-1,
            ),
            "risk_target": torch.stack(
                [
                    1.0 - motion,
                    1.0 - appearance,
                    1.0 - torch.minimum(motion_confidence, appearance_confidence),
                    torch.abs(motion - appearance),
                ],
                dim=-1,
            ),
            "valid_motion": valid,
            "valid_appearance": valid,
            "sample_weight": torch.ones(batch_size),
        },
    }


def test_train_only_skips_validation_and_selects_fixed_last_epoch(tmp_path):
    model = IWG(reid_dim=4)
    summary = train_iwg(
        model,
        [_batch()],
        _ExplodingValidationLoader(),  # type: ignore[arg-type]
        {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "amp": False,
            "skip_validation": True,
            "full_val_every": 1,
            "seed": 42,
        },
        str(tmp_path),
        full_val_loader=_ExplodingValidationLoader(),  # type: ignore[arg-type]
    )

    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] is None
    assert summary["best_metric_source"] == "fixed_last_epoch"
    assert summary["checkpoints"]["best"] is None
    assert summary["checkpoints"]["selected"] == summary["checkpoints"]["last"]
    assert (tmp_path / "iwg_last.pt").is_file()
    assert not (tmp_path / "iwg_best.pt").exists()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["val"] == {} and "full_val" not in row for row in rows)
