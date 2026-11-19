"""Tests for the appearance match input and motion low-label weighting options."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from shutrack.data.legacy_adapter import build_online_evidence
from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.training.loss import gate_loss
from shutrack.training.train import _load_init_state_with_extension


def _base_config(*, appearance_match_input: bool = False) -> ShutrackV2Config:
    return ShutrackV2Config(
        reid_dim=8,
        d_model=32,
        n_heads=4,
        hce_layers=2,
        ff_dim=64,
        dropout=0.0,
        appearance_proj_dim=16,
        competition_enabled=True,
        appearance_match_input=appearance_match_input,
    )


def _inputs(batch_size: int = 4, seed: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "history_evidence": torch.randn(batch_size, 5, 13, generator=generator),
        "history_track": F.normalize(
            torch.randn(batch_size, 5, 8, generator=generator), dim=-1
        ),
        "history_detection": F.normalize(
            torch.randn(batch_size, 5, 8, generator=generator), dim=-1
        ),
        "history_mask": torch.zeros(batch_size, 5, dtype=torch.bool),
        "history_has_detection": torch.ones(batch_size, 5, dtype=torch.bool),
        "current_evidence": torch.randn(batch_size, 13, generator=generator),
        "current_track": F.normalize(
            torch.randn(batch_size, 8, generator=generator), dim=-1
        ),
        "current_detection": F.normalize(
            torch.randn(batch_size, 8, generator=generator), dim=-1
        ),
        "current_competition": torch.randn(batch_size, 3, generator=generator),
        "current_has_detection": torch.ones(batch_size, dtype=torch.bool),
        "history_reset": torch.zeros(batch_size, 5, dtype=torch.bool),
    }


def test_appearance_head_extension_is_identity_at_init():
    torch.manual_seed(11)
    base = ShutrackV2Model(_base_config(appearance_match_input=False))
    torch.manual_seed(11)
    extended = ShutrackV2Model(_base_config(appearance_match_input=True))
    _load_init_state_with_extension(extended, base)
    assert extended.rsc.appearance_head[1].weight.shape == (64, 33)
    assert torch.equal(
        extended.rsc.appearance_head[1].weight[:, 32],
        torch.zeros(64),
    )
    inputs = _inputs()
    base.eval()
    extended.eval()
    with torch.no_grad():
        base_logits = base(**inputs)["gate_logits"]
        zero_match = extended(**inputs, current_match_distance=torch.zeros(4))[
            "gate_logits"
        ]
        matched = extended(
            **inputs, current_match_distance=torch.tensor([0.5, -1.0, 2.0, 0.1])
        )["gate_logits"]
    torch.testing.assert_close(base_logits, zero_match, rtol=0.0, atol=1e-6)
    # The new input column is zero-initialized, so the option is an exact
    # identity at init; perturbing the column must change the output, proving
    # the cosine dimension is actually wired into the appearance head.
    with torch.no_grad():
        extended.rsc.appearance_head[1].weight[:, 32] = 0.1
    with torch.no_grad():
        perturbed = extended(
            **inputs, current_match_distance=torch.tensor([0.5, -1.0, 2.0, 0.1])
        )["gate_logits"]
    assert not torch.allclose(base_logits, perturbed, atol=1e-6)
    assert (
        sum(p.numel() for p in extended.parameters())
        - sum(p.numel() for p in base.parameters())
        == 64
    )


def test_motion_low_weight_one_matches_legacy_formula():
    torch.manual_seed(5)
    logits = torch.randn(64, 2)
    target = torch.rand(64, 2)
    valid = torch.rand(64, 2) > 0.2
    valid[0] = False
    weight = torch.rand(64) + 0.1
    outputs = {"gate_logits": logits}
    batch = {
        "safe_gate_target": target,
        "valid_channels": valid,
        "sample_weight": weight,
    }
    loss, _ = gate_loss(outputs, batch, motion_low_weight=1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    weights = valid.float() * weight.unsqueeze(-1)
    reference = ((bce + 0.1 * mse) * weights).sum() / weights.sum()
    torch.testing.assert_close(loss, reference, rtol=0.0, atol=1e-7)


def test_motion_low_weight_reweights_only_the_motion_channel():
    torch.manual_seed(7)
    logits = torch.randn(32, 2)
    target = torch.rand(32, 2)
    valid = torch.ones(32, 2, dtype=torch.bool)
    weight = torch.rand(32) + 0.5
    outputs = {"gate_logits": logits}
    batch = {
        "safe_gate_target": target,
        "valid_channels": valid,
        "sample_weight": weight,
    }
    _, legacy_components = gate_loss(outputs, batch, motion_low_weight=1.0)
    loss, components = gate_loss(outputs, batch, motion_low_weight=4.0)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mse = (torch.sigmoid(logits) - target).square()
    elementwise = bce + 0.1 * mse
    low = target[:, 0] < 0.5
    motion_weights = weight * torch.where(low, torch.full_like(weight, 4.0), torch.ones_like(weight))
    expected_motion = (elementwise[:, 0] * motion_weights).sum() / motion_weights.sum()
    expected_appearance = (elementwise[:, 1] * weight).sum() / weight.sum()
    expected = 0.5 * (expected_motion + expected_appearance)
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        legacy_components["appearance_gate_mean"],
        components["appearance_gate_mean"],
        rtol=0.0,
        atol=0.0,
    )
    share = (motion_weights * low.float()).sum() / motion_weights.sum()
    torch.testing.assert_close(
        components["motion_low_weight_share"], share, rtol=1e-6, atol=1e-7
    )
    assert components["motion_low_count"].item() == int(low.sum())
    # Empty channels fall back to the other channel, then to zero.
    invalid_motion = valid.clone()
    invalid_motion[:, 0] = False
    loss_motion_empty, _ = gate_loss(
        outputs,
        {
            "safe_gate_target": target,
            "valid_channels": invalid_motion,
            "sample_weight": weight,
        },
        motion_low_weight=4.0,
    )
    expected_appearance_only = (elementwise[:, 1] * weight).sum() / weight.sum()
    torch.testing.assert_close(
        loss_motion_empty, expected_appearance_only, rtol=1e-6, atol=1e-7
    )
    loss_empty, _ = gate_loss(
        outputs,
        {
            "safe_gate_target": target,
            "valid_channels": torch.zeros(32, 2, dtype=torch.bool),
            "sample_weight": weight,
        },
        motion_low_weight=4.0,
    )
    assert loss_empty.item() == 0.0


def test_packed_dataset_exposes_standardized_match_distance(tiny_packed_dir):
    dataset = PackedShutrackDataset(str(tiny_packed_dir))
    indices = list(range(len(dataset)))
    items = PackedShutrackDataset.collate_fn([dataset[index] for index in indices])
    batched = dataset.batch(indices)
    for index in indices:
        sequence_position, local_index = dataset._locate(index)
        sequence = dataset.sequences[sequence_position]
        packed = torch.load(
            tiny_packed_dir / sequence / "data.pt", weights_only=False
        )
        arrays = packed["arrays"]
        current = int(arrays["event_indices"][local_index][-1])
        expected = float(arrays["timeline_scalar_feats"][current][2])
        assert abs(float(items["current_match_distance"][index]) - expected) < 1e-6
        assert abs(float(batched["current_match_distance"][index]) - expected) < 1e-6


def test_online_evidence_match_distance_is_normalized_cosine():
    mean = np.zeros(63)
    std = np.ones(63)
    mean[2] = 0.25
    std[2] = 0.5
    raw_cosine = 0.6
    params = {
        "has_detection": True,
        "image_width": 1280,
        "image_height": 720,
        "frame_id": 10,
        "cosine_distance": raw_cosine,
        "iou_similarity": 0.4,
        "iou_distance": 0.6,
        "confidence_distance": 0.1,
        "angle_distance": 0.2,
        "raw_cost": 0.5,
        "final_cost": 0.45,
        "assignment_threshold": 0.7,
        "assignment_round": 1,
        "detection_source": 0,
        "detection": {"box": np.array([10.0, 20.0, 50.0, 90.0]), "score": 0.9, "source": 0},
        "pre_update_state": None,
    }
    evidence = build_online_evidence(
        scalar_params=params,
        track_feature=np.ones(8, dtype=np.float32),
        detection_feature=np.ones(8, dtype=np.float32),
        has_detection=True,
        competition_enabled=True,
        scalar_mean=mean,
        scalar_std=std,
        frame_id=10,
        track_id=3,
        segment_id=1,
        observation_count=6,
    )
    expected = (raw_cosine - mean[2]) / std[2]
    assert abs(evidence.match_distance - expected) < 1e-9
