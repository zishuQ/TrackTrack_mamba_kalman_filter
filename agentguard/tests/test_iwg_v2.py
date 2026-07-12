from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agentguard.models.iwg import IWG
from agentguard.rollout_labels import (
    compute_label_confidence,
    compute_safe_soft_target,
)


def _inputs(batch_size: int = 3, reid_dim: int = 16):
    generator = torch.Generator().manual_seed(7)
    return {
        "track_feats": torch.randn(batch_size, 6, reid_dim, generator=generator),
        "det_feats": torch.randn(batch_size, 6, reid_dim, generator=generator),
        "scalar_feats": torch.randn(batch_size, 6, 63, generator=generator),
        "mask": torch.tensor([[True, True, False, False, False, False]] * batch_size),
    }


def test_iwg_v2_outputs_composable_embedding_and_legacy_aliases():
    model = IWG(reid_dim=16).eval()
    with torch.no_grad():
        outputs = model(**_inputs())

    assert outputs["event_embedding"].shape == (3, 128)
    assert outputs["policy_logits"].shape == (3, 5)
    assert outputs["policy_probs"].shape == (3, 5)
    assert outputs["base_gate"].shape == (3, 2)
    assert outputs["cue_logits"].shape == (3, 3)
    assert outputs["cue"].shape == (3, 3)
    assert outputs["event_logits"].shape == (3, 10)
    assert outputs["gate_residual"].shape == (3, 2)
    torch.testing.assert_close(outputs["base_gate"], outputs["gate"])
    assert torch.isfinite(outputs["event_embedding"]).all()
    assert torch.isfinite(outputs["base_gate"]).all()


def test_iwg_v2_heads_have_nonlinearity_and_normalization():
    model = IWG(reid_dim=16)
    for head in (
        model.policy_head,
        model.gate_residual_head,
        model.event_head,
        model.cue_head,
    ):
        assert any(isinstance(module, nn.GELU) for module in head)
        assert any(isinstance(module, nn.LayerNorm) for module in head)


def test_all_iwg_v2_heads_and_shared_encoder_receive_finite_gradients():
    model = IWG(reid_dim=16).train()
    outputs = model(**_inputs())
    loss = (
        outputs["policy_logits"].square().mean()
        + outputs["gate_residual"].square().mean()
        + outputs["event_logits"].square().mean()
        + outputs["cue_logits"].square().mean()
        + outputs["event_embedding"].square().mean()
    )
    loss.backward()

    groups = {
        "encoder": model.encoder,
        "transformer": model.transformer,
        "policy_head": model.policy_head,
        "gate_residual_head": model.gate_residual_head,
        "event_head": model.event_head,
        "cue_head": model.cue_head,
    }
    for name, module in groups.items():
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients, name
        assert all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), name


@pytest.mark.parametrize(
    "benefit,tau,coverage,expected",
    [
        (0.0, 0.01, 1.0, 1.0),
        (0.1, 0.01, 0.0, 1.0),
    ],
)
def test_safe_soft_target_falls_back_to_full_update(
    benefit,
    tau,
    coverage,
    expected,
):
    confidence = compute_label_confidence(benefit, tau, coverage)
    assert compute_safe_soft_target(benefit, tau, confidence) == pytest.approx(expected)


def test_safe_soft_target_tracks_strong_high_coverage_evidence():
    negative_confidence = compute_label_confidence(-0.1, 0.01, 1.0)
    positive_confidence = compute_label_confidence(0.1, 0.01, 1.0)
    assert negative_confidence == pytest.approx(1.0)
    assert positive_confidence == pytest.approx(1.0)
    assert compute_safe_soft_target(-0.1, 0.01, negative_confidence) < 0.001
    assert compute_safe_soft_target(0.1, 0.01, positive_confidence) > 0.999


def test_compact_dataset_prefers_safe_targets_and_exposes_confidence():
    from agentguard.v0_pipeline import _CompactDatasetBase

    targets = _CompactDatasetBase._targets(
        {
            "candidate_type": "A",
            "motion_soft_target": 0.1,
            "appearance_soft_target": 0.2,
            "motion_safe_target": 0.8,
            "appearance_safe_target": 0.9,
            "motion_label_confidence": 0.4,
            "appearance_label_confidence": 0.6,
            "policy_soft_target": [1.0, 0.0, 0.0, 0.0, 0.0],
            "policy_safe_soft_target": [0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    assert targets["motion_target"].item() == pytest.approx(0.8)
    assert targets["appearance_target"].item() == pytest.approx(0.9)
    assert targets["motion_label_confidence"].item() == pytest.approx(0.4)
    assert targets["appearance_label_confidence"].item() == pytest.approx(0.6)
    np.testing.assert_allclose(
        targets["policy_soft_target"].numpy(),
        [0.0, 1.0, 0.0, 0.0, 0.0],
    )


def test_iwg_training_loss_supervises_cue_head():
    from agentguard.training.train_iwg import _compute_iwg_loss

    model = IWG(reid_dim=16).train()
    outputs = model(**_inputs())
    targets = {
        "motion_target": torch.tensor([0.2, 0.8, 1.0]),
        "appearance_target": torch.tensor([0.9, 0.1, 1.0]),
        "motion_label_confidence": torch.tensor([0.8, 0.7, 0.0]),
        "appearance_label_confidence": torch.tensor([0.6, 0.9, 0.0]),
        "policy_soft_target": torch.full((3, 5), 0.2),
        "valid_motion": torch.ones(3, dtype=torch.bool),
        "valid_appearance": torch.ones(3, dtype=torch.bool),
        "sample_weight": torch.ones(3),
    }
    loss, components = _compute_iwg_loss(outputs, targets)
    loss.backward()

    assert "cue_loss" in components
    assert torch.isfinite(components["cue_loss"])
    gradients = [parameter.grad for parameter in model.cue_head.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
