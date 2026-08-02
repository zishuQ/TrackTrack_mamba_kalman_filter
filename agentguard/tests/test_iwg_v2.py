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
    assert outputs["risk_logits"].shape == (3, 4)
    assert outputs["risk"].shape == (3, 4)
    assert outputs["iwg_gate_residual"].shape == (3, 2)
    torch.testing.assert_close(outputs["base_gate"], outputs["gate"])
    assert torch.isfinite(outputs["event_embedding"]).all()
    assert torch.isfinite(outputs["base_gate"]).all()


def test_iwg_forward_sequence_shapes_and_last_position_parity():
    model = IWG(reid_dim=16).eval()
    inputs = _inputs()
    with torch.no_grad():
        single = model(**inputs)
        sequence = model.forward_sequence(**inputs)

    expected_dims = {
        "policy_logits": 5,
        "policy_probs": 5,
        "base_gate": 2,
        "gate": 2,
        "event_embedding": 128,
        "cue_logits": 3,
        "cue": 3,
        "risk_logits": 4,
        "risk": 4,
        "iwg_gate_residual": 2,
    }
    for key, dim in expected_dims.items():
        assert sequence[key].shape == (3, 6, dim)
        torch.testing.assert_close(single[key], sequence[key][:, -1], atol=1e-6, rtol=0.0)


def test_iwg_forward_sequence_is_causal():
    model = IWG(reid_dim=16).eval()
    inputs = _inputs(batch_size=1)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["track_feats"][:, 4:] += 100.0
    changed["det_feats"][:, 4:] -= 100.0
    changed["scalar_feats"][:, 4:] *= -50.0

    with torch.no_grad():
        original = model.forward_sequence(**inputs)
        perturbed = model.forward_sequence(**changed)
    for key in original:
        torch.testing.assert_close(
            original[key][:, :4], perturbed[key][:, :4], atol=1e-6, rtol=0.0
        )


def test_iwg_forward_selects_last_valid_position():
    model = IWG(reid_dim=16).eval()
    inputs = _inputs(batch_size=1)
    inputs["mask"][:, 4:] = True
    with torch.no_grad():
        single = model(**inputs)
        sequence = model.forward_sequence(**inputs)
    for key in single:
        torch.testing.assert_close(single[key], sequence[key][:, 3], atol=1e-6, rtol=0.0)


def test_iwg_single_forward_transforms_one_window_per_sample():
    model = IWG(reid_dim=16).eval()
    inputs = _inputs(batch_size=3)
    transformed_batch_sizes = []
    handle = model.transformer.register_forward_pre_hook(
        lambda _module, args: transformed_batch_sizes.append(args[0].shape[0])
    )
    try:
        with torch.no_grad():
            model(**inputs)
            model.forward_sequence(**inputs)
    finally:
        handle.remove()

    assert transformed_batch_sizes == [3, 18]


def test_iwg_v2_heads_have_nonlinearity_and_normalization():
    model = IWG(reid_dim=16)
    for head in (
        model.policy_head,
        model.iwg_gate_residual_head,
        model.risk_head,
        model.cue_head,
    ):
        assert any(isinstance(module, nn.GELU) for module in head)
        assert any(isinstance(module, nn.LayerNorm) for module in head)


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
