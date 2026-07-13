from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agentguard.models.iwg import IWG
from agentguard.data.cache_schema import COMPACT_CACHE_SCHEMA_VERSION, FEATURE_SCHEMA_SHA256
from agentguard.data.label_schema import (
    ROLLOUT_LABEL_SCHEMA_SHA256,
    ROLLOUT_LABEL_SCHEMA_VERSION,
    make_cue_target,
    make_risk_targets,
)
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


def _production_targets():
    motion = torch.tensor([0.2, 0.8, 0.6])
    appearance = torch.tensor([0.9, 0.1, 0.4])
    motion_confidence = torch.tensor([0.8, 0.7, 0.3])
    appearance_confidence = torch.tensor([0.6, 0.9, 0.4])
    return {
        "motion_target": motion,
        "appearance_target": appearance,
        "motion_label_confidence": motion_confidence,
        "appearance_label_confidence": appearance_confidence,
        "cue_target": torch.stack(
            [motion_confidence, appearance_confidence, torch.maximum(motion_confidence, appearance_confidence)],
            dim=-1,
        ),
        "risk_target": torch.stack(
            [1.0 - motion, 1.0 - appearance, 1.0 - torch.minimum(motion_confidence, appearance_confidence), torch.abs(motion - appearance)],
            dim=-1,
        ),
        "policy_soft_target": torch.full((3, 5), 0.2),
        "valid_motion": torch.ones(3, dtype=torch.bool),
        "valid_appearance": torch.ones(3, dtype=torch.bool),
        "sample_weight": torch.ones(3),
    }


def test_all_iwg_v2_heads_and_shared_encoder_receive_real_finite_gradients():
    from agentguard.training.train_iwg import _compute_iwg_loss

    model = IWG(reid_dim=16).train()
    outputs = model(**_inputs())
    loss, components = _compute_iwg_loss(outputs, _production_targets())
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["cue_loss"])
    assert torch.isfinite(components["risk_loss"])

    groups = {
        "encoder": model.encoder,
        "transformer": model.transformer,
        "policy_head": model.policy_head,
        "iwg_gate_residual_head": model.iwg_gate_residual_head,
        "risk_head": model.risk_head,
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

    label = {
            "candidate_type": "A",
            "motion_soft_target": 0.1,
            "appearance_soft_target": 0.2,
            "motion_safe_target": 0.8,
            "appearance_safe_target": 0.9,
            "motion_label_confidence": 0.4,
            "appearance_label_confidence": 0.6,
            "policy_soft_target": [1.0, 0.0, 0.0, 0.0, 0.0],
            "policy_safe_soft_target": [0.0, 1.0, 0.0, 0.0, 0.0],
            "motion_benefit": -0.1,
            "appearance_benefit": -0.2,
            "valid_motion": True,
            "valid_appearance": True,
            "sample_weight": 1.0,
            "label_schema_version": ROLLOUT_LABEL_SCHEMA_VERSION,
            "label_schema_sha256": ROLLOUT_LABEL_SCHEMA_SHA256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "cache_schema_version": COMPACT_CACHE_SCHEMA_VERSION,
    }
    label["cue_target"] = make_cue_target(0.4, 0.6)
    label["risk_targets"] = make_risk_targets(0.1, 0.2, 0.4, 0.6)
    targets = _CompactDatasetBase._targets(label)
    assert targets["motion_target"].item() == pytest.approx(0.1)
    assert targets["appearance_target"].item() == pytest.approx(0.2)
    assert targets["motion_safe_target"].item() == pytest.approx(0.8)
    assert targets["appearance_safe_target"].item() == pytest.approx(0.9)
    assert targets["motion_label_confidence"].item() == pytest.approx(0.4)
    assert targets["appearance_label_confidence"].item() == pytest.approx(0.6)
    np.testing.assert_allclose(
        targets["policy_soft_target"].numpy(),
        [1.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_compact_dataset_rejects_legacy_or_incomplete_labels():
    from agentguard.v0_pipeline import _CompactDatasetBase

    with pytest.raises(ValueError, match="missing required fields"):
        _CompactDatasetBase._targets({"label_schema_version": 2})


def test_iwg_training_loss_supervises_cue_head():
    from agentguard.training.train_iwg import _compute_iwg_loss

    model = IWG(reid_dim=16).train()
    outputs = model(**_inputs())
    loss, components = _compute_iwg_loss(outputs, _production_targets())
    loss.backward()

    assert "cue_loss" in components
    assert "risk_loss" in components
    assert torch.isfinite(components["cue_loss"])
    gradients = [parameter.grad for parameter in model.cue_head.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    risk_gradients = [parameter.grad for parameter in model.risk_head.parameters()]
    assert all(gradient is not None for gradient in risk_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in risk_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in risk_gradients)
