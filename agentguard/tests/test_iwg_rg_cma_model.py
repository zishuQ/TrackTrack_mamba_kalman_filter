from __future__ import annotations

import torch
import torch.nn.functional as F

from agentguard.models.event_encoder import EventEncoder
from agentguard.models.iwg_rg_cma import (
    IWGRGCMA,
    RG_CMA_CORRECTION_BOUND,
    SafeDirectIWG,
)
from agentguard.training.loss_iwg_rg_cma import compute_iwg_rg_cma_loss


def _batch(batch_size: int = 2, length: int = 8, reid_dim: int = 16) -> dict:
    generator = torch.Generator().manual_seed(71)
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    padding[0, :2] = True
    detection = ~padding
    detection[0, 4] = False
    reset = torch.zeros_like(padding)
    reset[0, 2] = True
    if batch_size > 1:
        reset[1, 0] = True
    safe = torch.rand((batch_size, 2), generator=generator)
    policy = torch.rand((batch_size, 5), generator=generator)
    policy /= policy.sum(dim=-1, keepdim=True)
    return {
        "track_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "det_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "scalar_feats": torch.randn(batch_size, length, 63, generator=generator),
        "padding_mask": padding,
        "has_detection_mask": detection,
        "reset_mask": reset,
        "safe_gate_target": safe,
        "policy_safe_soft_target": policy,
        "cue_target": torch.rand((batch_size, 3), generator=generator),
        "risk_target": torch.rand((batch_size, 4), generator=generator),
        "valid_motion": torch.ones(batch_size, dtype=torch.bool),
        "valid_appearance": torch.ones(batch_size, dtype=torch.bool),
        "sample_weight": torch.ones(batch_size),
    }


def _forward(model: IWGRGCMA, batch: dict) -> dict:
    return model(
        batch["track_feats"],
        batch["det_feats"],
        batch["scalar_feats"],
        batch["padding_mask"],
        batch["has_detection_mask"],
        batch["reset_mask"],
    )


def _has_finite_nonzero_grad(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(grad).all() for grad in gradients) and any(
        grad.abs().sum() > 0 for grad in gradients
    )


def _has_no_grad(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or parameter.grad.abs().sum() == 0
        for parameter in module.parameters()
    )


def test_event_encoder_exposes_finite_modal_tokens_without_changing_fusion():
    torch.manual_seed(3)
    encoder = EventEncoder(16).eval()
    track = torch.randn(5, 16)
    detection = torch.randn(5, 16)
    scalar = torch.randn(5, 63)
    fused = encoder(track, detection, scalar)
    exposed, appearance, motion = encoder.forward_with_modal_tokens(
        track, detection, scalar
    )
    torch.testing.assert_close(exposed, fused, atol=0.0, rtol=0.0)
    assert appearance.shape == motion.shape == (5, 64)
    assert torch.isfinite(appearance).all() and torch.isfinite(motion).all()


def test_safe_direct_uses_linear_linear_gate_heads_and_detached_auxiliary_heads():
    model = SafeDirectIWG(16)
    assert list(type(layer) for layer in model.policy_head) == [
        torch.nn.Linear,
        torch.nn.Linear,
    ]
    assert list(type(layer) for layer in model.iwg_gate_residual_head) == [
        torch.nn.Linear,
        torch.nn.Linear,
    ]


def test_rg_cma_zero_init_masks_and_correction_bound():
    model = IWGRGCMA(16).eval()
    batch = _batch()
    with torch.no_grad():
        outputs = _forward(model, batch)
    torch.testing.assert_close(outputs["refined_gate"], outputs["base_gate"])
    assert outputs["gate_correction"].abs().max() == 0
    assert outputs["appearance_token"].shape == (2, 64)
    assert outputs["motion_token"].shape == (2, 64)
    assert outputs["motion_attention_weights"].shape == (2, 4, 6)
    assert outputs["cross_modal_attention_weights"].shape == (2, 4, 3, 3)
    # Use a six-position input so the two left-padding positions remain in the
    # current causal window.
    short_batch = _batch(length=6)
    with torch.no_grad():
        short_outputs = _forward(model, short_batch)
    assert short_outputs["motion_attention_weights"][0, :, :2].abs().max() == 0
    assert short_outputs["appearance_attention_weights"][0, :, :2].abs().max() == 0

    reset_current = dict(batch)
    reset_current["reset_mask"] = torch.zeros_like(batch["reset_mask"])
    reset_current["reset_mask"][:, -1] = True
    with torch.no_grad():
        reset_outputs = _forward(model, reset_current)
    assert reset_outputs["motion_attention_weights"][:, :, :-1].abs().max() == 0
    assert reset_outputs["appearance_attention_weights"][:, :, :-1].abs().max() == 0

    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            head[-1].weight.fill_(50.0)
            head[-1].bias.fill_(50.0)
        bounded = _forward(model, batch)
    assert bounded["gate_correction"].abs().max() <= RG_CMA_CORRECTION_BOUND + 1e-6


def test_no_scalar_reliability_mode_only_masks_scalar_evidence():
    torch.manual_seed(37)
    full = IWGRGCMA(16).eval()
    no_scalar = IWGRGCMA(16, reliability_mode="no-scalar").eval()
    no_scalar.load_state_dict(full.state_dict(), strict=True)
    batch = _batch()
    with torch.no_grad():
        outputs = _forward(full, batch)
        base_outputs = {
            key: outputs[key]
            for key in ("base_gate", "policy_probs", "cue", "risk")
        }
        scalar_current = batch["scalar_feats"][:, -1]
        full_input = full._build_reliability_input(
            scalar_current=scalar_current,
            base_outputs=base_outputs,
        )
        no_scalar_input = no_scalar._build_reliability_input(
            scalar_current=scalar_current,
            base_outputs=base_outputs,
        )
        no_scalar_zero_input = no_scalar._build_reliability_input(
            scalar_current=torch.zeros_like(scalar_current),
            base_outputs=base_outputs,
        )
    torch.testing.assert_close(no_scalar_input, no_scalar_zero_input)
    assert not torch.equal(full_input, no_scalar_input)


def test_rg_cma_sequence_is_strictly_causal():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = IWGRGCMA(16).eval()
    batch = _batch(batch_size=1, length=9)
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    changed["track_feats"][:, 6:] += 80.0
    changed["det_feats"][:, 6:] -= 70.0
    changed["scalar_feats"][:, 6:] *= -30.0
    with torch.no_grad():
        original = model.forward_sequence(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["has_detection_mask"],
            batch["reset_mask"],
        )
        perturbed = model.forward_sequence(
            changed["track_feats"],
            changed["det_feats"],
            changed["scalar_feats"],
            changed["padding_mask"],
            changed["has_detection_mask"],
            changed["reset_mask"],
        )
    for key in (
        "appearance_token",
        "motion_token",
        "event_embedding",
        "base_gate",
        "refined_gate",
        "motion_attention_weights",
        "appearance_attention_weights",
        "cross_modal_attention_weights",
    ):
        torch.testing.assert_close(
            original[key][:, :6], perturbed[key][:, :6], atol=1e-6, rtol=0.0
        )


def test_production_loss_gradients_and_iwg_gradient_isolation():
    torch.manual_seed(17)
    model = IWGRGCMA(16).train()
    # After the zero-preserving first update, gradients reach all RG-CMA layers.
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.02)
    batch = _batch()
    outputs = _forward(model, batch)
    loss, _components = compute_iwg_rg_cma_loss(outputs, batch)
    loss.backward()
    expected = (
        model.iwg.encoder,
        model.iwg.transformer,
        model.iwg.policy_head,
        model.iwg.iwg_gate_residual_head,
        model.iwg.cue_head,
        model.iwg.risk_head,
        model.motion_projection,
        model.appearance_projection,
        model.motion_temporal_attention,
        model.appearance_temporal_attention,
        model.reliability_projection,
        model.cross_modal_attention,
        model.motion_correction_head,
        model.appearance_correction_head,
    )
    assert all(_has_finite_nonzero_grad(module) for module in expected)

    model.zero_grad(set_to_none=True)
    outputs = _forward(model, batch)
    correction_target = torch.clamp(
        batch["safe_gate_target"] - outputs["base_gate"].detach(),
        -RG_CMA_CORRECTION_BOUND,
        RG_CMA_CORRECTION_BOUND,
    )
    isolated = (
        F.binary_cross_entropy_with_logits(outputs["cue_logits"], batch["cue_target"])
        + F.binary_cross_entropy_with_logits(outputs["risk_logits"], batch["risk_target"])
        + F.binary_cross_entropy(
            outputs["refined_gate"].clamp(1e-6, 1 - 1e-6),
            batch["safe_gate_target"],
        )
        + F.smooth_l1_loss(outputs["gate_correction"], correction_target)
        + outputs["gate_correction"].abs().mean()
    )
    isolated.backward()
    assert _has_no_grad(model.iwg.encoder)
    assert _has_no_grad(model.iwg.transformer)
    assert _has_no_grad(model.iwg.policy_head)
    assert _has_no_grad(model.iwg.iwg_gate_residual_head)
    assert _has_finite_nonzero_grad(model.iwg.cue_head)
    assert _has_finite_nonzero_grad(model.iwg.risk_head)
    assert _has_finite_nonzero_grad(model.motion_temporal_attention)
    assert _has_finite_nonzero_grad(model.cross_modal_attention)


def test_loss_v2_is_explicit_and_reports_correction_diagnostics():
    torch.manual_seed(23)
    model = IWGRGCMA(16).eval()
    batch = _batch()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.02)
            torch.nn.init.normal_(head[-1].bias, std=0.02)
        outputs = _forward(model, batch)

    default_loss, default_components = compute_iwg_rg_cma_loss(outputs, batch)
    explicit_old_loss, explicit_old_components = compute_iwg_rg_cma_loss(
        outputs,
        batch,
        residual_beta=1.0,
        residual_weight=0.5,
        revision_weight=0.01,
        hard_example_gain=0.0,
    )
    torch.testing.assert_close(default_loss, explicit_old_loss)
    for key in default_components:
        torch.testing.assert_close(default_components[key], explicit_old_components[key])

    loss_v2, components_v2 = compute_iwg_rg_cma_loss(
        outputs,
        batch,
        residual_beta=0.01,
        residual_weight=1.0,
        revision_weight=0.0,
        hard_example_gain=2.0,
    )
    assert torch.isfinite(loss_v2)
    for key in (
        "correction_abs_mean",
        "correction_abs_p50",
        "correction_abs_p95",
        "correction_target_abs_mean",
        "correction_nonzero_rate",
        "correction_saturation_rate",
        "correction_sign_agreement",
    ):
        assert key in components_v2
        assert torch.isfinite(components_v2[key])
    assert components_v2["correction_abs_p95"] >= components_v2["correction_abs_p50"]
    assert 0.0 <= components_v2["correction_nonzero_rate"] <= 1.0
    assert 0.0 <= components_v2["correction_saturation_rate"] <= 1.0
