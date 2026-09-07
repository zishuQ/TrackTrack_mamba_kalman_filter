from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agentguard.models.event_encoder import EventEncoder
from agentguard.models.iwg_rg_cma import (
    ARCHITECTURE_CLEAN_CROSS_MODAL,
    ARCHITECTURE_DIRECT_BASE,
    ARCHITECTURE_LEGACY,
    ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
    ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
    ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
    ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
    ARCHITECTURE_SELECTIVE_CORRECTION,
    CLEAN_MOTION_SCALAR_INDICES,
    CLEAN_RELIABILITY_SCALAR_INDICES,
    IWGRGCMA,
    RG_CMA_CORRECTION_BOUND,
    SafeDirectIWG,
)
from agentguard.training.loss_iwg_rg_cma import (
    _safe_selective_correction_target,
    compute_iwg_rg_cma_loss,
)


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


def test_safe_selective_target_has_exact_abstention_and_decisive_residuals():
    safe = torch.tensor([[0.8, 0.8], [0.325, 0.925]])
    oracle = torch.tensor([[0.1, 0.55], [0.1, 0.9]])
    confidence = torch.tensor([[0.49, 0.9], [0.75, 0.75]])

    target, unlock = _safe_selective_correction_target(
        safe,
        oracle,
        confidence,
        correction_bound=0.05,
    )

    torch.testing.assert_close(unlock[0], torch.zeros(2))
    torch.testing.assert_close(target[0], torch.zeros(2))
    torch.testing.assert_close(unlock[1], torch.ones(2))
    torch.testing.assert_close(target[1], torch.tensor([-0.05, -0.025]))
    assert (target <= 0.0).all()


def test_safe_selective_auxiliary_losses_do_not_oppose_exact_target():
    batch = _batch(batch_size=2, length=6)
    base = torch.tensor([[0.4, 0.4], [0.1, 0.1]])
    correction = torch.full((2, 2), -0.05, requires_grad=True)
    batch.update(
        {
            "safe_gate_target": torch.tensor(
                [[0.4375, 0.4375], [0.28, 0.28]]
            ),
            "oracle_gate_target": torch.full((2, 2), 0.1),
            "gate_confidence": torch.tensor(
                [[0.625, 0.625], [0.8, 0.8]]
            ),
        }
    )
    outputs = {
        "base_gate": base,
        "refined_gate": torch.clamp(base + correction, 0.0, 1.0),
        "gate_correction": correction,
        "policy_probs": batch["policy_safe_soft_target"],
        "cue_logits": torch.zeros((2, 3)),
        "risk_logits": torch.zeros((2, 4)),
    }

    _, components = compute_iwg_rg_cma_loss(
        outputs,
        batch,
        correction_bound=0.05,
        residual_target_mode="safe-selective",
        revision_weight=0.05,
        no_harm_weight=0.1,
    )

    torch.testing.assert_close(components["residual_loss"], torch.tensor(0.0))
    torch.testing.assert_close(components["revision_loss"], torch.tensor(0.0))
    torch.testing.assert_close(components["no_harm_loss"], torch.tensor(0.0))

    wrong = dict(outputs)
    wrong_correction = torch.full((2, 2), 0.01)
    wrong["gate_correction"] = wrong_correction
    wrong["refined_gate"] = torch.clamp(base + wrong_correction, 0.0, 1.0)
    _, wrong_components = compute_iwg_rg_cma_loss(
        wrong,
        batch,
        correction_bound=0.05,
        residual_target_mode="safe-selective",
        revision_weight=0.05,
        no_harm_weight=0.1,
    )
    assert wrong_components["no_harm_loss"] > 0.0


def test_safe_selective_cma_gradient_is_nonzero_and_base_is_isolated():
    torch.manual_seed(29)
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
    ).train()
    batch = _batch(batch_size=2, length=6)
    oracle = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    confidence = torch.full((2, 2), 0.75)
    batch["oracle_gate_target"] = oracle
    batch["gate_confidence"] = confidence
    batch["safe_gate_target"] = confidence * oracle + (1.0 - confidence)

    outputs = _forward(model, batch)
    _, components = compute_iwg_rg_cma_loss(
        outputs,
        batch,
        correction_bound=0.05,
        residual_target_mode="safe-selective",
        residual_weight=0.5,
        revision_weight=0.05,
        no_harm_weight=0.1,
    )
    cma_objective = (
        components["final_loss"]
        + 0.5 * components["residual_loss"]
        + 0.05 * components["revision_loss"]
        + 0.1 * components["no_harm_loss"]
    )
    cma_objective.backward()

    assert _has_no_grad(model.iwg)
    assert _has_finite_nonzero_grad(model.motion_correction_head)
    assert _has_finite_nonzero_grad(model.appearance_correction_head)


def test_direct_base_gate_is_continuous_and_policy_is_auxiliary_only():
    torch.manual_seed(101)
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_DIRECT_BASE,
    ).eval()
    batch = _batch()
    with torch.no_grad():
        before = _forward(model, batch)
        for parameter in model.iwg.policy_head.parameters():
            parameter.add_(torch.randn_like(parameter) * 20.0)
        after = _forward(model, batch)
    torch.testing.assert_close(before["base_gate"], after["base_gate"])
    assert not torch.equal(before["policy_probs"], after["policy_probs"])
    assert ((before["base_gate"] > 0.0) & (before["base_gate"] < 1.0)).all()


def test_clean_cross_modal_uses_pure_motion_indices_and_two_tokens():
    torch.manual_seed(103)
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_CLEAN_CROSS_MODAL,
    ).eval()
    scalar = torch.randn(2, 6, 63)
    reliability_changed = scalar.clone()
    reliability_changed[..., list(CLEAN_RELIABILITY_SCALAR_INDICES)] += 50.0
    motion_changed = scalar.clone()
    motion_changed[..., CLEAN_MOTION_SCALAR_INDICES[0]] += 50.0
    with torch.no_grad():
        clean = model._encode_clean_motion(scalar)
        same_motion = model._encode_clean_motion(reliability_changed)
        changed_motion = model._encode_clean_motion(motion_changed)
        outputs = _forward(model, _batch())
    torch.testing.assert_close(clean, same_motion)
    assert not torch.equal(clean, changed_motion)
    assert outputs["cross_modal_attention_weights"].shape == (2, 4, 2, 2)
    assert not hasattr(model, "reliability_projection")


def test_legacy_clean_cross_modal_preserves_legacy_base_and_uses_two_tokens():
    torch.manual_seed(105)
    legacy = IWGRGCMA(16, correction_bound=0.05).eval()
    clean = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL,
    ).eval()
    clean.iwg.load_state_dict(legacy.iwg.state_dict(), strict=True)
    batch = _batch()
    with torch.no_grad():
        legacy_base = legacy.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        clean_base = clean.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        outputs = _forward(clean, batch)
    torch.testing.assert_close(legacy_base["base_gate"], clean_base["base_gate"])
    assert outputs["cross_modal_attention_weights"].shape == (2, 4, 2, 2)
    assert not hasattr(clean, "reliability_projection")


def test_base_conditioned_correction_heads_add_one_base_gate_feature():
    torch.manual_seed(106)
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED,
    ).eval()
    batch = _batch()
    with torch.no_grad():
        outputs = _forward(model, batch)
    assert model.motion_correction_head[0].in_features == 129
    assert model.appearance_correction_head[0].in_features == 129
    assert outputs["cross_modal_attention_weights"].shape == (2, 4, 2, 2)
    assert not hasattr(model, "reliability_projection")


def test_bidirectional_cma_uses_both_opposite_modality_histories():
    torch.manual_seed(108)
    legacy = IWGRGCMA(16, correction_bound=0.05).eval()
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA,
    ).eval()
    model.iwg.load_state_dict(legacy.iwg.state_dict(), strict=True)
    batch = _batch()
    with torch.no_grad():
        legacy_base = legacy.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        enhanced_base = model.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        outputs = _forward(model, batch)
    torch.testing.assert_close(legacy_base["base_gate"], enhanced_base["base_gate"])
    assert outputs["appearance_to_motion_attention_weights"].shape == (2, 4, 6)
    assert outputs["motion_to_appearance_attention_weights"].shape == (2, 4, 6)
    assert outputs["cross_modal_attention_weights"].shape == (2, 4, 2, 2)
    assert hasattr(model, "appearance_to_motion_cross_attention")
    assert hasattr(model, "motion_to_appearance_cross_attention")
    assert model.motion_correction_head[0].in_features == 128
    assert model.appearance_correction_head[0].in_features == 128
    assert not hasattr(model, "reliability_projection")

    model.train()
    for head in (model.motion_correction_head, model.appearance_correction_head):
        torch.nn.init.normal_(head[-1].weight, std=0.1)
        torch.nn.init.normal_(head[-1].bias, std=0.1)
    _forward(model, batch)["refined_gate"].sum().backward()
    assert _has_finite_nonzero_grad(model.appearance_to_motion_cross_attention)
    assert _has_finite_nonzero_grad(model.motion_to_appearance_cross_attention)


def test_full_history_6x6_cma_precedes_endpoint_temporal_readout():
    torch.manual_seed(110)
    legacy = IWGRGCMA(16, correction_bound=0.05).eval()
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_LEGACY_CLEAN_6X6_CMA,
    ).eval()
    model.iwg.load_state_dict(legacy.iwg.state_dict(), strict=True)
    batch = _batch()
    with torch.no_grad():
        legacy_base = legacy.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        enhanced_base = model.iwg(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["reset_mask"],
        )
        outputs = _forward(model, batch)
    torch.testing.assert_close(legacy_base["base_gate"], enhanced_base["base_gate"])
    assert outputs["appearance_to_motion_6x6_attention_weights"].shape == (
        2,
        4,
        6,
        6,
    )
    assert outputs["motion_to_appearance_6x6_attention_weights"].shape == (
        2,
        4,
        6,
        6,
    )
    assert outputs["motion_attention_weights"].shape == (2, 4, 6)
    assert outputs["appearance_attention_weights"].shape == (2, 4, 6)
    assert "cross_modal_attention_weights" not in outputs
    assert hasattr(model, "bidirectional_history_cross_attention")
    assert not hasattr(model, "appearance_to_motion_cross_attention")
    assert not hasattr(model, "motion_to_appearance_cross_attention")
    assert not hasattr(model, "cross_modal_attention")

    model.train()
    for head in (model.motion_correction_head, model.appearance_correction_head):
        torch.nn.init.normal_(head[-1].weight, std=0.1)
        torch.nn.init.normal_(head[-1].bias, std=0.1)
    _forward(model, batch)["refined_gate"].sum().backward()
    assert _has_finite_nonzero_grad(model.bidirectional_history_cross_attention)
    assert _has_finite_nonzero_grad(model.motion_temporal_attention)
    assert _has_finite_nonzero_grad(model.appearance_temporal_attention)


def test_selective_correction_scales_raw_delta_and_no_harm_is_explicit():
    torch.manual_seed(107)
    model = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_SELECTIVE_CORRECTION,
    ).eval()
    batch = _batch()
    with torch.no_grad():
        for head in (model.motion_correction_head, model.appearance_correction_head):
            torch.nn.init.normal_(head[-1].weight, std=0.1)
            torch.nn.init.normal_(head[-1].bias, std=0.1)
        outputs = _forward(model, batch)
    assert ((outputs["correction_scale"] >= 0.0) & (outputs["correction_scale"] <= 1.0)).all()
    torch.testing.assert_close(
        outputs["gate_correction"],
        outputs["raw_gate_correction"] * outputs["correction_scale"],
    )

    controlled = dict(outputs)
    controlled["base_gate"] = batch["safe_gate_target"].clone()
    controlled["refined_gate"] = torch.clamp(
        batch["safe_gate_target"] + 0.2, 0.0, 1.0
    )
    controlled["gate_correction"] = (
        controlled["refined_gate"] - controlled["base_gate"]
    )
    loss_without, components = compute_iwg_rg_cma_loss(
        controlled, batch, correction_bound=0.05, no_harm_weight=0.0
    )
    loss_with, _ = compute_iwg_rg_cma_loss(
        controlled, batch, correction_bound=0.05, no_harm_weight=1.0
    )
    assert components["no_harm_loss"] > 0.0
    torch.testing.assert_close(
        loss_with - loss_without, components["no_harm_loss"]
    )


def test_clean_and_selective_variants_share_identical_common_initialization():
    torch.manual_seed(109)
    clean = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_CLEAN_CROSS_MODAL,
    )
    torch.manual_seed(109)
    selective = IWGRGCMA(
        16,
        correction_bound=0.05,
        architecture_variant=ARCHITECTURE_SELECTIVE_CORRECTION,
    )
    clean_state = clean.state_dict()
    selective_state = selective.state_dict()
    assert set(clean_state).issubset(selective_state)
    for key, value in clean_state.items():
        torch.testing.assert_close(value, selective_state[key])


_PREDICTION_KEYS = (
    "base_gate",
    "refined_gate",
    "final_gate",
    "gate_correction",
    "policy_probs",
    "cue",
)
_INFERENCE_VARIANTS = (
    (False, ARCHITECTURE_CLEAN_CROSS_MODAL, 6),
    (False, ARCHITECTURE_DIRECT_BASE, 6),
    (False, ARCHITECTURE_LEGACY, 6),
    (False, ARCHITECTURE_LEGACY_CLEAN_6X6_CMA, 6),
    (False, ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA, 6),
    (False, ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL, 6),
    (False, ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED, 6),
    (False, ARCHITECTURE_SELECTIVE_CORRECTION, 6),
    (False, ARCHITECTURE_LEGACY, 8),
    (True, ARCHITECTURE_CLEAN_CROSS_MODAL, 6),
    (True, ARCHITECTURE_DIRECT_BASE, 6),
    (True, ARCHITECTURE_LEGACY, 6),
    (True, ARCHITECTURE_LEGACY_CLEAN_6X6_CMA, 6),
    (True, ARCHITECTURE_LEGACY_CLEAN_BIDIRECTIONAL_CMA, 6),
    (True, ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL, 6),
    (True, ARCHITECTURE_LEGACY_CLEAN_CROSS_MODAL_BASE_CONDITIONED, 6),
    (True, ARCHITECTURE_SELECTIVE_CORRECTION, 6),
    (True, ARCHITECTURE_LEGACY, 8),
)


def _variant_model(variant: str, context_size: int) -> IWGRGCMA:
    kwargs = {"context_size": context_size, "architecture_variant": variant}
    if variant != ARCHITECTURE_LEGACY:
        kwargs["correction_bound"] = 0.05
    return IWGRGCMA(16, **kwargs).eval()


@pytest.mark.parametrize(
    "return_diagnostics, variant, context_size",
    _INFERENCE_VARIANTS,
)
def test_inference_without_diagnostics_preserves_predictions(
    return_diagnostics, variant, context_size
):
    torch.manual_seed(41)
    model = _variant_model(variant, context_size)
    batch = _batch(length=context_size)
    with torch.no_grad():
        with_diag = model(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["has_detection_mask"],
            batch["reset_mask"],
            return_diagnostics=True,
        )
        without_diag = model(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["has_detection_mask"],
            batch["reset_mask"],
            return_diagnostics=False,
        )
        selected = model(
            batch["track_feats"],
            batch["det_feats"],
            batch["scalar_feats"],
            batch["padding_mask"],
            batch["has_detection_mask"],
            batch["reset_mask"],
            return_diagnostics=return_diagnostics,
        )
    for key in _PREDICTION_KEYS:
        torch.testing.assert_close(with_diag[key], without_diag[key], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(selected[key], with_diag[key], atol=1e-6, rtol=0.0)
    assert "motion_attention_weights" in with_diag
    assert "motion_attention_weights" not in without_diag
    if return_diagnostics:
        assert "motion_attention_weights" in selected
    else:
        assert "motion_attention_weights" not in selected
