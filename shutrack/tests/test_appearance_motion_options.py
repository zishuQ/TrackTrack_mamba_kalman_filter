"""Gate-loss tests for the motion channel weight."""

import torch
import torch.nn.functional as F

from shutrack.training.loss import gate_loss


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
