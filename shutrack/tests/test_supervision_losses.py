"""Unit tests for the paired-supervision losses (appearance rank / KF speed)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from shutrack.training.loss import (
    appearance_rank_loss,
    gated_appearance_template,
    motion_speed_l1_loss,
)


def test_gated_template_matches_online_formula() -> None:
    torch.manual_seed(0)
    template = F.normalize(torch.randn(4, 16), dim=-1)
    detection = F.normalize(torch.randn(4, 16), dim=-1)
    score = torch.tensor([0.9, 0.8, 0.7, 0.6])
    for gate_value in (0.0, 0.5, 1.0):
        gate = torch.full((4,), gate_value)
        actual = gated_appearance_template(
            template, detection, score, gate, alpha=0.95
        )
        for index in range(4):
            beta = 0.95 + 0.05 * (1.0 - float(score[index]))
            full = beta * template[index] + (1.0 - beta) * detection[index]
            if gate_value == 0.0:
                expected = template[index]
            elif gate_value == 1.0:
                expected = full / full.norm()
            else:
                blended = (1.0 - gate_value) * template[index] + gate_value * full
                expected = blended / blended.norm()
            assert torch.allclose(actual[index], expected, atol=1e-6)


def _appearance_batch(batch_size: int = 8, dim: int = 16) -> dict:
    torch.manual_seed(1)
    return {
        "rank_template0": F.normalize(torch.randn(batch_size, dim), dim=-1),
        "rank_detection": F.normalize(torch.randn(batch_size, dim), dim=-1),
        "rank_positive": F.normalize(torch.randn(batch_size, dim), dim=-1),
        "rank_competitor": F.normalize(torch.randn(batch_size, dim), dim=-1),
        "rank_score": torch.full((batch_size,), 0.8),
        "rank_valid": torch.ones(batch_size, dtype=torch.bool),
        "appearance_valid": torch.ones(batch_size, dtype=torch.bool),
        "appearance_target": torch.full((batch_size,), 0.5),
    }


def test_rank_loss_gradient_only_on_appearance_channel() -> None:
    batch = _appearance_batch()
    logits = torch.zeros(8, 2, requires_grad=True)
    loss, components = appearance_rank_loss({"gate_logits": logits}, batch)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss)
    assert float(grad[:, 0].abs().max()) == 0.0
    assert float(grad[:, 1].abs().sum()) > 0.0
    assert float(components["rank_valid_ratio"]) == 1.0


def test_rank_loss_fallback_uses_soft_label() -> None:
    batch = _appearance_batch()
    batch["rank_valid"] = torch.zeros(8, dtype=torch.bool)
    batch["appearance_target"] = torch.ones(8)
    logits = torch.zeros(8, 2, requires_grad=True)
    loss, components = appearance_rank_loss({"gate_logits": logits}, batch)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss)
    assert float(components["fallback_ratio"]) == 1.0
    assert float(grad[:, 0].abs().max()) == 0.0
    assert float(grad[:, 1].abs().sum()) > 0.0


def test_speed_l1_gradient_only_on_motion_channel() -> None:
    batch = {
        "kf_velocity_g0": torch.tensor([[0.01, -0.02], [0.0, 0.01], [0.02, 0.0]]),
        "kf_velocity_g1": torch.tensor([[0.012, -0.018], [0.001, 0.011], [0.019, 0.002]]),
        "gt_velocity": torch.tensor([[0.011, -0.019], [0.0, 0.01], [0.021, 0.001]]),
        "speed_valid": torch.tensor([True, True, False]),
    }
    logits = torch.zeros(3, 2, requires_grad=True)
    loss, components = motion_speed_l1_loss({"gate_logits": logits}, batch, weight=50.0)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss)
    assert float(grad[:, 1].abs().max()) == 0.0
    assert float(grad[:, 0].abs().sum()) > 0.0
    assert float(components["valid_ratio"]) > 0.0
