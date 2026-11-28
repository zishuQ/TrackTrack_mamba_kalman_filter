"""HCE/RSC model contract, gradient, branch and cold-start tests."""

from __future__ import annotations

import pytest
import torch

from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.tests.conftest import make_model
from shutrack.training.loss import gate_loss


def _batch(
    batch_size: int,
    reid_dim: int = 8,
    mask=None,
    reset=None,
    competition_enabled: bool = True,
):
    generator = torch.Generator().manual_seed(3)
    shape = (batch_size, 5, reid_dim)
    batch = {
        "history_evidence": torch.randn(batch_size, 5, 13, generator=generator),
        "history_track": torch.randn(*shape, generator=generator),
        "history_detection": torch.randn(*shape, generator=generator),
        "history_mask": (
            torch.zeros(batch_size, 5, dtype=torch.bool)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool)
        ),
        "history_has_detection": torch.ones(batch_size, 5, dtype=torch.bool),
        "current_evidence": torch.randn(batch_size, 13, generator=generator),
        "current_track": torch.randn(batch_size, reid_dim, generator=generator),
        "current_detection": torch.randn(batch_size, reid_dim, generator=generator),
        "current_competition": torch.randn(batch_size, 3, generator=generator),
        "current_has_detection": torch.ones(batch_size, dtype=torch.bool),
    }
    if not competition_enabled:
        batch["current_competition"] = torch.zeros(batch_size, 3)
    if reset is not None:
        batch["history_reset"] = torch.as_tensor(reset, dtype=torch.bool)
    return batch


def test_gate_shapes_range_and_independence():
    model = make_model()
    model.eval()
    with torch.no_grad():
        outputs = model(**_batch(6))
    gates = outputs["gates"]
    assert outputs["gate_logits"].shape == (6, 2)
    assert gates.shape == (6, 2)
    assert torch.isfinite(gates).all()
    assert ((gates >= 0) & (gates <= 1)).all()
    sums = gates.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_no_legacy_heads_or_parameters():
    model = make_model()
    names = [name.lower() for name, _ in model.named_parameters()]
    for forbidden in ("policy", "cue", "risk", "correction", "refined", "base_gate"):
        assert all(forbidden not in name for name in names)
    assert hasattr(model, "hce") and hasattr(model, "rsc")


def test_competition_branch_on_and_off():
    full = make_model(competition_enabled=True)
    lean = make_model(competition_enabled=False)
    assert full.rsc.condition_encoder is not None
    assert lean.rsc.condition_encoder is None
    full.eval()
    lean.eval()
    with torch.no_grad():
        full_gates = full(**_batch(4))["gates"]
        lean_gates = lean(**_batch(4, competition_enabled=False))["gates"]
    assert torch.isfinite(full_gates).all()
    assert torch.isfinite(lean_gates).all()
    assert full.parameter_report()["total"] > lean.parameter_report()["total"]


def test_cold_start_and_all_padding_history_is_finite():
    model = make_model()
    model.eval()
    batch = _batch(3)
    batch["history_evidence"].zero_()
    batch["history_track"].zero_()
    batch["history_detection"].zero_()
    batch["history_mask"][:] = True
    batch["history_has_detection"][:] = False
    with torch.no_grad():
        outputs = model(**batch)
    assert torch.isfinite(outputs["gates"]).all()


def test_current_without_detection_is_finite():
    model = make_model()
    model.eval()
    batch = _batch(2)
    batch["current_has_detection"][:] = False
    batch["current_detection"].zero_()
    with torch.no_grad():
        outputs = model(**batch)
    assert torch.isfinite(outputs["gates"]).all()


def test_history_reset_mask_removes_history_influence():
    model = make_model()
    model.eval()
    first = _batch(1)
    second = _batch(1)
    second["history_evidence"] = torch.randn_like(second["history_evidence"]) * 7.0
    second["history_track"] = torch.randn_like(second["history_track"]) * 7.0
    second["history_detection"] = torch.randn_like(second["history_detection"]) * 7.0
    reset = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        gates_first = model(**{**first, "history_reset": reset})["gates"]
        gates_second = model(**{**second, "history_reset": reset})["gates"]
    torch.testing.assert_close(gates_first, gates_second, atol=1e-6, rtol=1e-6)


def test_hce_and_rsc_receive_valid_gradients():
    model = make_model()
    outputs = model(**_batch(8))
    batch = {
        "safe_gate_target": torch.rand(8, 2),
        "valid_channels": torch.ones(8, 2, dtype=torch.bool),
        "sample_weight": torch.ones(8),
    }
    loss, _components = gate_loss(outputs, batch)
    loss.backward()
    for module_name in ("hce", "rsc"):
        module = getattr(model, module_name)
        gradient_norm = sum(
            float(parameter.grad.norm())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert gradient_norm > 0.0, f"{module_name} received no gradient"


def test_loss_all_invalid_is_finite_zero_with_gradient_path():
    model = make_model()
    outputs = model(**_batch(2))
    batch = {
        "safe_gate_target": torch.zeros(2, 2),
        "valid_channels": torch.zeros(2, 2, dtype=torch.bool),
        "sample_weight": torch.ones(2),
    }
    loss, components = gate_loss(outputs, batch)
    assert float(loss) == 0.0
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.hce.parameters()
    )


def test_parameter_growth_is_modest():
    full = ShutrackV2Model(ShutrackV2Config(reid_dim=2048))
    lean = ShutrackV2Model(
        ShutrackV2Config(reid_dim=2048, competition_enabled=False)
    )
    assert full.parameter_report()["total"] < 2_000_000
    assert lean.parameter_report()["total"] < full.parameter_report()["total"]
    assert full.parameter_report()["hce"] > 0
    assert full.parameter_report()["rsc"] > 0
