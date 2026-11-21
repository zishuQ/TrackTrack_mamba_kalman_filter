"""Contracts for the opt-in query/FiLM/bilinear architecture."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from shutrack.models.query_controller import BilinearGate, QueryRSC
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.test_model import _batch
from shutrack.training.checkpoint import (
    build_checkpoint_payload, model_from_config, save_checkpoint_atomic,
)


def new_model(**options):
    config = dict(reid_dim=8, d_model=32, n_heads=4, hce_layers=2,
                  ff_dim=64, appearance_proj_dim=16,
                  architecture_variant="query_film_bilinear")
    config.update(options)
    torch.manual_seed(42)
    return ShutrackV2Model(ShutrackV2Config(**config))


def save_new(path, model):
    payload = build_checkpoint_payload(
        model=model, training_config={}, dataset_descriptor={},
        epoch=0, global_step=0, best_metric=1.0,
        normalization_mean=np.zeros(63), normalization_std=np.ones(63),
        include_optimizer=False,
    )
    save_checkpoint_atomic(path, payload)
    return payload


def test_exact_full_size_parameter_budget():
    model = new_model(reid_dim=2048, d_model=128, ff_dim=256, appearance_proj_dim=256)
    report = model.parameter_report()
    assert report["total"] == 1_208_220
    assert report["hce"] == 398_336
    assert report["rsc"] == 84_866
    assert report["event_fusion"] == 33_152
    assert model.appearance_encoder.reid_proj.weight.shape == (256, 2048)
    assert not hasattr(model, "event_norm")
    assert model.competition_encoder is None
    assert model.rsc.condition_encoder[0].in_features == 3


@pytest.mark.parametrize("competition_enabled", [False, True])
def test_queries_shapes_and_two_independent_gates(competition_enabled):
    model = new_model(competition_enabled=competition_enabled).eval()
    with torch.no_grad():
        result = model(**_batch(4))
    assert result["gate_logits"].shape == (4, 2)
    assert torch.isfinite(result["gates"]).all()
    assert ((result["gates"] > 0) & (result["gates"] < 1)).all()
    assert not torch.allclose(result["gates"].sum(-1), torch.ones(4))


def test_film_is_identity_at_initialization_and_can_use_competition():
    rsc = QueryRSC(d_model=32)
    current = torch.randn(4, 32)
    first = torch.zeros(4, 3)
    second = torch.full((4, 3), 3.0)
    base = torch.stack([head(current) for head in rsc.base_queries], dim=1)
    torch.testing.assert_close(rsc.make_queries(current, first), base, rtol=0, atol=0)
    torch.testing.assert_close(rsc.make_queries(current, second), base, rtol=0, atol=0)
    with torch.no_grad():
        rsc.modulations[1].weight.normal_(std=0.1)
    changed = rsc.make_queries(current, second)
    torch.testing.assert_close(changed[:, 0], base[:, 0], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 1], base[:, 1])
    changed.square().mean().backward()
    assert rsc.condition_encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mask_type", ["padding", "reset"])
def test_masked_history_cannot_affect_gates(mask_type):
    model = new_model().eval()
    first = _batch(2)
    second = copy.deepcopy(first)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    first["history_mask" if mask_type == "padding" else "history_reset"] = mask
    second["history_mask" if mask_type == "padding" else "history_reset"] = mask
    for name in ("history_evidence", "history_track", "history_detection"):
        second[name][mask] = torch.randn_like(second[name][mask]) * 100
    with torch.no_grad():
        actual = model(**first)["gate_logits"]
        perturbed = model(**second)["gate_logits"]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, perturbed, rtol=0, atol=0)


def test_missing_current_detection_cannot_supply_appearance():
    model = new_model().eval()
    first = _batch(2)
    first["current_has_detection"].fill_(False)
    second = copy.deepcopy(first)
    second["current_track"] = torch.randn_like(second["current_track"]) * 20
    second["current_detection"] = torch.randn_like(second["current_detection"]) * 20
    with torch.no_grad():
        before = model(**first)["gate_logits"]
        after = model(**second)["gate_logits"]
    assert torch.isfinite(before).all()
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_current_queries_and_history_both_change_output():
    model = new_model().eval()
    batch = _batch(3)
    with torch.no_grad():
        original = model(**batch)["gate_logits"]
        current_changed = model(**{**batch, "current_evidence":
                                  torch.randn_like(batch["current_evidence"])})["gate_logits"]
        past_changed = model(**{**batch, "history_evidence":
                               torch.randn_like(batch["history_evidence"])})["gate_logits"]
    assert not torch.allclose(original, current_changed)
    assert not torch.allclose(original, past_changed)


def test_bilinear_score_matches_explicit_formula_and_has_gradients():
    head = BilinearGate(8).double()
    query = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    history = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    normalized_q, normalized_h = head.query_norm(query), head.history_norm(history)
    reference = torch.einsum("bi,ij,bj->b", normalized_q, head.interaction, normalized_h)
    reference = reference[:, None] / (8 ** 0.5)
    reference = reference + head.linear(torch.cat([normalized_q, normalized_h], dim=-1))
    torch.testing.assert_close(head(query, history), reference)
    head(query, history).sum().backward()
    for parameter in (query, history, head.interaction):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_backward_reaches_both_heads_memory_fusion_and_film():
    model = new_model()
    logits = model(**_batch(5))["gate_logits"]
    F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    modules = [model.hce.encoder, model.hce.read_attention, model.event_fusion,
               model.appearance_encoder, model.evidence_encoder,
               *model.rsc.gate_heads, *model.rsc.modulations]
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_checkpoint_and_normal_runtime_loader_roundtrip(tmp_path):
    model = new_model().eval()
    path = tmp_path / "new.pt"
    save_new(path, model)
    loaded, payload = load_runtime_model(path, device="cpu")
    assert payload["model_config"]["architecture_variant"] == "query_film_bilinear"
    batch = _batch(3)
    with torch.no_grad():
        torch.testing.assert_close(model(**batch)["gate_logits"],
                                   loaded(**batch)["gate_logits"], rtol=0, atol=0)
    payload["model_config"]["architecture_variant"] = "legacy"
    save_checkpoint_atomic(path, payload)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_runtime_model(path)


def test_legacy_config_without_variant_remains_legacy():
    model = model_from_config({"reid_dim": 8})
    assert model.config.architecture_variant == "legacy"
    assert hasattr(model.rsc, "appearance_head")
    assert not hasattr(model.rsc, "gate_heads")


def test_unsupported_model_options_fail_clearly():
    with pytest.raises(ValueError, match="Unknown architecture_variant"):
        new_model(architecture_variant="typo")
    with pytest.raises(ValueError, match="does not support appearance_match_input"):
        new_model(appearance_match_input=True)


def test_training_options_preserve_legacy_and_choose_256_for_new():
    # Script imports its helpers as direct modules, as it does at the CLI.
    import sys
    from pathlib import Path
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from train_supervision_groups import resolve_model_options
    assert resolve_model_options(SimpleNamespace()) == ("legacy", 64)
    assert resolve_model_options(SimpleNamespace(
        architecture_variant="query_film_bilinear", appearance_proj_dim=None,
    )) == ("query_film_bilinear", 256)
    assert resolve_model_options(SimpleNamespace(
        architecture_variant="query_film_bilinear", appearance_proj_dim=128,
    )) == ("query_film_bilinear", 128)
    with pytest.raises(ValueError, match="positive"):
        resolve_model_options(SimpleNamespace(appearance_proj_dim=0))


def test_old_checkpoint_still_passes_generic_resume_contract(tmp_path):
    from shutrack.training.train import _validate_resume_contract, build_model
    model = new_model(architecture_variant="legacy")
    payload = save_new(tmp_path / "old.pt", model)
    payload["model_config"].pop("architecture_variant")
    dataset = SimpleNamespace(
        reid_dim=8, scalar_mean=np.zeros(63), scalar_std=np.ones(63),
        competition_enabled=True, descriptor=lambda: {},
    )
    _validate_resume_contract(payload, model, dataset)
    selected = build_model(dataset, {
        "architecture_variant": "query_film_bilinear", "appearance_proj_dim": 256,
    })
    assert selected.config.architecture_variant == "query_film_bilinear"


def test_weights_only_initialization_requires_same_architecture():
    from shutrack.training.train import _load_init_state_with_extension
    source = new_model()
    target = new_model()
    _load_init_state_with_extension(target, source)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="model_config"):
        _load_init_state_with_extension(new_model(architecture_variant="legacy"), source)


def test_new_model_uses_existing_tracker_adapter(tmp_path, monkeypatch):
    import shutrack.tests.test_runtime_adapter as runtime_tests

    def write_new(path, competition_enabled=True):
        model = new_model(competition_enabled=competition_enabled)
        save_new(path, model)
        return model

    monkeypatch.setattr(runtime_tests, "write_tiny_checkpoint", write_new)
    runtime_tests.test_matched_event_produces_gates_and_delegates_update(tmp_path)


def test_new_model_obeys_existing_offline_online_and_causality_checks(monkeypatch):
    import shutrack.tests.test_offline_online_equivalence as parity
    monkeypatch.setattr(parity, "make_model", new_model)
    parity.test_offline_windows_match_per_event_online_evaluation()
    parity.test_future_events_do_not_change_current_window_or_gate()
