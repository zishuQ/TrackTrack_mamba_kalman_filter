from __future__ import annotations

import torch

from agentguard.models.iwg_tsrm import IWGTSRM
from agentguard.training.loss_iwg_tsrm import compute_iwg_tsrm_loss


def make_joint_batch(batch_size: int = 2, length: int = 8, reid_dim: int = 16) -> dict:
    generator = torch.Generator().manual_seed(19)
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    padding[0, :2] = True
    has_detection = ~padding
    has_detection[0, 4] = False
    reset = torch.zeros_like(padding)
    reset[0, 2] = True
    reset[1, 0] = True
    label = has_detection.clone()
    motion = torch.rand((batch_size, length), generator=generator)
    appearance = torch.rand((batch_size, length), generator=generator)
    motion_conf = torch.rand((batch_size, length), generator=generator)
    appearance_conf = torch.rand((batch_size, length), generator=generator)
    policy = torch.rand((batch_size, length, 5), generator=generator)
    policy = policy / policy.sum(dim=-1, keepdim=True)
    return {
        "track_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "det_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "scalar_feats": torch.randn(batch_size, length, 63, generator=generator),
        "padding_mask": padding,
        "has_detection_mask": has_detection,
        "reset_mask": reset,
        "label_mask": label,
        "valid_motion": label.clone(),
        "valid_appearance": label.clone(),
        "base_gate_target": torch.stack([motion, appearance], dim=-1),
        "final_gate_target": torch.clamp(torch.stack([motion, appearance], dim=-1) + 0.1, 0.0, 1.0),
        "policy_soft_target": policy,
        "cue_target": torch.stack(
            [motion_conf, appearance_conf, torch.maximum(motion_conf, appearance_conf)], dim=-1
        ),
        "risk_target": torch.stack(
            [1.0 - motion, 1.0 - appearance, 1.0 - torch.minimum(motion_conf, appearance_conf), torch.abs(motion - appearance)],
            dim=-1,
        ),
        "sample_weight": label.float(),
    }


def _assert_nonzero_finite_gradients(module: torch.nn.Module, name: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients), name
    assert all(torch.isfinite(gradient).all() for gradient in gradients), name
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), name


def test_production_joint_loss_is_finite_and_trains_every_branch():
    model = IWGTSRM(reid_dim=16).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_joint_batch()

    # Step one moves the strictly zero-initialized delta head. At exact zero,
    # strength has a mathematically zero derivative; step two audits all heads.
    outputs = model(batch)
    loss, components = compute_iwg_tsrm_loss(outputs, batch, training_mode="joint")
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    outputs = model(batch)
    loss, _ = compute_iwg_tsrm_loss(outputs, batch, training_mode="joint")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    groups = {
        "encoder": model.iwg.encoder,
        "transformer": model.iwg.transformer,
        "policy": model.iwg.policy_head,
        "iwg_residual": model.iwg.iwg_gate_residual_head,
        "cue": model.iwg.cue_head,
        "risk": model.iwg.risk_head,
        "tcn": model.tsrm.tcn_blocks,
        "gru": model.tsrm.gru,
        "fusion": model.tsrm.fusion_head,
        "delta": model.tsrm.delta_head,
        "strength": model.tsrm.strength_head,
        "dynamics": model.tsrm.dynamics_head,
    }
    for name, module in groups.items():
        _assert_nonzero_finite_gradients(module, name)


def test_joint_loss_masks_padding_unmatched_and_empty_supervision():
    model = IWGTSRM(reid_dim=16).train()
    batch = make_joint_batch()
    batch["label_mask"].zero_()
    batch["valid_motion"].zero_()
    batch["valid_appearance"].zero_()
    outputs = model(batch)
    loss, components = compute_iwg_tsrm_loss(outputs, batch, training_mode="joint")

    assert torch.isfinite(loss)
    assert components["base_gate_loss"].item() == 0.0
    assert components["final_gate_loss"].item() == 0.0
    assert components["dynamics_loss"].item() > 0.0


def test_base_training_mode_does_not_require_tsrm_outputs():
    model = IWGTSRM(reid_dim=16).train()
    batch = make_joint_batch()
    outputs = model.iwg.forward_sequence(
        batch["track_feats"], batch["det_feats"], batch["scalar_feats"], batch["padding_mask"]
    )
    outputs = model.apply_unmatched_sentinel(outputs, batch["has_detection_mask"])
    loss, components = compute_iwg_tsrm_loss(outputs, batch, training_mode="base")
    assert torch.isfinite(loss)
    assert components["final_gate_loss"].item() == 0.0
    assert components["dynamics_loss"].item() == 0.0
