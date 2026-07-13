from __future__ import annotations

import torch

from agentguard.models.iwg_tsrm import IWGTSRM


def make_joint_batch(batch_size: int = 2, length: int = 8, reid_dim: int = 16) -> dict:
    generator = torch.Generator().manual_seed(19)
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    padding[0, :2] = True
    has_detection = ~padding
    has_detection[0, 4] = False
    reset = torch.zeros_like(padding)
    reset[0, 2] = True
    if batch_size > 1:
        reset[1, 0] = True
    label = has_detection.clone()
    valid_motion = label.clone()
    valid_appearance = label.clone()
    motion = torch.rand((batch_size, length), generator=generator)
    appearance = torch.rand((batch_size, length), generator=generator)
    motion_conf = torch.rand((batch_size, length), generator=generator)
    appearance_conf = torch.rand((batch_size, length), generator=generator)
    base_target = torch.stack([motion, appearance], dim=-1)
    final_target = torch.clamp(base_target + 0.1, 0.0, 1.0)
    cue_target = torch.stack(
        [motion_conf, appearance_conf, torch.maximum(motion_conf, appearance_conf)],
        dim=-1,
    )
    risk_target = torch.stack(
        [
            1.0 - motion,
            1.0 - appearance,
            1.0 - torch.minimum(motion_conf, appearance_conf),
            torch.abs(motion - appearance),
        ],
        dim=-1,
    )
    policy = torch.rand((batch_size, length, 5), generator=generator)
    policy = policy / policy.sum(dim=-1, keepdim=True)
    return {
        "track_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "det_feats": torch.randn(batch_size, length, reid_dim, generator=generator),
        "scalar_feats": torch.randn(batch_size, length, 63, generator=generator),
        "padding_mask": padding,
        "mask": padding,
        "has_detection_mask": has_detection,
        "reset_mask": reset,
        "label_mask": label,
        "valid_motion": valid_motion,
        "valid_appearance": valid_appearance,
        "base_gate_target": base_target,
        "final_gate_target": final_target,
        "policy_soft_target": policy,
        "cue_target": cue_target,
        "risk_target": risk_target,
        "sample_weight": label.float(),
    }


def test_tsrm_zero_init_bounded_correction_and_unmatched_sentinel():
    model = IWGTSRM(reid_dim=16, delta_max=0.2).eval()
    batch = make_joint_batch()
    with torch.no_grad():
        outputs = model(batch)

    torch.testing.assert_close(outputs["final_gate"], outputs["base_gate"])
    assert outputs["temporal_gate_correction"].abs().max().item() == 0.0
    assert outputs["temporal_gate_delta"].abs().max().item() <= 0.2
    unmatched = ~batch["has_detection_mask"]
    torch.testing.assert_close(
        outputs["policy_probs"][unmatched],
        outputs["policy_probs"].new_tensor([0.0, 0.0, 0.0, 1.0, 0.0]).expand(unmatched.sum(), -1),
    )
    torch.testing.assert_close(outputs["base_gate"][unmatched], torch.zeros_like(outputs["base_gate"][unmatched]))
    torch.testing.assert_close(outputs["final_gate"][unmatched], torch.zeros_like(outputs["final_gate"][unmatched]))


def test_iwg_tsrm_is_strictly_causal():
    model = IWGTSRM(reid_dim=16).eval()
    batch = make_joint_batch(batch_size=1)
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    changed["track_feats"][:, 5:] += 50.0
    changed["det_feats"][:, 5:] -= 50.0
    changed["scalar_feats"][:, 5:] *= -20.0
    with torch.no_grad():
        original = model(batch)
        perturbed = model(changed)
    for key in (
        "base_gate",
        "event_embedding",
        "memory",
        "temporal_gate_correction",
        "final_gate",
        "predicted_next_scalar_delta",
    ):
        torch.testing.assert_close(
            original[key][:, :5], perturbed[key][:, :5], atol=1e-6, rtol=0.0
        )
