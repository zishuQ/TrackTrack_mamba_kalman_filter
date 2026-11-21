"""GT-based motion harm weighting: compatibility and gradient direction."""
import sys
from pathlib import Path
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from shutrack.training.loss import motion_box_diou_loss, _diou_xyxy


def batch():
    gt = torch.tensor([[0., 0., 10., 10.]] * 3)
    shifted = torch.tensor([8., 0., 18., 10.])
    b0 = gt.clone()
    b0[0] = shifted
    b1 = gt.clone()
    b1[1] = shifted
    return {"pred_box_g0": b0, "pred_box_g1": b1, "gt_box_t1": gt,
            "motion_gt_valid": torch.tensor([True, True, False])}


def test_factor_one_preserves_loss_and_parameter_gradients_exactly():
    torch.manual_seed(2)
    model = torch.nn.Linear(4, 2)
    features = torch.randn(3, 4)
    data = batch()
    logits = model(features)
    loss, _ = motion_box_diou_loss({"gate_logits": logits}, data, harm_weight=1.)
    actual = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
    gate = logits.sigmoid()[:, 0]
    boxes = data["pred_box_g0"] + gate[:, None]*(data["pred_box_g1"]-data["pred_box_g0"])
    mask = data["motion_gt_valid"].float()
    reference = ((1-_diou_xyxy(boxes, data["gt_box_t1"]))*mask).sum()/mask.sum().clamp_min(1)
    expected = torch.autograd.grad(reference, tuple(model.parameters()))
    assert torch.equal(loss, reference)
    for a,b in zip(actual, expected):
        assert torch.equal(a,b)


def test_weighting_emphasizes_harm_but_keeps_correct_gradient_directions():
    data = batch()
    logits = torch.zeros(3, 2, requires_grad=True)
    loss, stats = motion_box_diou_loss({"gate_logits": logits}, data, harm_weight=8.)
    grad, = torch.autograd.grad(loss, logits)
    # Row0 benefits from more writing; row1 benefits from less.
    assert grad[0, 0] < 0 < grad[1, 0]
    assert torch.allclose(grad[1, 0], -8 * grad[0, 0])
    assert torch.equal(grad[:, 1], torch.zeros(3))  # No direct appearance-logit gradient.
    assert torch.equal(grad[2], torch.zeros(2))
    assert stats["harm_fraction"].item() == .5
    assert stats["harm_weight_share"].item() == pytest.approx(8/9)
    assert torch.isfinite(loss)


def test_empty_mask_returns_zero_with_zero_gradients():
    data = batch()
    data["motion_gt_valid"][:] = False
    logits = torch.zeros(3, 2, requires_grad=True)
    loss, stats = motion_box_diou_loss({"gate_logits": logits}, data, harm_weight=8.)
    grad, = torch.autograd.grad(loss, logits)
    assert loss.item() == 0
    assert torch.equal(grad, torch.zeros_like(grad))
    assert stats["harm_fraction"] == stats["harm_weight_share"] == 0


@pytest.mark.parametrize("factor", [0., .5, float("nan"), float("inf")])
def test_bad_factors_rejected(factor):
    with pytest.raises(ValueError):
        motion_box_diou_loss({"gate_logits": torch.zeros(3,2)}, batch(), harm_weight=factor)
