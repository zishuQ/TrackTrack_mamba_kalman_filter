"""DIoU in error-pool diagnostics must match the actual training objective."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"tools"))
from diagnose_error_pool import diou
from shutrack.training.loss import _diou_xyxy

def test_diagnostic_matches_training_for_overlapping_and_disjoint_boxes():
    rng=np.random.default_rng(42)
    a=rng.uniform(-50,50,(128,4))
    b=rng.uniform(-50,50,(128,4))
    a[:,2:]=a[:,:2]+rng.uniform(1,30,(128,2))
    b[:,2:]=b[:,:2]+rng.uniform(1,30,(128,2))
    expected=_diou_xyxy(torch.from_numpy(a),torch.from_numpy(b)).numpy()
    actual=np.array([diou(x,y) for x,y in zip(a,b)])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

def test_diou_uses_joint_enclosing_box_not_sum_of_individual_diagonals():
    a=np.array([0.,0.,10.,10.])
    b=np.array([20.,0.,30.,10.])
    assert abs(diou(a,b)-(-.4))<1e-12
