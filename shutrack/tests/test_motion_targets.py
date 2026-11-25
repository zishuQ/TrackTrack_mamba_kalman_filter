"""Oracle targets, stable saturated gradients and process-local trainer isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
from generate_motion_gate_targets import generate_sequence
from train_motion_gate_targets import target_training
import train_supervision_groups as trainer
from shutrack.training.loss import _diou_xyxy
from shutrack.training.motion_targets import (
    generate_targets, motion_target_bce, numpy_diou, source_fingerprint,
)


def arrays():
    a=np.tile([0.,0.,10.,10.],(4,1)).astype(np.float32)
    b=a+np.array([20.,0.,20.,0.],dtype=np.float32)
    gt=np.stack([a[0],b[1],a[2]+.35*(b[2]-a[2]),a[3]])
    return {"frame":np.arange(4),"current_event_index":np.arange(4),
            "pred_box_g0":a,"pred_box_g1":b,"gt_box_t1":gt,
            "motion_gt_valid":np.array([True,True,True,False])}


def test_targets_cover_both_endpoints_and_partial_update():
    x=arrays()
    target=generate_targets(x["pred_box_g0"],x["pred_box_g1"],x["gt_box_t1"],x["motion_gt_valid"])
    np.testing.assert_allclose(target,[0.,1.,.35,1.],rtol=0,atol=1e-7)
    other=generate_targets(x["pred_box_g0"],x["pred_box_g1"],x["gt_box_t1"],x["motion_gt_valid"],block_size=1)
    np.testing.assert_array_equal(target,other)


def test_flat_curves_and_missing_gt_preserve_full_write():
    x=arrays()
    x["gt_box_t1"][-1]=np.nan
    x["pred_box_g1"]=x["pred_box_g0"].copy()
    target=generate_targets(x["pred_box_g0"],x["pred_box_g1"],x["gt_box_t1"],x["motion_gt_valid"])
    np.testing.assert_array_equal(target,np.ones(4,dtype=np.float32))


def test_numpy_diou_matches_actual_training_formula():
    rng=np.random.default_rng(7)
    a=rng.uniform(0,100,(64,4))
    b=rng.uniform(0,100,(64,4))
    a[:,2:]+=a[:,:2]
    b[:,2:]+=b[:,:2]
    expected=_diou_xyxy(torch.from_numpy(a),torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(numpy_diou(a,b),expected,rtol=0,atol=1e-12)


def test_saturated_bad_gate_has_gradient_and_invalid_appearance_channels_are_zero():
    logits=torch.tensor([[18.,2.],[18.,2.]],requires_grad=True)
    batch={"motion_gate_target":torch.tensor([.2,.9]),
           "motion_gt_valid":torch.tensor([True,False])}
    loss,_=motion_target_bce({"gate_logits":logits},batch)
    loss.backward()
    assert logits.grad[0,0].item()==pytest.approx(.8)
    assert logits.grad[1,0].item()==0
    assert logits.grad[:,1].abs().sum().item()==0


def test_all_invalid_motion_produces_connected_zero_loss():
    logits=torch.zeros(2,2,requires_grad=True)
    loss,_=motion_target_bce({"gate_logits":logits},{
        "motion_gate_target":torch.ones(2),"motion_gt_valid":torch.zeros(2,dtype=torch.bool)})
    loss.backward()
    assert loss.item()==0
    assert logits.grad.abs().sum().item()==0


def test_labels_are_row_bound_and_generator_reuses_identical_sidecar(tmp_path):
    source=tmp_path/"source"
    destination=tmp_path/"targets"
    source.mkdir()
    x=arrays()
    np.savez(source/"samples.npz",**x)
    first=generate_sequence(source,destination)
    second=generate_sequence(source,destination)
    assert first==second
    assert (destination/"motion_gate_target.npy").stat().st_size<1024
    old_fingerprint=source_fingerprint(x)
    x["frame"]=x["frame"][::-1].copy()
    assert source_fingerprint(x)!=old_fingerprint
    np.savez(source/"samples.npz",**x)
    with pytest.raises(ValueError,match="different source rows"):
        generate_sequence(source,destination)


def test_corrupt_target_file_is_detected(tmp_path):
    source=tmp_path/"source"
    destination=tmp_path/"targets"
    source.mkdir()
    np.savez(source/"samples.npz",**arrays())
    generate_sequence(source,destination)
    np.save(destination/"motion_gate_target.npy",np.zeros(4,dtype=np.float32))
    with pytest.raises(ValueError,match="checksum"):
        generate_sequence(source,destination)


def test_scoped_training_adapter_restores_original_entry_on_exception(tmp_path):
    names=("SequenceData","batch_tensors","PAIRED_TENSOR_KEYS","motion_box_diou_loss",
           "plan_phase","build_checkpoint_payload","save_checkpoint_atomic")
    old={name:getattr(trainer,name) for name in names}
    with pytest.raises(RuntimeError,match="deliberate"):
        with target_training(tmp_path/"targets",tmp_path):
            assert trainer.motion_box_diou_loss is not old["motion_box_diou_loss"]
            assert "motion_gate_target" in trainer.PAIRED_TENSOR_KEYS
            raise RuntimeError("deliberate")
    for name,value in old.items():
        assert getattr(trainer,name) is value


def test_target_loader_reads_once_and_keeps_source_inputs_unchanged(tmp_path):
    source=tmp_path/"data"/"sequence"
    source.mkdir(parents=True)
    x=arrays()
    np.savez(source/"samples.npz",**x)
    np.save(source/"event_evidence.npy",np.zeros((4,13),dtype=np.float32))
    np.save(source/"event_track.npy",np.zeros((4,8),dtype=np.float32))
    np.save(source/"event_detection_index.npy",np.arange(4))
    np.save(source/"event_has_detection.npy",np.ones(4,dtype=bool))
    targets=tmp_path/"targets"
    generate_sequence(source,targets/"sequence")
    with target_training(targets,tmp_path):
        data=trainer.SequenceData("sequence","train",tmp_path/"data","SportsMOT")
        np.testing.assert_allclose(data.motion_gate_target,[0.,1.,.35,1.],rtol=0,atol=1e-7)
        assert data.event_track.dtype==np.float32
