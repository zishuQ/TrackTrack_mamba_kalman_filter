#!/usr/bin/env python3
"""Real-data numerical, gradient, augmentation and throughput review. No val runs."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
import gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"tools",ROOT,ROOT/"3. Tracker",ROOT/"agentguard/src"):
    sys.path.insert(0,str(path))
import train_supervision_groups as base
import train_window_groups as trainer
from shutrack.models.shutrack import ShutrackV2Config,ShutrackV2Model
from shutrack.training.window_replay import prepare_window_batch,replay_loss,window_index,WindowSequenceBank
from shutrack.training.observation_noise import ObservationNoise,corrupt_observations
from shutrack.training.recurrent_motion import kf_update_gated,warp_predict
from agentguard.motion.nsa_numpy import NSAKalmanFilter
from shutrack.training.loss import motion_box_diou_loss,appearance_rank_loss,appearance_column_loss,_diou_xyxy

OUT=ROOT/"outputs/agentguard/experiments/window_training_speed_audit_20261124"
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def options(data_root):
    return SimpleNamespace(
        window_frames=5,window_supervision="full",augment="none",seed=42,
        augment_pos_std=1.,augment_size_std=1.,augment_miss=.004,
        augment_burst=0.,augment_continue=.4,augment_small_area=2500.,
        augment_boundary_margin=20.,group="appearance_row_column",
        motion_harm_weight=2.,rank_alpha=.95,rank_tau=.1,rank_weight=18.,
        column_weight=50.,column_hinge=False)

def real_data():
    root=ROOT/"outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data"
    name=sorted(p.name for p in root.iterdir() if (p/"samples.npz").is_file())[0]
    data=base.SequenceData(name,"train",root,"SportsMOT")
    data.preload()
    data.open_cache()
    ck=ROOT/"outputs/agentguard/experiments/sportsmot_e100_20261123/runs/E/groups/appearance_row_column/checkpoints/epoch_100.pt"
    payload=torch.load(ck,map_location="cpu",weights_only=False)
    data.input_statistics=(np.asarray(payload["normalization_mean"]),
                           np.asarray(payload["normalization_std"]))
    taus=payload["training_config"]["taus"]
    target=base.derive_targets(data.samples,taus)
    return data,target,options(root)

def check_noise():
    rows=np.arange(4000,dtype=np.int64)
    boxes=np.broadcast_to(np.array([640,360,100,100],np.float32),(4000,16,4)).copy()
    valid=np.ones((4000,16),bool);scores=np.ones((4000,16),np.float32)*.95
    cfg=ObservationNoise(missing=0.,burst=0.)
    noisy,_,mask=corrupt_observations(boxes,scores,valid,valid,rows,
         seed=42,epoch=3,width=1280,height=720,config=cfg)
    std=(noisy-boxes).std((0,1))
    assert np.all(np.abs(std-1)<.03),std
    assert mask.all()
    parts=[]
    for chunk in np.array_split(rows,7):
        parts.append(corrupt_observations(boxes[chunk],scores[chunk],valid[chunk],valid[chunk],
             chunk,seed=42,epoch=3,width=1280,height=720,config=cfg)[0])
    assert np.array_equal(noisy,np.concatenate(parts))
    cfg=ObservationNoise(missing=1.,burst=0.,stay_missing=1.)
    _,_,keep=corrupt_observations(boxes,scores,valid,valid,rows,
         seed=42,epoch=3,width=1280,height=720,config=cfg)
    assert keep[:,0].all()
    assert not ((~keep[:,:-1]) & keep[:,1:]).any()
    assert keep[:,-1].mean()<.002
    cfg=ObservationNoise(missing=0.,burst=1.,stay_missing=0.)
    _,_,keep=corrupt_observations(boxes,scores,valid,valid,rows,
         seed=42,epoch=3,width=1280,height=720,config=cfg)
    assert keep[:,0].all() and not keep[:,1:].any()
    original=valid.copy();original[:,0]=False
    _,_,keep=corrupt_observations(boxes,scores,original,valid,rows,
         seed=42,epoch=3,width=1280,height=720,config=ObservationNoise(missing=0.))
    assert not keep[:,0].any()
    return {"gaussian_std":std.tolist(),"microbatch_invariant":True,
            "missing_and_burst":True,"profiles":{d:vars(ObservationNoise.for_dataset(d))
            for d in ("SportsMOT","MOT17","MOT20")}}

def check_math(data):
    events=data.small["current_event_index"][np.arange(32,40)]
    means=np.asarray(data.event_state_mean[events],np.float64)
    covs=np.asarray(data.event_state_cov[events],np.float64).reshape(-1,8,8)
    z=np.asarray(data.event_det_measurement[events],np.float64)
    scores=np.asarray(data.event_det_score[events],np.float64)
    warps=np.asarray(data.event_warp_next[events],np.float64).reshape(-1,2,3)
    kf=NSAKalmanFilter()
    worst=0.
    for gate in (0.,.25,1.):
        mt,pt=kf_update_gated(torch.tensor(means),torch.tensor(covs),
                            torch.tensor(z),torch.tensor(scores),
                            torch.full((len(events),),gate,dtype=torch.float64))
        for i in range(len(events)):
            full,pfull=kf.update(means[i],covs[i],z[i],scores[i])
            wanted=means[i]+gate*(full-means[i])
            p=(1-gate)*covs[i]+gate*pfull
            worst=max(worst,float(np.max(np.abs(mt[i].numpy()-wanted))),
                      float(np.max(np.abs(pt[i].numpy()-p))))
        next_mean,next_cov=warp_predict(mt,pt,torch.tensor(warps))
        for i in range(len(events)):
            moved,pc=kf.apply_warp(mt[i].numpy(),pt[i].numpy(),warps[i])
            wanted,pwanted=kf.predict(moved,pc)
            worst=max(worst,float(np.max(np.abs(next_mean[i].numpy()-wanted))),
                      float(np.max(np.abs(next_cov[i].numpy()-pwanted))))
        assert worst<1e-5,worst
    # Differentiate through the KF gain and a later prediction.
    m=torch.tensor(means,requires_grad=True)
    p=torch.tensor(covs,requires_grad=True)
    gate=torch.full((len(events),),.5,dtype=torch.float64,requires_grad=True)
    mu,pu=kf_update_gated(m,p,torch.tensor(z),torch.tensor(scores),gate)
    mu,pu=warp_predict(mu,pu,torch.tensor(warps))
    grads=torch.autograd.grad(mu.square().mean()+pu.square().mean(),(m,p,gate))
    assert all(torch.isfinite(g).all() for g in grads)
    return {"numpy_max_abs_error":worst,"finite_state_cov_gate_gradients":True}

def model_for(data,device,dropout=0.):
    torch.manual_seed(42)
    return ShutrackV2Model(ShutrackV2Config(reid_dim=data.event_track.shape[1],
         appearance_proj_dim=256,architecture_variant="query_film_bilinear",
         dropout=dropout)).to(device).train()

def checks(data,target,args,device):
    events,_,valid=window_index(data,5)
    rows=np.flatnonzero(valid.all(axis=1))[:4]
    # Exact agreement of vectorized chain construction with the old scalar form.
    for row in np.concatenate((np.arange(12),rows)):
        old_events,_,_=trainer.chain_for_row(data,int(row),5)
        assert old_events==events[row,valid[row]].tolist()
    model=model_for(data,device)
    pack=prepare_window_batch(base,data,target,rows,1,args)
    loss,parts,_=replay_loss(base,model,pack,args,device,debug=True)
    assert torch.isfinite(loss)
    bank=WindowSequenceBank(data,target,device)
    gpu_pack=prepare_window_batch(base,data,target,rows,1,args,bank=bank)
    bank_loss,bank_parts,_=replay_loss(base,model,gpu_pack,args,device,debug=True)
    torch.testing.assert_close(loss,bank_loss,rtol=1e-5,atol=1e-5)
    for left,right in zip(parts["debug_steps"],bank_parts["debug_steps"]):
        for key in ("mean","cov","evidence","feature","logits"):
            torch.testing.assert_close(left[key],right[key],rtol=1e-5,atol=1e-4)
    del bank_loss,bank_parts,gpu_pack,bank
    for i,row in enumerate(rows):
        single=prepare_window_batch(base,data,target,np.array([row]),1,args)
        _,sp,_=replay_loss(base,model,single,args,device,debug=True)
        for batch_step,single_step in zip(parts["debug_steps"],sp["debug_steps"]):
            for key in ("mean","cov","evidence","feature"):
                torch.testing.assert_close(batch_step[key][i],single_step[key][0],
                                           rtol=3e-4,atol=3e-4)
    # Future loss reaches an earlier gate, without direct supervision of that gate.
    first=parts["debug_steps"][0]["logits"]
    first.retain_grad()
    last=parts["debug_steps"][-1]["mean"]
    later_loss=last[:,:4].square().mean()
    later_loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert float(first.grad[:,0].abs().sum())>0, {"valid":pack["valid"].tolist(),"first_logits":first.detach().cpu().tolist(),"gradient":first.grad.detach().cpu().tolist()}
    model.zero_grad(set_to_none=True)
    # One-step evidence and objective match the unchanged original trainer.
    args.window_frames=1
    p1=prepare_window_batch(base,data,target,rows,1,args)
    l1,details,_=replay_loss(base,model,p1,args,device,debug=True)
    raw=base.batch_tensors(data,rows,target,args.group,compact_reid=True)
    old_batch=base.to_device(raw,torch.device(device))
    expected=old_batch["current_evidence"]
    actual=details["debug_steps"][0]["evidence"]
    evidence_error=float((actual-expected).abs().max())
    assert evidence_error<1e-3,evidence_error
    outputs=model(**base.model_inputs(old_batch))
    ml,_=motion_box_diou_loss(outputs,old_batch,harm_weight=args.motion_harm_weight)
    al,_=appearance_rank_loss(outputs,old_batch,alpha=args.rank_alpha,tau=args.rank_tau,
                             rank_weight=args.rank_weight)
    al=al+args.column_weight*appearance_column_loss(outputs,old_batch,
          alpha=args.rank_alpha,tau=args.rank_tau,tensor_statistics=True)
    objective_error=float((l1-(ml+al)).detach().abs())
    assert objective_error<.01,objective_error
    args.window_frames=5
    args.window_supervision="anchor"
    warm=prepare_window_batch(base,data,target,rows,1,args)
    _,warm_details,_=replay_loss(base,model,warm,args,device,debug=True)
    warm_error=float((warm_details["debug_steps"][-1]["evidence"]-expected).abs().max())
    assert warm_error<1e-3,warm_error
    args.window_supervision="full"
    # Missing current observations must reach model masks as well as evidence.
    args.augment="mottrack";args.augment_miss=1.;args.augment_continue=1.
    augmented=prepare_window_batch(base,data,target,rows,2,args)
    _,dp,_=replay_loss(base,model,augmented,args,device,debug=True)
    assert not dp["debug_steps"][-1]["has"].any()
    assert torch.all(dp["debug_steps"][-1]["evidence"][:,-1]==0)
    args.augment="none";args.augment_miss=.004;args.augment_continue=.4
    del parts,details,dp,model
    gc.collect()
    if device=="cuda": torch.cuda.empty_cache()
    return {"chain_binding":True,"batched_vs_single_replay":True,
            "future_gate_gradient":True,"one_step_evidence_max_abs":evidence_error,
            "one_step_objective_abs":objective_error,"missing_model_mask":True,"gpu_bank_equivalent":True,"noise_only_clean_history_evidence_abs":warm_error}

def benchmark(data,target,args,steps,batch):
    old=load_module("window_before_review",OUT/"train_window_groups_before.py")
    old_math=load_module("window_math_before_review",OUT/"recurrent_motion_before.py")
    old.kf_update_gated=old_math.kf_update_gated;old.warp_predict=old_math.warp_predict
    rows=np.arange(128,128+batch,dtype=np.int64)
    report={}
    for name,fn in (("before",None),("packed_G1",replay_loss),
                    ("packed_G2",replay_loss),("packed_G3",replay_loss)):
        args.window_supervision="anchor" if name=="packed_G2" else "full"
        args.augment="mottrack" if name in ("packed_G2","packed_G3") else "none"
        model=model_for(data,"cuda",dropout=.05)
        bank=WindowSequenceBank(data,target,"cuda") if fn is not None else None
        times=[];prepare_times=[]
        for it in range(steps+2):
            torch.cuda.synchronize()
            t=time.perf_counter()
            if fn is None:
                loss,_,_=old.run_microbatch(model,data,target,rows,2,args,
                                           torch.device("cuda"),print,{},False)
                prepared_time=None
            else:
                prep=time.perf_counter()
                prepared=prepare_window_batch(base,data,target,rows,2,args,pin_memory=True,bank=bank)
                prepared_time=time.perf_counter()-prep
                loss,_,_=fn(base,model,prepared,args,torch.device("cuda"))
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if it>=2:
                times.append(elapsed)
                if prepared_time is not None: prepare_times.append(prepared_time)
            del loss
        report[name]={"batch":batch,"median_step_seconds":float(np.median(times)),
                      "mean_step_seconds":float(np.mean(times)),
                      "cpu_prepare_seconds":float(np.mean(prepare_times)) if prepare_times else None,
                      "estimated_321_batch_epoch_seconds":float(np.mean(times))*321}
        del model,bank
        gc.collect();torch.cuda.empty_cache()
    args.window_supervision="full";args.augment="none"
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--batch",type=int,default=1024)
    opt=parser.parse_args()
    torch.set_num_threads(4)
    data,target,args=real_data()
    device="cuda" if torch.cuda.is_available() else "cpu"
    report={"noise":check_noise(),"math":check_math(data),
            "real_data":checks(data,target,args,device)}
    if opt.benchmark: report["benchmark"]=benchmark(data,target,args,opt.steps,opt.batch)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"review_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    data.close_cache();data.samples.close()
if __name__=="__main__": main()


