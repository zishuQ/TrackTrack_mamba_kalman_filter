"""Packed, differentiable window replay for SHUTrack.

The association candidate context is held at the recorded replay values. KF
state, observation masks, history evidence and EMA templates are rebuilt within
the window. No GT enters model inputs.
"""
from __future__ import annotations
import numpy as np
import torch
from shutrack.schema import EVIDENCE_SCALAR_INDICES
from shutrack.training.loss import appearance_rank_loss, appearance_column_loss, _diou_xyxy
from shutrack.training.observation_noise import ObservationNoise, corrupt_observations, _uniform
from shutrack.training.recurrent_motion import (
    evidence_from_state, kf_update, warp_predict, ema_update,
)

WINDOW_IMPLEMENTATION = "packed_replay_v2"

def window_index(data, length):
    cache = getattr(data, "_window_indices_cache", None)
    if cache is None:
        cache = data._window_indices_cache = {}
    if length in cache:
        return cache[length]
    candidates = np.column_stack((data.small["window_indices"],
                                  data.small["current_event_index"]))
    safe = np.maximum(candidates, 0)
    frames = np.asarray(data.event_frame[safe])
    targets = np.asarray(data.event_target_row[safe])
    allowed = (candidates >= 0) & (targets >= 0)
    suffix = np.zeros_like(allowed)
    suffix[:, -1] = allowed[:, -1]
    for t in range(candidates.shape[1]-2, -1, -1):
        warp = np.asarray(data.event_warp_next[safe[:, t]]).reshape(-1, 2, 3)
        link = np.any(warp[..., :2] != 0, axis=(1, 2))
        suffix[:, t] = (allowed[:, t] & suffix[:, t+1]
                        & (frames[:, t+1] == frames[:, t]+1) & link)
    lengths = np.minimum(suffix.sum(axis=1), length)
    if np.any(lengths == 0):
        raise ValueError("A training anchor has no replay state/sample binding")
    # Right-align: the anchor is always the final step, irrespective of warmup length.
    columns = np.arange(candidates.shape[1]-length, candidates.shape[1])
    if length > candidates.shape[1]:
        raise ValueError("window exceeds recorded context capacity")
    events = safe[:, columns].copy()
    valid = np.arange(length)[None] >= (length-lengths[:, None])
    events = np.where(valid, events, safe[:, -1, None])
    rows = np.asarray(data.event_target_row[events], np.int64)
    cache[length] = (events, rows, valid)
    return cache[length]

def _tensor_tree(value, pin):
    if isinstance(value, dict):
        return {k: _tensor_tree(v, pin) for k,v in value.items()}
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        return tensor.pin_memory() if pin else tensor
    return value

def _to_device(value, device):
    if isinstance(value, dict):
        return {k:_to_device(v, device) for k,v in value.items()}
    if torch.is_tensor(value):
        tensor = value.to(device, non_blocking=value.is_pinned())
        return tensor.float() if tensor.dtype == torch.float16 else tensor
    return value

def noise_config(args):
    return ObservationNoise(
        position_std=args.augment_pos_std, size_std=args.augment_size_std,
        missing=args.augment_miss, burst=args.augment_burst,
        stay_missing=args.augment_continue, small_area=args.augment_small_area,
        boundary_margin=args.augment_boundary_margin,
    )

def prepare_window_batch(base, data, target, rows, epoch, args, *, pin_memory=False, bank=None):
    rows = np.asarray(rows, np.int64)
    all_events, all_rows, all_valid = window_index(data, args.window_frames)
    events, step_rows, valid = all_events[rows], all_rows[rows], all_valid[rows]
    n, length = valid.shape
    if not hasattr(data, "_img_size"):
        import json
        summary=json.loads((data.dir/"summary.json").read_text())
        data._img_size=(float(summary["img_width"]),float(summary["img_height"]))
    width,height=data._img_size
    measurement=np.asarray(data.event_det_measurement[events],np.float32).copy()
    scores=np.asarray(data.event_det_score[events],np.float32).copy()
    has=np.asarray(data.event_has[events],bool) & valid
    if args.augment == "mottrack":
        measurement,scores,has=corrupt_observations(
            measurement,scores,has,valid,rows,seed=args.seed,epoch=epoch,
            width=width,height=height,config=noise_config(args))
    simulated_mask=np.zeros((n,length),bool)
    fallback_mask=np.zeros((n,length),bool)
    if args.augment == "error_pool":
        pool=np.asarray(args.error_pool_array,np.float32)
        event_rows=np.asarray(data.event_target_row[events],np.int64)
        safe_rows=np.maximum(event_rows,0)
        gt0=np.asarray(data.gt_box_t0[safe_rows],np.float64)
        gt_valid=np.asarray(data.gt_valid_t0[safe_rows],bool)&(event_rows>=0)
        original_has=np.asarray(data.event_has[events],bool)
        eligible=valid&original_has&gt_valid
        fallback=valid&original_has&~gt_valid
        u=_uniform(rows,args.seed,epoch,length,91)
        picks=np.minimum((u*len(pool)).astype(np.int64),len(pool)-1)
        residual=pool[picks].astype(np.float64)
        sim=gt0.copy()
        sim[...,0]+=residual[...,0]*np.maximum(gt0[...,2],1e-9)
        sim[...,1]+=residual[...,1]*np.maximum(gt0[...,3],1e-9)
        sim[...,2]*=np.exp(residual[...,2])
        sim[...,3]*=np.exp(residual[...,3])
        sim[...,2:]=np.maximum(sim[...,2:],1.0)
        measurement=np.where(eligible[...,None],sim,measurement).astype(np.float32)
        simulated_mask=eligible
        fallback_mask=fallback
    frames=np.asarray(data.event_frame[events],np.int64)
    gap=np.asarray(data.event_gap[events],np.int64)
    first=np.argmax(valid,axis=1)
    first_events=events[np.arange(n),first]
    detection_ids=np.asarray(data.event_det[events],np.int64)
    det_feature=None
    if bank is None:
        det_feature=np.zeros((n,length,data.event_track.shape[1]),np.float32)
        hit=(detection_ids>=0)&has
        det_feature[hit]=data.cache.features[detection_ids[hit]]
    # Only full sequence supervision needs K model calls. The noise-only
    # experiment warms the state with fixed replay writes and calls the model once.
    input_rows=step_rows if args.window_supervision == "full" else rows[:,None]
    model_length=input_rows.shape[1]
    if bank is None:
        raw=base.batch_tensors(data,input_rows.reshape(-1),target,"window_inputs",
                               compact_reid=True)
        inputs={k:v.reshape((n,model_length)+v.shape[1:])
                for k,v in base.model_inputs(raw).items()}
        app=base.batch_tensors(data,rows,target,args.group,compact_reid=True)
        app_keys=("rank_template0","rank_detection","rank_positive","rank_competitor",
                  "rank_score","rank_valid","appearance_target","appearance_valid",
                  "column_positive","column_valid","column_offset","column_rival","column_score")
        app={k:app[k] for k in app_keys if k in app}
    else:
        inputs=app=None
    mean,std=data.input_statistics
    cols=list(EVIDENCE_SCALAR_INDICES)
    pack={
        "inputs":inputs,"app":app,"events":events,
        "bank":bank,"input_rows":input_rows,"anchor_rows":rows,
        "detection_ids":detection_ids,"first_events":first_events,
        "history_events":data.small["window_indices"][input_rows],
        "valid":valid,"measurement":measurement,"score":scores,"has":has,
        "frame":frames,"stored_gap":gap,
        "initial_mean":np.asarray(data.event_state_mean[first_events],np.float32),
        "initial_cov":np.asarray(data.event_state_cov[first_events],np.float32).reshape(n,8,8),
        "initial_feature":(np.asarray(data.event_track[first_events],np.float32) if bank is None else None),
        "initial_last_frame":frames[np.arange(n),first]-gap[np.arange(n),first],
        "detection_feature":det_feature,
        "warp":np.asarray(data.event_warp_next[events],np.float32).reshape(n,length,2,3),
        "gt":data.small["gt_box_t1"][step_rows],
        "gt_valid":data.small["motion_gt_valid"][step_rows]&valid,
        "norm_mean":np.asarray(mean[cols],np.float32),
        "norm_std":np.asarray(std[cols],np.float32),
        "width":width,"height":height,
        "transitions":int(valid.sum()) if args.window_supervision=="full" else n,
        "missing":int((valid&~has).sum()),"count":n,
        "simulated_mask":simulated_mask,"fallback_mask":fallback_mask,
        "simulated":int(simulated_mask.sum()),"fallback":int(fallback_mask.sum()),
    }
    return _tensor_tree(pack,pin_memory)

def _box(mean):
    return torch.cat((mean[:,:2]-mean[:,2:4]/2,
                      mean[:,:2]+mean[:,2:4]/2),dim=-1)

def _next_mean(mean,warp):
    pairs=mean.reshape(-1,4,2)
    moved=pairs @ warp[:,:,:2].transpose(-1,-2)
    shift=torch.cat((warp[:,:,2],torch.zeros_like(mean[:,2:])),dim=-1)
    moved=moved.reshape(-1,8)+shift
    return torch.cat((moved[:,:4]+moved[:,4:],moved[:,4:]),dim=-1)

def replay_loss(base,model,prepared,args,device,grad_probe=None,probe_allowed=False,
                *,debug=False):
    pack=_to_device(prepared,device)
    if pack["bank"] is not None:
        bank=pack["bank"]
        pack["inputs"]=bank.inputs(pack["input_rows"],pack["history_events"])
        pack["app"]=bank.appearance(pack["anchor_rows"])
        pack["initial_feature"]=bank.track[pack["first_events"]]
        ids=pack["detection_ids"]
        pack["detection_feature"]=bank.features[ids.clamp_min(0)].masked_fill(
            ((ids<0)|~pack["has"])[:,:,None],0)
    mean,cov,feature=pack["initial_mean"],pack["initial_cov"],pack["initial_feature"]
    last_frame=pack["initial_last_frame"]
    length=pack["valid"].shape[1]
    motion_sum=mean.new_zeros(())
    weight_sum=mean.new_zeros(())
    gate_sum=mean.new_zeros(2)
    gate_count=mean.new_zeros(())
    history=[]
    debug_steps=[]
    anchor_outputs=anchor_track=anchor_det=None
    for position in range(length):
        active=pack["valid"][:,position]
        present=pack["has"][:,position]&active
        measurement=pack["measurement"][:,position]
        score=pack["score"][:,position]
        gap=(pack["frame"][:,position]-last_frame).clamp(0,30)
        raw=evidence_from_state(mean,cov,measurement,score,gap,present,
                                pack["width"],pack["height"])
        evidence=torch.cat(((raw[:,:12]-pack["norm_mean"])/pack["norm_std"],
                            raw[:,12:]),dim=-1)
        current_track=feature
        current_det=pack["detection_feature"][:,position]
        needs_model=(args.window_supervision=="full" or position==length-1)
        if needs_model:
            ip=position if args.window_supervision=="full" else 0
            batch={k:v[:,ip] for k,v in pack["inputs"].items()}
            # Replace every earlier in-window event in the five-event history.
            he,ht,hd,hh=(batch["history_evidence"],batch["history_track"],
                         batch["history_detection"],batch["history_has_detection"])
            hids=pack["history_events"][:,ip]
            for prev in history:
                select=(hids==prev["event"][:,None])&prev["valid"][:,None]
                he=torch.where(select[:,:,None],prev["evidence"][:,None],he)
                ht=torch.where(select[:,:,None],prev["track"][:,None],ht)
                hd=torch.where(select[:,:,None],prev["detection"][:,None],hd)
                hh=torch.where(select,prev["has"][:,None],hh)
            batch.update(history_evidence=he,history_track=ht,history_detection=hd,
                         history_has_detection=hh,current_evidence=evidence,
                         current_track=current_track,current_detection=current_det,
                         current_has_detection=present)
            outputs=model(**batch)
            gates=torch.sigmoid(outputs["gate_logits"])
            gm,ga=gates[:,0],gates[:,1]
            gate_sum=gate_sum+(gates*active[:,None]).sum(0)
            gate_count=gate_count+active.sum()
        else:
            # Data-only corrupted history, with the original full-write policy.
            gm=ga=torch.ones_like(score)
            outputs=None
        history.append({"event":pack["events"][:,position],"valid":active,
                        "evidence":evidence,"track":current_track,
                        "detection":current_det,"has":present})
        safe_measurement=torch.where(present[:,None],measurement,mean[:,:4])
        full_mean,full_cov=kf_update(mean,cov,safe_measurement,score)
        updated=mean+gm[:,None]*(full_mean-mean)
        updated_cov=(1-gm[:,None,None])*cov+gm[:,None,None]*full_cov
        updated_cov=0.5*(updated_cov+updated_cov.transpose(-1,-2))
        updated=torch.where(present[:,None],updated,mean)
        updated_cov=torch.where(present[:,None,None],updated_cov,cov)
        warp=pack["warp"][:,position]
        warp_ok=(warp[:,:,:2].abs().sum((-1,-2))>0)
        identity=warp.new_tensor([[1.,0.,0.],[0.,1.,0.]]).expand_as(warp)
        warp=torch.where(warp_ok[:,None,None],warp,identity)
        predicted,predicted_cov=warp_predict(updated,updated_cov,warp)
        if args.window_supervision=="full" or position==length-1:
            valid=pack["gt_valid"][:,position]&warp_ok
            gt=pack["gt"][:,position]
            with torch.no_grad():
                l0=1-_diou_xyxy(_box(_next_mean(mean,warp)),gt)
                l1=1-_diou_xyxy(_box(_next_mean(full_mean,warp)),gt)
                harmful=valid&present&(l1>l0+1e-6)
                weights=valid.float()*(1+(args.motion_harm_weight-1)*harmful.float())
            error=1-_diou_xyxy(_box(predicted),gt)
            motion_sum=motion_sum+(error*weights).sum()
            weight_sum=weight_sum+weights.sum()
        new_feature=ema_update(feature,current_det,score,ga,alpha=args.rank_alpha)
        feature=torch.where(present[:,None],new_feature,feature)
        mean=torch.where(active[:,None],predicted,mean)
        cov=torch.where(active[:,None,None],predicted_cov,cov)
        last_frame=torch.where(present,pack["frame"][:,position],last_frame)
        if position==length-1:
            anchor_outputs,anchor_track,anchor_det=outputs,current_track,current_det
        if debug:
            debug_steps.append({"mean":mean,"cov":cov,"evidence":evidence,
                                "has":present,"feature":feature,
                                "logits":None if outputs is None else outputs["gate_logits"]})
    app=dict(pack["app"])
    anchor_has=pack["has"][:,-1]
    app["current_track"]=anchor_track
    app["current_detection"]=anchor_det
    app["rank_template0"]=anchor_track
    app["rank_detection"]=anchor_det
    for name in ("appearance_valid","rank_valid","column_valid"):
        app[name]=app[name].bool()&anchor_has
    appearance,_=appearance_rank_loss(anchor_outputs,app,alpha=args.rank_alpha,
                                     tau=args.rank_tau,rank_weight=args.rank_weight)
    appearance=appearance+args.column_weight*appearance_column_loss(
        anchor_outputs,app,alpha=args.rank_alpha,tau=args.rank_tau,
        hinge=args.column_hinge,tensor_statistics=True)
    motion=motion_sum/weight_sum.clamp_min(1.)
    loss=motion+appearance
    if probe_allowed:
        trunk=base.shared_parameters(model)
        grad_probe["grad_norm_motion_shared"]=base.loss_grad_norm(motion,trunk)
        grad_probe["grad_norm_appearance_shared"]=base.loss_grad_norm(appearance,trunk)
    parts={"motion":motion.detach(),"appearance":appearance.detach(),
           "transitions":prepared["transitions"],"missing":prepared["missing"],
           "simulated":prepared["simulated"],"fallback":prepared["fallback"]}
    if debug:
        parts["debug_steps"]=debug_steps
    return loss,parts,(gate_sum/gate_count.clamp_min(1)).detach()


class WindowSequenceBank:
    """One sequence's immutable source features on GPU, never the whole dataset.

    Only index/mask/noise metadata is prefetched on CPU. Model inputs are gathered
    on the consumer CUDA stream. No encoded features or gradients are cached.
    """
    def __init__(self,data,target,device):
        from shutrack.schema import COMPETITION_SCALAR_INDICES
        self.device=torch.device(device)
        def upload(value,dtype=None):
            array=np.asarray(value)
            tensor=torch.from_numpy(np.array(array,copy=True)).to(self.device)
            return tensor.to(dtype) if dtype is not None else tensor
        mean,std=data.input_statistics
        raw=np.asarray(data.event_evidence,np.float32).copy()
        cols=list(EVIDENCE_SCALAR_INDICES)
        raw[:,:12]=(raw[:,:12]-mean[cols])/std[cols]
        self.evidence=upload(raw)
        self.track=upload(data.event_track,torch.float32)
        self.det_index=upload(data.event_det,torch.long)
        self.has=upload(data.event_has,torch.bool)
        self.features=upload(data.cache.features,torch.float32)
        self.current=upload(data.small["current_event_index"],torch.long)
        competition=np.asarray(data.small["current_competition"],np.float32).copy()
        ccols=list(COMPETITION_SCALAR_INDICES)
        competition=(competition-mean[ccols])/std[ccols]
        self.competition=upload(competition.astype(np.float32))
        data.load_rank_arrays()
        self.rank_pos=upload(data.rank_pos,torch.long)
        self.positive=upload(data._rank_arrays[2],torch.float32)
        self.competitor=upload(data._rank_arrays[3],torch.float32)
        self.rank_score=upload(data._rank_arrays[4],torch.float32)
        self.appearance_valid=upload(target["valid_appearance"],torch.bool)
        self.appearance_target=upload(target["appearance_safe"],torch.float32)
        self.column={key:upload(data.small[key],
                    torch.long if key=="column_det" else torch.float32)
                     for key in ("column_det","column_offset","column_rival","column_score")}
        self.nbytes=sum(value.numel()*value.element_size() for value in vars(self).values()
                        if torch.is_tensor(value))
        self.nbytes+=sum(value.numel()*value.element_size() for value in self.column.values())

    def inputs(self,rows,history):
        n,k=rows.shape
        current=self.current[rows]
        mask=history<0
        safe=history.clamp_min(0)
        det=self.det_index[safe]
        hdet=self.features[det.clamp_min(0)]
        hdet=hdet.masked_fill((det<0)[:,:,:,None],0)
        htrack=self.track[safe]
        he=self.evidence[safe].masked_fill(mask[:,:,:,None],0)
        ht=htrack.masked_fill(mask[:,:,:,None],0)
        hdet=hdet.masked_fill(mask[:,:,:,None],0)
        cdet=self.det_index[current]
        return {
            "history_evidence":he,"history_track":ht,"history_detection":hdet,
            "history_mask":mask,"history_has_detection":self.has[safe]&~mask,
            "history_reset":torch.zeros_like(mask),"current_evidence":self.evidence[current],
            "current_track":self.track[current],
            "current_detection":self.features[cdet.clamp_min(0)].masked_fill(
                (cdet<0)[:,:,None],0),
            "current_competition":self.competition[rows],
            "current_has_detection":self.has[current],
        }

    def appearance(self,rows):
        pos=self.rank_pos[rows]
        valid=pos>=0
        # Some tiny sequences have no rank positives.
        if self.positive.shape[0]:
            positive=self.positive[pos.clamp_min(0)]
            competitor=self.competitor[pos.clamp_min(0)]
            score=self.rank_score[pos.clamp_min(0)]
        else:
            positive=torch.zeros((len(rows),self.track.shape[1]),device=self.device)
            competitor=torch.zeros_like(positive)
            score=torch.zeros(len(rows),device=self.device)
        col=self.column["column_det"][rows]
        colvalid=col>=0
        return {
            "rank_positive":positive.masked_fill(~valid[:,None],0),
            "rank_competitor":competitor.masked_fill(~valid[:,None],0),
            "rank_score":score.masked_fill(~valid,0),"rank_valid":valid,
            "appearance_target":self.appearance_target[rows],
            "appearance_valid":self.appearance_valid[rows],
            "column_positive":self.features[col.clamp_min(0)].masked_fill(~colvalid[:,None],0),
            "column_valid":colvalid,
            **{key:self.column[key][rows] for key in
               ("column_offset","column_rival","column_score")},
        }
