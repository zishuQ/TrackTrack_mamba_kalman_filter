#!/usr/bin/env python3
"""Read-only state-rich replay data integrity/footprint verification.

Scans source arrays in bounded chunks; only writes the audit JSON.
No tracking, training, collection, or cache mutation.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tools"))
from paired_training_input import official_sequences

OUT=ROOT/"outputs/agentguard/experiments/window_training_speed_audit_20261124"
ROOTS={
    "train45":ROOT/"outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data",
    "trainval90":ROOT/"outputs/agentguard/experiments/sportsmot_mechanisms_formal_20261124/data",
}
STATE={"event_state_mean":(8,),"event_state_cov":(64,),
       "event_det_measurement":(4,),"event_det_score":(),"event_warp_next":(6,),
       "event_gap":(),"event_frame":(),"event_target_row":()}
CORE={"event_evidence":(13,),"event_detection_index":(),"event_has_detection":()}

def box(mean):
    return np.concatenate((mean[:,:2]-mean[:,2:4]/2,
                           mean[:,:2]+mean[:,2:4]/2),axis=-1)

def next_mean(mean,warp):
    moved=(mean.reshape(-1,4,2) @ warp[:,:,:2].transpose(0,2,1)).reshape(-1,8)
    moved[:,:2]+=warp[:,:,2]
    moved[:,:4]+=moved[:,4:]
    return moved

def reconstructed_evidence(mean,cov,z,score,gap,has,w,h):
    pw=np.maximum(mean[:,2],1);ph=np.maximum(mean[:,3],1)
    return np.column_stack((
        (z[:,0]-mean[:,0])/pw*has,(z[:,1]-mean[:,1])/ph*has,
        np.log(np.maximum(z[:,2],1)/pw)*has,np.log(np.maximum(z[:,3],1)/ph)*has,
        mean[:,4]/pw,mean[:,5]/ph,mean[:,6]/pw,mean[:,7]/ph,
        np.log(np.maximum(cov[:,0,0]/w**2+1e-6,1e-12)),
        np.log(np.maximum(cov[:,1,1]/h**2+1e-6,1e-12)),
        score*has,np.clip(gap,0,30)/30,has))

def footprint(root,seen):
    logical=allocated=unique=files=symlinks=state_bytes=0
    per_file=defaultdict(int)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        stat=p.stat();files+=1
        logical+=stat.st_size
        allocated+=stat.st_blocks*512
        inode=(stat.st_dev,stat.st_ino)
        if inode not in seen:
            seen.add(inode);unique+=stat.st_blocks*512
        symlinks+=int(p.is_symlink())
        per_file[p.name]+=stat.st_size
        if p.stem in STATE and p.suffix==".npy":
            state_bytes+=stat.st_size
    return {"files":files,"logical_bytes":logical,"allocated_bytes":allocated,
            "additional_unique_allocated_bytes":unique,"symlinks":symlinks,
            "new_state_arrays_bytes":state_bytes,"by_filename_bytes":dict(per_file)}

def inspect(directory):
    summary=json.loads((directory/"summary.json").read_text())
    bad=[]
    if not summary.get("complete"):
        bad.append("summary incomplete")
    arrays={}
    m=int(summary["events"]);n=int(summary["samples"])
    for key,tail in dict(STATE,**CORE).items():
        path=directory/(key+".npy")
        if not path.is_file():
            bad.append("missing "+key);continue
        value=np.load(path,mmap_mode="r",allow_pickle=False)
        arrays[key]=value
        if value.shape!=(m,)+tail:
            bad.append(f"{key}: shape {value.shape}, wanted {(m,)+tail}")
        for start in range(0,m,8192):
            if not np.isfinite(value[start:start+8192]).all():
                bad.append("nonfinite "+key);break
    if bad:
        return {"samples":n,"events":m,"errors":bad}
    with np.load(directory/"samples.npz",allow_pickle=False) as a:
        sample={key:a[key] for key in ("frame","detection_index","current_event_index",
            "window_indices","pred_box_g0","pred_box_g1","motion_gt_valid")}
    cur=sample["current_event_index"]
    if len(cur)!=n or np.any((cur<0)|(cur>=m)):
        bad.append("current event index out of range")
        return {"samples":n,"events":m,"errors":bad}
    checks={
        "sample_frame":np.array_equal(arrays["event_frame"][cur],sample["frame"]),
        "sample_detection":np.array_equal(arrays["event_detection_index"][cur],sample["detection_index"]),
        "event_target_inverse":np.array_equal(arrays["event_target_row"][cur],np.arange(n)),
        "history_index_range":bool(np.all((sample["window_indices"]>=-1)&(sample["window_indices"]<m))),
        "has_detection_evidence":np.array_equal(arrays["event_has_detection"],
                                               arrays["event_evidence"][:,12].astype(bool)),
    }
    bad += [key for key,ok in checks.items() if not ok]
    gap=arrays["event_gap"]
    relevant=arrays["event_target_row"]>=0
    missing_state=(gap<0)|np.all(arrays["event_state_mean"]==0,axis=1)
    active_missing=int((missing_state&relevant).sum())
    if active_missing:bad.append("sample-bound states missing")
    cov=np.asarray(arrays["event_state_cov"]).reshape(-1,8,8)
    diag=np.diagonal(cov,axis1=1,axis2=2)
    if not np.all(diag[relevant]>0):bad.append("nonpositive KF prior covariance diagonal")
    if not np.all(arrays["event_state_mean"][relevant,2:4]>0):
        bad.append("nonpositive sample-bound state size")
    max0=max1=ev_error=0.;checked=0
    for start in range(0,n,4096):
        selected=np.arange(start,min(start+4096,n));idx=cur[selected]
        mean=np.asarray(arrays["event_state_mean"][idx],np.float64)
        p=np.asarray(arrays["event_state_cov"][idx],np.float64).reshape(-1,8,8)
        z=np.asarray(arrays["event_det_measurement"][idx],np.float64)
        score=np.asarray(arrays["event_det_score"][idx],np.float64)
        has=np.asarray(arrays["event_has_detection"][idx])
        evidence=reconstructed_evidence(mean,p,z,score,np.asarray(gap[idx]),has,
                                        summary["img_width"],summary["img_height"])
        ev_error=max(ev_error,float(np.abs(evidence-arrays["event_evidence"][idx]).max()))
        warp=np.asarray(arrays["event_warp_next"][idx],np.float64).reshape(-1,2,3)
        valid=sample["motion_gt_valid"][selected].astype(bool)
        if not valid.any():continue
        mean=mean[valid];p=p[valid];z=z[valid];score=score[valid];warp=warp[valid]
        scale=mean[:,[2,3,2,3]]/20
        projected=p[:,:4,:4].copy()
        d=np.arange(4)
        projected[:,d,d]+=scale**2*(1-score[:,None])
        gain=np.linalg.solve(projected,p[:,:,:4].transpose(0,2,1)).transpose(0,2,1)
        updated=mean+np.einsum("bij,bj->bi",gain,z-mean[:,:4])
        actual0=box(next_mean(mean,warp));actual1=box(next_mean(updated,warp))
        max0=max(max0,float(np.abs(actual0-sample["pred_box_g0"][selected[valid]]).max()))
        max1=max(max1,float(np.abs(actual1-sample["pred_box_g1"][selected[valid]]).max()))
        checked+=int(valid.sum())
    if max(max0,max1)>.01:bad.append("endpoint reproduction exceeds .01 pixel")
    if ev_error>5e-4:bad.append("raw evidence reproduction exceeds 5e-4")
    if summary.get("warp_max_abs_error",0)!=0:bad.append("recorded GMC differs from online")
    if (summary.get("column_counts") or {}).get("actual_mismatch",0)!=0:
        bad.append("recorded column assignment mismatch")
    return {"samples":n,"events":m,"complete":summary.get("complete"),
            "checks":checks,"sample_bound_missing_states":active_missing,
            "unbound_missing_states":int((missing_state&~relevant).sum()),
            "endpoint_rows_checked":checked,"g0_max_abs_px":max0,"g1_max_abs_px":max1,
            "raw_evidence_max_abs":ev_error,"errors":bad}

def main():
    seen=set();result={"roots":{}}
    train=official_sequences("train","SportsMOT")
    expected={"train45":train,"trainval90":train|official_sequences("val","SportsMOT")}
    all_errors=[]
    for label,root in ROOTS.items():
        found={p.name for p in root.iterdir() if (p/"samples.npz").is_file()}
        item={"root":str(root),"expected_sequences":len(expected[label]),
              "found_sequences":len(found),"footprint":footprint(root,seen),"per_sequence":{}}
        if found!=expected[label]:
            all_errors.append(label+": sequence set mismatch")
        for name in sorted(found):
            row=inspect(root/name);item["per_sequence"][name]=row
            if row["errors"]:all_errors.append({label+"/"+name:row["errors"]})
        rows=list(item["per_sequence"].values())
        for key in ("samples","events","endpoint_rows_checked","sample_bound_missing_states",
                    "unbound_missing_states"):
            item[key]=sum(row.get(key,0) for row in rows)
        for key in ("g0_max_abs_px","g1_max_abs_px","raw_evidence_max_abs"):
            item[key]=max(row.get(key,0) for row in rows)
        item["data_ok"]=all(not r["errors"] for r in rows) and found==expected[label]
        result["roots"][label]=item
        print(json.dumps({"root":label,**{k:v for k,v in item.items()
                          if k not in ("root","per_sequence")}},indent=2),flush=True)
    result["all_data_ok"]=not all_errors
    result["errors"]=all_errors
    result["combined_unique_allocated_bytes"]=sum(item["footprint"]["additional_unique_allocated_bytes"]
                                                   for item in result["roots"].values())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"state_data_integrity_and_footprint.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"all_data_ok":result["all_data_ok"],"errors":all_errors,
          "combined_unique_allocated_gib":result["combined_unique_allocated_bytes"]/2**30},indent=2))
    if all_errors:raise SystemExit(1)
if __name__=="__main__":
    main()
