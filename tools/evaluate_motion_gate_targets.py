#!/usr/bin/env python3
"""Compare gate decisions on held-out cached events using the same new architecture."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import train_supervision_groups as trainer
from shutrack.training.checkpoint import load_shutrack_checkpoint
from shutrack.training.loss import _diou_xyxy
from shutrack.training.motion_targets import SOURCE_KEYS, source_fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--target-root",required=True)
    parser.add_argument("--sequences",nargs="+",required=True)
    parser.add_argument("--dataset",default="SportsMOT")
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device("cuda")
    models={}
    payloads={}
    for name,path in (("baseline",args.baseline),("new",args.checkpoint)):
        model,payload=load_shutrack_checkpoint(path,map_location="cpu")
        models[name]=model.to(device).eval()
        payloads[name]=payload
    for key in ("normalization_mean","normalization_std"):
        # Statistics are refit on the same 45 train sequences; only float
        # accumulation order differs (~1e-14), so exact equality is too strict.
        np.testing.assert_allclose(payloads["baseline"][key],payloads["new"][key],
                                   rtol=1e-6,atol=1e-8)
    totals={name:{} for name in models}
    valid_total=0
    full_total=0.
    oracle_total=0.
    full_harmful_total=0.
    full_useful_total=0.
    with torch.inference_mode():
        for name in args.sequences:
            data=trainer.SequenceData(name,"val",Path(args.data_root),args.dataset)
            data.open_cache()
            data.input_statistics=(np.asarray(payloads["new"]["normalization_mean"]),
                                   np.asarray(payloads["new"]["normalization_std"]))
            metadata=json.loads((Path(args.target_root)/name/"metadata.json").read_text())
            source={key:data.small[key] if key in data.small else data.samples[key] for key in SOURCE_KEYS}
            if metadata["source_sha256"]!=source_fingerprint(source):
                raise ValueError(f"{name}: held-out target rows differ")
            labels=np.load(Path(args.target_root)/name/"motion_gate_target.npy",allow_pickle=False)
            targets=trainer.derive_targets(data.samples,payloads["new"]["training_config"]["taus"])
            try:
                for start in range(0,data.n,1024):
                    rows=np.arange(start,min(data.n,start+1024))
                    batch=trainer.to_device(trainer.batch_tensors(data,rows,targets,"control",
                                                                  compact_reid=True),device)
                    target=torch.from_numpy(labels[rows]).to(device)
                    valid=batch["motion_gt_valid"].bool()
                    b0,b1,gt=(batch[key].float() for key in ("pred_box_g0","pred_box_g1","gt_box_t1"))
                    l0=1-_diou_xyxy(b0,gt)
                    l1=1-_diou_xyxy(b1,gt)
                    harmful=valid & (l1>l0+1e-6)
                    useful=valid & ~harmful
                    partial=valid & (target<1)
                    valid_total+=int(valid.sum())
                    full_total+=float((l1*valid).sum())
                    full_harmful_total+=float((l1*harmful).sum())
                    full_useful_total+=float((l1*useful).sum())
                    oracle_total+=float(((1-_diou_xyxy(b0+target[:,None]*(b1-b0),gt))*valid).sum())
                    for model_name,model in models.items():
                        gate=model(**trainer.model_inputs(batch))["gates"][:,0].float()
                        loss=1-_diou_xyxy(b0+gate[:,None]*(b1-b0),gt)
                        values={"gate_sum":(gate*valid).sum(),"loss_sum":(loss*valid).sum(),
                                "target_abs_error_sum":((gate-target).abs()*valid).sum(),
                                "harmful_gate_sum":(gate*harmful).sum(),"harmful_count":harmful.sum(),
                                "harmful_gated_loss_sum":(loss*harmful).sum(),
                                "useful_gate_sum":(gate*useful).sum(),"useful_count":useful.sum(),
                                "useful_gated_loss_sum":(loss*useful).sum(),
                                "partial_gate_sum":(gate*partial).sum(),"partial_count":partial.sum()}
                        for key,value in values.items():
                            totals[model_name][key]=totals[model_name].get(key,0.)+float(value)
                print(json.dumps({"sequence":name,"valid_total":valid_total}),flush=True)
            finally:
                data.close_cache()
                data.samples.close()
    result={"valid_events":valid_total,"constant_full_write_loss":full_total/max(1,valid_total),
            "oracle_target_loss":oracle_total/max(1,valid_total)}
    for name,sums in totals.items():
        result[name]={"gate_mean":sums["gate_sum"]/valid_total,
                      "diou_loss":sums["loss_sum"]/valid_total,
                      "target_mae":sums["target_abs_error_sum"]/valid_total,
                      "net_loss_change":(sums["loss_sum"]-full_total)/max(1,valid_total)}
        for group in ("harmful","useful","partial"):
            count=sums[group+"_count"]
            result[name][group+"_count"]=count
            result[name][group+"_gate_mean"]=sums[group+"_gate_sum"]/count if count else None
        harmful_count=sums["harmful_count"]
        useful_count=sums["useful_count"]
        result[name]["harmful_savings_total"]=full_harmful_total-sums["harmful_gated_loss_sum"]
        result[name]["harmful_savings_mean"]=(full_harmful_total-sums["harmful_gated_loss_sum"])/harmful_count if harmful_count else None
        result[name]["useful_damage_total"]=sums["useful_gated_loss_sum"]-full_useful_total
        result[name]["useful_damage_mean"]=(sums["useful_gated_loss_sum"]-full_useful_total)/useful_count if useful_count else None
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
