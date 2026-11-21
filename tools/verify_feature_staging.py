#!/usr/bin/env python3
"""Check real paired batches and optimizer updates; time one staged shard."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as training
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model
from shutrack.training.feature_staging import release_sequence_pages, staged_sequence_rows
from shutrack.training.loss import motion_box_diou_loss, appearance_rank_loss, appearance_column_loss


def updates(data, targets, rows, config, staged):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = ShutrackV2Model(ShutrackV2Config(**config)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    with staged_sequence_rows(data, rows, "appearance_row_column", enabled=staged):
        for start in range(0, len(rows), 64):
            batch = training.to_device(training.batch_tensors(
                data, rows[start:start+64], targets, "appearance_row_column", compact_reid=True),
                torch.device("cuda"))
            outputs = model(**training.model_inputs(batch))
            motion = motion_box_diou_loss(outputs, batch)[0]
            appearance = appearance_rank_loss(outputs, batch, alpha=.95,tau=.1,rank_weight=18)[0]
            column = appearance_column_loss(outputs, batch, alpha=.95,tau=.1)
            loss = motion + appearance + 50 * column
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
    state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    opt = optimizer.state_dict()
    moments = {str(key)+"/"+name:value.detach().cpu().clone()
               for key,item in opt["state"].items() for name,value in item.items()
               if torch.is_tensor(value)}
    result = (float(loss.detach()), state, moments)
    del model, optimizer, outputs, loss, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    print("loading checkpoint", flush=True)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config=payload["training_config"]
    root=Path(config["data_root"])
    datasets={name:training.SequenceData(name,config["split"],root,config["dataset"])
              for name in config["sequence_names"]}
    print("sequence metadata loaded", flush=True)
    for data in datasets.values():
        data.open_cache()
        data.load_rank_arrays()
        release_sequence_pages(data, discard_cache=True)
        data.input_statistics=(np.asarray(payload["normalization_mean"]),np.asarray(payload["normalization_std"]))
    epoch=int(payload["epoch"])+1
    order,plans,_,_=training.plan_phase(datasets,(epoch-1)%config["shards"],
        config["shards"],config["sequence_sampling"],config["sequence_sampling_power"],config["seed"],epoch)
    reports=[]
    try:
        for name in order:
            print("checking " + name, flush=True)
            data=datasets[name]
            targets=training.derive_targets(data.samples,config["taus"])
            rows=plans[name]
            sample=rows[:128]
            print("building reference batch " + name, flush=True)
            reference=training.batch_tensors(data,sample,targets,config["group"],compact_reid=True)
            print("staging " + name, flush=True)
            release_sequence_pages(data)
            started=time.perf_counter()
            with staged_sequence_rows(data,rows,config["group"]) as stage:
                print(json.dumps({"staged":name,**stage}),flush=True)
                actual=training.batch_tensors(data,sample,targets,config["group"],compact_reid=True)
                for key in reference:
                    np.testing.assert_array_equal(actual[key],reference[key],err_msg=name+"/"+key)
                    assert actual[key].dtype==reference[key].dtype
                del actual,reference
                batches=0
                gather_started=time.perf_counter()
                for start in range(0,len(rows),config["batch_size"]):
                    batch=training.batch_tensors(data,rows[start:start+config["batch_size"]],
                                                targets,config["group"],compact_reid=True)
                    batches+=1
                gather_seconds=time.perf_counter()-gather_started
                del batch
                report={"sequence":name,"samples":len(rows),"batches":batches,
                        "stage_seconds":stage["seconds"],"staged_gib":stage["bytes"]/2**30,
                        "batch_build_seconds":gather_seconds,"batch_values_equal":True}
            report["total_seconds"]=time.perf_counter()-started
            reports.append(report)
            print(json.dumps(report),flush=True)
        # Identical short optimizer updates, including all three D losses.
        data=datasets[order[0]]
        targets=training.derive_targets(data.samples,config["taus"])
        sample=plans[order[0]][:128]
        old=updates(data,targets,sample,payload["model_config"],False)
        new=updates(data,targets,sample,payload["model_config"],True)
        parameter_diff=max((old[1][k]-new[1][k]).abs().max().item() for k in old[1])
        optimizer_diff=max((old[2][k]-new[2][k]).abs().max().item() for k in old[2])
        checks={"loss_abs_diff":abs(old[0]-new[0]),"parameter_max_abs_diff":parameter_diff,
                "optimizer_max_abs_diff":optimizer_diff}
        assert max(checks.values())<=1e-6,checks
        report={"epoch_plan":epoch,"sequences":reports,"optimizer_checks":checks,
                "peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        Path(args.out).write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"ok":True,**checks,"peak_rss_gib":report["peak_rss_gib"]}),flush=True)
    finally:
        for data in datasets.values(): data.close_cache()


if __name__=="__main__":
    main()
