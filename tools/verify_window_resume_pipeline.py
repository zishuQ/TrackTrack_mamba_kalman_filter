#!/usr/bin/env python3
"""Check dropout/optimizer resume and command plans; never run inference or API calls."""
from __future__ import annotations
import copy
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import verify_window_training_review as review
import run_sportsmot_g123_experiment as pipeline
import run_sportsmot_mechanism_formal as formal
from shutrack.training.checkpoint import build_checkpoint_payload, save_checkpoint_atomic

def main():
    data,target,args=review.real_data()
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    rows=np.arange(32,36,dtype=np.int64)
    prepared=review.prepare_window_batch(review.base,data,target,rows,3,args)
    review.base._seed_everything(42)
    model=review.model_for(data,device,dropout=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-step/100)
    def step(net,opt,sched):
        opt.zero_grad(set_to_none=True)
        loss,_,_=review.replay_loss(review.base,net,prepared,args,device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step();sched.step()
        return loss.detach()
    step(model,optimizer,schedule)
    out=review.OUT
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume_check_",dir=out) as temp:
        path=Path(temp)/"resume.pt"
        mean,std=data.input_statistics
        payload=build_checkpoint_payload(
            model=model,training_config={"window_implementation":"packed_replay_v2"},
            dataset_descriptor={"dataset":"SportsMOT","split":"train"},
            epoch=1,global_step=1,best_metric=0.,
            optimizer=optimizer,scheduler=schedule,normalization_mean=mean,
            normalization_std=std,include_optimizer=True)
        save_checkpoint_atomic(path,payload)
        wanted_loss=step(model,optimizer,schedule)
        wanted={k:v.detach().clone() for k,v in model.state_dict().items()}
        resumed=review.model_for(data,device,dropout=.05)
        opt=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1-step/100)
        state=torch.load(path,map_location="cpu",weights_only=False)
        resumed.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        sched.load_state_dict(state["scheduler_state_dict"])
        rng=state["rng_state"]
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"]);random.setstate(rng["python"])
        actual_loss=step(resumed,opt,sched)
        torch.testing.assert_close(wanted_loss,actual_loss,atol=0,rtol=0)
        worst=max(float((wanted[k]-v).abs().max()) for k,v in resumed.state_dict().items())
        assert worst==0,worst
        metadata=pipeline.data_steps()
        assert (metadata["samples"],metadata["steps_per_epoch"],metadata["total_steps"]) == (305631,321,32100)
        plans={}
        for winner in ("G0","G1","G2","G3"):
            (Path(temp)/"report.json").write_text(json.dumps({"best":{"label":winner}}))
            owner=SimpleNamespace(ROOT=pipeline.ROOT,MECH_EXP=Path(temp),
                RUNS=pipeline.RUNS,abc=pipeline.abc,train_command=pipeline.train_command)
            plan=formal.plan(owner)
            assert (plan["samples"],plan["sequences"],plan["steps_per_epoch"],
                    plan["total_steps"],plan["warmup_steps"]) == (593762,90,624,62400,312)
            command=plan["command"]
            assert command[command.index("--split")+1]=="trainval"
            assert command[command.index("--warmup-steps")+1]=="312"
            assert command[command.index("--window-supervision")+1]==plan["spec"]["window_supervision"]
            assert "formal_review_v2" in command[command.index("--groups-root")+1]
            plans[winner]={key:plan[key] for key in
                           ("sequences","samples","total_steps","warmup_steps","spec")}
        for name in pipeline.RUNS:
            command,_=pipeline.abc.eval_command(name,100)
            checkpoint=command[command.index("--shutrack-checkpoint")+1]
            assert "review_v2" in checkpoint and checkpoint.endswith("epoch_100.pt")
    result={"resume_loss_abs":float((wanted_loss-actual_loss).abs()),
            "resume_parameter_max_abs":worst,"dropout_rng_restored":True,
            "formal_plans":plans,"eval_final_only":True}
    (out/"resume_pipeline_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
