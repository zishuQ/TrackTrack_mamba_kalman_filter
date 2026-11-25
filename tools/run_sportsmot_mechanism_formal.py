#!/usr/bin/env python3
"""Formal trainval/test/post/single-submission stage for reviewed G0--G3.

Imported lazily by the mechanism driver. Planning performs no API requests.
Test/post/package/submission reuse the existing project pipeline.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

def plan(mechanisms):
    root=mechanisms.ROOT
    summary=json.loads((mechanisms.MECH_EXP/"report.json").read_text())
    winner=summary["best"]["label"]
    spec=mechanisms.RUNS.get(winner, {
        "window_frames":1,"window_supervision":"full","augment":"none"})
    data=root/"outputs/agentguard/experiments/sportsmot_mechanisms_formal_20261124/data"
    sequences=sorted(path for path in data.iterdir() if (path/"samples.npz").is_file())
    expected=set(mechanisms.abc.official_sequences("train","SportsMOT")) | set(
        mechanisms.abc.official_sequences("val","SportsMOT"))
    if {path.name for path in sequences} != expected:
        raise ValueError("Formal data does not match the 90 official trainval sequences")
    counts=[]
    for path in sequences:
        with np.load(path/"samples.npz",allow_pickle=False) as samples:
            counts.append(len(samples["frame"]))
    steps=sum(math.ceil(n/1024) for n in counts)
    exp=root/"outputs/agentguard/experiments/sportsmot_mechanisms_formal_review_v2_20261124"/winner
    groups=exp/"runs/groups"
    command=mechanisms.train_command(winner,spec,{"total_steps":steps*100})
    for flag,value in (("--data-root",data),("--groups-root",groups),
                       ("--warmup-steps",int(steps*100*.005))):
        command[command.index(flag)+1]=str(value)
    command=[x for x in command if x!="--resume"]
    command += ["--split","trainval"]
    checkpoint=groups/"appearance_row_column/checkpoints/epoch_100.pt"
    if (checkpoint.parent/"last.pt").is_file() and not checkpoint.is_file():
        command.append("--resume")
    return {"winner":winner,"spec":spec,"data":str(data),"experiment":str(exp),
            "checkpoint":str(checkpoint),"sequences":len(sequences),
            "samples":sum(counts),"steps_per_epoch":steps,"total_steps":steps*100,
            "warmup_steps":int(steps*100*.005),"command":list(map(str,command))}

def run(mechanisms):
    import torch
    import run_midpoint_test_submit as post
    selected=plan(mechanisms)
    exp=Path(selected["experiment"])
    post.EXP=exp
    post.SUBDIR=exp/"submissions"
    post.LOGS=exp/"logs"
    post.STATE_PATH=exp/"state.json"
    post.HEAVY=set(post.HEAVY)|{"train_window_groups.py"}
    post.NON_TERMINAL=tuple(set(post.NON_TERMINAL)|{"Submitted","Scoring","Submitted to Queue"})
    spec={
        **post.DATASETS["SportsMOT"],
        "checkpoint":Path(selected["checkpoint"]),"epoch":100,
        "previous_hota":75.840,
        "suffix":"mechanisms_review_v2_"+selected["winner"].lower(),
        "label":"sportsmot_mechanisms_review_v2_"+selected["winner"].lower(),
    }
    post.DATASETS={"SportsMOT":spec}
    driver=post.Driver()
    (exp/"selection_and_schedule.json").write_text(json.dumps(selected,indent=2)+"\n")
    (post.LOGS/"sportsmot_test_sequences.txt").write_text("\n".join(spec["sequences"])+"\n")
    if not spec["checkpoint"].is_file():
        driver.run("formal_train_"+selected["winner"],selected["command"])
    payload=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    if (payload["epoch"]!=100 or payload["global_step"]!=selected["total_steps"]
            or len(payload["training_config"]["sequence_names"])!=90):
        raise ValueError("Formal checkpoint epoch/schedule/sequence count mismatch")
    driver.state["results"].setdefault("SportsMOT",{})["checkpoint"]=driver.checkpoint_info(
        "SportsMOT",spec)
    driver.save()
    existing=driver.state["results"]["SportsMOT"]
    if "test" not in existing:
        post_dir=driver.tracking_and_post("SportsMOT",spec)
    else:
        post_dir=Path(existing["test"]["post_dir"])
        if len(list(post_dir.glob("*.txt")))!=150:
            post_dir=driver.tracking_and_post("SportsMOT",spec)
    package=existing.get("package")
    if (not package or not Path(package["zip"]).is_file()
            or post.sha256(package["zip"])!=package["sha256"]):
        package=driver.package("SportsMOT",spec,post_dir)
    try:
        submission=driver.submit("SportsMOT",spec,package)
    except RuntimeError as error:
        driver.state["ready_for_submission"]={
            "zip":package["zip"],"reason":str(error),"at":post.now()}
        driver.save()
        raise
    scores=submission.get("scores") or {}
    if isinstance(scores,list):
        scores={item.get("column_key"):item.get("score") for item in scores
                if isinstance(item,dict)}
    result={"selected":selected,"submission_id":submission.get("id"),
            "status":submission.get("status"),"scores":scores,
            "previous_hota":75.840}
    if scores.get("HOTA") is not None:
        result["hota_delta"]=float(scores["HOTA"])-75.840
    (exp/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    (exp/"report.md").write_text(
        "# SportsMOT reviewed mechanism formal result\n\n"
        +f"- Selected: {selected['winner']}\n"
        +f"- Trainval: {selected['sequences']} sequences, {selected['total_steps']} steps\n"
        +f"- Submission: {result['submission_id']} / {result['status']}\n"
        +f"- Scores: {json.dumps(scores)}\n"
        +f"- Previous HOTA: 75.840; delta: {result.get('hota_delta')}\n")
    return result
