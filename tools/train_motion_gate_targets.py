#!/usr/bin/env python3
"""Run cached GT gate supervision in its own process using the paired trainer.

The original trainer/loss source files remain unchanged. Adaptations are scoped
to this process and restored when the run returns.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer
from shutrack.training.motion_targets import (
    GRID, SOURCE_KEYS, TIE_TOLERANCE, motion_target_bce, source_fingerprint,
)
from shutrack.training.loss import _diou_xyxy
from shutrack.training.paired_pipeline import DeferredMetrics


class PilotComplete(Exception):
    pass


def arguments_from_config(config, groups_root, resume):
    args = dict(config)
    args.update(groups_root=str(groups_root), resume=resume,
                max_epochs=config["epochs"], limit_sequences=0, sequences=None,
                device="cuda", prefetch=1, no_compact_reid=False,
                no_pin_memory=False, stage_features=None, motion_harm_weight=1.0)
    args["from_scratch"] = config["from_scratch"]
    args["init"] = config.get("init")
    return SimpleNamespace(**args)


@contextmanager
def target_training(target_root, group_dir, stop_after_epoch=0):
    target_root = Path(target_root).resolve()
    group_dir = Path(group_dir)
    names = ("SequenceData", "batch_tensors", "PAIRED_TENSOR_KEYS",
             "motion_box_diou_loss", "plan_phase", "build_checkpoint_payload",
             "save_checkpoint_atomic")
    original = {name: getattr(trainer, name) for name in names}
    instances = []
    epoch_metrics = None

    class TargetSequenceData(original["SequenceData"]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            directory = target_root / self.sequence
            metadata = json.loads((directory / "metadata.json").read_text())
            source = {key: self.small[key] if key in self.small else self.samples[key]
                      for key in SOURCE_KEYS}
            if metadata["source_sha256"] != source_fingerprint(source):
                raise ValueError(f"{self.sequence}: gate labels do not match source sample rows")
            if metadata["grid"] != GRID.tolist() or metadata["tie_tolerance"] != TIE_TOLERANCE:
                raise ValueError(f"{self.sequence}: gate label policy differs")
            self.motion_gate_target = np.load(directory / "motion_gate_target.npy", allow_pickle=False)
            if self.motion_gate_target.shape != (self.n,) or self.motion_gate_target.dtype != np.float32:
                raise ValueError(f"{self.sequence}: gate labels have wrong shape or dtype")
            if metadata["target_sha256"] != hashlib.sha256(self.motion_gate_target.tobytes()).hexdigest():
                raise ValueError(f"{self.sequence}: gate label checksum differs")
            instances.append(self)

    def batch_tensors(sequence, rows, *args, **kwargs):
        batch = original["batch_tensors"](sequence, rows, *args, **kwargs)
        batch["motion_gate_target"] = sequence.motion_gate_target[rows]
        return batch

    def plan_phase(*args, **kwargs):
        nonlocal epoch_metrics
        epoch_metrics = DeferredMetrics(torch.device("cuda"))
        return original["plan_phase"](*args, **kwargs)

    def motion_loss(outputs, batch, **kwargs):
        result = motion_target_bce(outputs, batch, **kwargs)
        if epoch_metrics is not None:
            with torch.no_grad():
                gate = outputs["gate_logits"].float()[:,0].sigmoid()
                valid = batch["motion_gt_valid"].bool()
                target = batch["motion_gate_target"]
                loss0 = 1 - _diou_xyxy(batch["pred_box_g0"].float(), batch["gt_box_t1"].float())
                loss1 = 1 - _diou_xyxy(batch["pred_box_g1"].float(), batch["gt_box_t1"].float())
                harmful = valid & (loss1 > loss0 + 1e-6)
                useful = valid & ~harmful
                partial = valid & (target < 1)
                box = batch["pred_box_g0"] + gate[:,None] * (batch["pred_box_g1"] - batch["pred_box_g0"])
                actual = 1 - _diou_xyxy(box.float(), batch["gt_box_t1"].float())
                for key,value in {
                    "valid":valid.sum(), "harmful_count":harmful.sum(),
                    "useful_count":useful.sum(), "partial_count":partial.sum(),
                    "harmful_gate_sum":(gate*harmful).sum(),
                    "useful_gate_sum":(gate*useful).sum(),
                    "partial_gate_sum":(gate*partial).sum(),
                    "target_sum":(target*valid).sum(),
                    "gate_abs_error_sum":((gate-target).abs()*valid).sum(),
                    "diou_sum":(actual*valid).sum(), "full_diou_sum":(loss1*valid).sum(),
                    "bce_sum":result[0].detach()*valid.sum(),
                }.items():
                    epoch_metrics.add(key,value)
        return result

    descriptor = {"kind":"gt_oracle_gate_bce", "target_root":str(target_root),
                  "grid":GRID.tolist(),"tie_tolerance":TIE_TOLERANCE}

    def checkpoint_payload(**kwargs):
        payload = original["build_checkpoint_payload"](**kwargs)
        payload["motion_supervision"] = descriptor
        if epoch_metrics is not None:
            sums = epoch_metrics.summary()
            n=max(1,sums["valid"])
            record={"epoch":kwargs["epoch"],"global_step":kwargs["global_step"],
                    "valid":sums["valid"],"target_mean":sums["target_sum"]/n,
                    "target_mae":sums["gate_abs_error_sum"]/n,
                    "motion_diou_loss":sums["diou_sum"]/n,
                    "full_write_diou_loss":sums["full_diou_sum"]/n,
                    "motion_target_bce":sums["bce_sum"]/n,
                    "cuda_peak_mib":torch.cuda.max_memory_allocated()/2**20,
                    "process_peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
            for name in ("harmful","useful","partial"):
                count=sums[name+"_count"]
                record[name+"_count"]=count
                record[name+"_gate_mean"]=sums[name+"_gate_sum"]/count if count else None
            with (group_dir/"motion_target_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            print("motion target epoch: "+json.dumps(record),flush=True)
        return payload

    def save_checkpoint(path, payload):
        original["save_checkpoint_atomic"](path,payload)
        if Path(path).name=="last.pt" and stop_after_epoch and payload["epoch"]>=stop_after_epoch:
            raise PilotComplete()

    trainer.SequenceData=TargetSequenceData
    trainer.batch_tensors=batch_tensors
    trainer.PAIRED_TENSOR_KEYS=original["PAIRED_TENSOR_KEYS"] | {"motion_gate_target"}
    trainer.motion_box_diou_loss=motion_loss
    trainer.plan_phase=plan_phase
    trainer.build_checkpoint_payload=checkpoint_payload
    trainer.save_checkpoint_atomic=save_checkpoint
    try:
        yield descriptor
    finally:
        for name,value in original.items():
            setattr(trainer,name,value)
        for data in instances:
            data.close_cache()
            data.samples.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--target-root",required=True)
    parser.add_argument("--groups-root",required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-epoch",type=int,default=0)
    options=parser.parse_args()
    torch.set_num_threads(4)
    config=json.loads(Path(options.config).read_text())
    group_dir=Path(options.groups_root)/config["group"]
    group_dir.mkdir(parents=True,exist_ok=True)
    with target_training(options.target_root,group_dir,options.stop_after_epoch) as descriptor:
        if options.resume:
            payload=torch.load(group_dir/"checkpoints/last.pt",map_location="cpu",weights_only=False)
            if payload.get("motion_supervision")!=descriptor:
                raise ValueError("Resume checkpoint uses different motion supervision")
        manifest=group_dir/"motion_supervision.json"
        if manifest.exists() and json.loads(manifest.read_text())!=descriptor:
            raise ValueError("Experiment directory uses different gate target settings")
        manifest.write_text(json.dumps(descriptor,indent=2)+"\n")
        try:
            trainer.train(config["group"],arguments_from_config(config,options.groups_root,options.resume))
        except PilotComplete:
            print(json.dumps({"pilot_complete":True,"stop_after_epoch":options.stop_after_epoch,
                              "checkpoint":str(group_dir/"checkpoints/last.pt")}),flush=True)


if __name__=="__main__":
    main()
