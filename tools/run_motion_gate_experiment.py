#!/usr/bin/env python3
"""Resource-assessed, serial SportsMOT gate-target experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from arch_night_driver import SM_DATA, SM_VAL

BASE=ROOT/"outputs/agentguard/experiments/shutrack_architecture_night_20261120"
DEFAULT_CONFIG=BASE/"ablation/SportsMOT/groups/appearance_row_column/config.json"
DEFAULT_OUTPUT=ROOT/"outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT"
PY=ROOT/".venv/bin/python"
HEAVY={"train_supervision_groups.py","train_motion_gate_targets.py","run.py",
       "arch_night_test_submit.py","generate_trainval_paired.py",
       "aflink_use.py","gbi_use.py","linear_interpolation_use.py","verify_feature_staging.py",
       "benchmark_inference_device_20261121.py","evaluate_motion_gate_targets.py"}


def resource_snapshot():
    memory={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,value=line.split(":",1)
        if key in ("MemTotal","MemAvailable"):
            memory[key]=int(value.split()[0])/1024
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total,memory.used,memory.free",
                                 "--format=csv,noheader,nounits"],text=True).splitlines()[0]
    total,used,free=(float(x.strip()) for x in raw.split(","))
    jobs=[]
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name)==os.getpid():
            continue
        try:
            argv=[a.decode(errors="replace") for a in (item/"cmdline").read_bytes().split(b"\0") if a]
            scripts={Path(arg).name for arg in argv if arg.endswith(".py")}
            if scripts & HEAVY:
                jobs.append({"pid":int(item.name),"scripts":sorted(scripts & HEAVY)})
            elif "arch_night_driver.py" in scripts:
                state=json.loads((BASE/"TASK_STATE.json").read_text())["tasks"]
                pending=[name for name in ("final_mot20_train","test_submit_mot20",
                         "final_mot17_train","test_submit_mot17")
                         if state.get(name,{}).get("status")!="complete"]
                if pending:
                    jobs.append({"pid":int(item.name),"pending_formal_tasks":pending})
        except (OSError,ValueError):
            continue
    return {"time":time.strftime("%Y-%m-%d %H:%M:%S"),"host_available_mib":memory["MemAvailable"],
            "host_total_mib":memory["MemTotal"],"gpu_total_mib":total,"gpu_used_mib":used,
            "gpu_free_mib":free,"active_jobs":jobs}


class Experiment:
    def __init__(self,config_path,output):
        self.config_path=Path(config_path).resolve()
        self.config=json.loads(self.config_path.read_text())
        self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        self.targets=self.output/"targets"
        self.groups=self.output/"groups"
        self.group=self.groups/self.config["group"]
        self.baseline=BASE/"ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt"
        self.state_path=self.output/"state.json"
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),"stages":{}}
        self.min_gpu=6400.
        preflight=self.output/"resource_preflight.json"
        if preflight.is_file():
            measured=[row for row in json.loads(preflight.read_text())
                      if row.get("batch_size")==1024 and "peak_reserved_mib" in row]
            if measured:
                self.min_gpu=max(2048.,math.ceil(measured[-1]["peak_reserved_mib"]+1024))
        self.min_host=18000.
        self.save()

    def save(self):
        temporary=self.state_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(self.state,indent=2)+"\n")
        temporary.replace(self.state_path)

    def wait_for_resources(self,stage):
        stable=0
        while True:
            snapshot=resource_snapshot()
            ready=(not snapshot["active_jobs"] and snapshot["gpu_free_mib"]>=self.min_gpu
                   and snapshot["host_available_mib"]>=self.min_host)
            snapshot.update(stage=stage,required_gpu_free_mib=self.min_gpu,
                            required_host_available_mib=self.min_host,ready=ready)
            with (self.output/"resource_checks.jsonl").open("a") as handle:
                handle.write(json.dumps(snapshot)+"\n")
            self.state["resource_assessment"]=snapshot
            self.save()
            print("resource assessment: "+json.dumps(snapshot),flush=True)
            stable=stable+1 if ready else 0
            if stable>=2:
                return
            time.sleep(15 if ready else 30)

    def run_command(self,name,command,cwd=ROOT):
        if self.state["stages"].get(name,{}).get("status")=="complete":
            return
        self.wait_for_resources(name)
        self.state["stages"][name]={"status":"running","command":[str(x) for x in command]}
        self.save()
        started=time.time()
        with (self.output/(name+".log")).open("a") as log:
            log.write(json.dumps([str(x) for x in command])+"\n")
            log.flush()
            result=subprocess.run([str(x) for x in command],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,"MALLOC_ARENA_MAX":"2","SHUTRACK_MAX_RSS_GB":"100000"})
        self.state["stages"][name].update(
            status="complete" if result.returncode==0 else "failed",
            exit_code=result.returncode,seconds=time.time()-started)
        self.save()
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {self.output/(name+'.log')}")

    def train_command(self):
        return [PY,"-u",ROOT/"tools/train_motion_gate_targets.py",
                "--config",self.config_path,"--target-root",self.targets,"--groups-root",self.groups]

    def evaluate_gates(self,name):
        self.run_command(name,[PY,"-u",ROOT/"tools/evaluate_motion_gate_targets.py",
            "--baseline",self.baseline,"--checkpoint",self.group/"checkpoints/last.pt",
            "--data-root",SM_DATA,"--target-root",self.targets,"--sequences",*SM_VAL,
            "--out",self.output/(name+".json")])

    def run(self):
        names=sorted(set(self.config["sequence_names"]) | set(SM_VAL))
        self.run_command("generate_targets",[PY,"-u",ROOT/"tools/generate_motion_gate_targets.py",
            "--data-root",SM_DATA,"--target-root",self.targets,"--sequences",*names])
        pilot_command=self.train_command()+["--stop-after-epoch","8"]
        if (self.group/"checkpoints/last.pt").is_file():
            pilot_command.append("--resume")
        self.run_command("pilot_train",pilot_command)
        pilot_records=[json.loads(line) for line in (self.group/"motion_target_metrics.jsonl").read_text().splitlines()]
        last=pilot_records[-1]
        self.state["pilot_assessment"]=last
        self.min_gpu=max(self.min_gpu,math.ceil(last["cuda_peak_mib"]+1024))
        self.min_host=max(18000.,math.ceil(last["process_peak_rss_gib"]*1024+4096))
        self.save()
        self.evaluate_gates("pilot_heldout_gates")
        self.run_command("full_train",self.train_command()+["--resume"])
        self.evaluate_gates("final_heldout_gates")
        out=self.output/"eval_val"
        self.run_command("tracking_val",[PY,"-u","run.py","--dataset","SportsMOT","--mode","val",
            "--sequences",*SM_VAL,"--seed","10000","--kf-type","nsa",
            "--agentguard-mode","shutrack-v2","--shutrack-checkpoint",self.group/"checkpoints/last.pt",
            "--agentguard-device","cpu","--detection-cache-root",ROOT/"outputs/agentguard/detection_cache",
            "--output_dir",out,"--tracker-suffix","motion_target",
            "--agentguard-stats-output",out/"gates.jsonl","--print-per-sequence-metrics",
            "--eval-json",out/"summary.json","--torch-num-threads","4","--torch-num-interop-threads","1"],
            cwd=ROOT/"3. Tracker")
        old=json.loads((BASE/"ablation/SportsMOT/evals/new/summary.json").read_text())
        new=json.loads((out/"summary.json").read_text())
        assert len(new["per_sequence"])==len(SM_VAL)
        report={"baseline":old["combined"],"motion_target":new["combined"],"delta":{}}
        for key in old["combined"]:
            report["delta"][key]=new["combined"][key]-old["combined"][key]
        (self.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
        lines=["# SportsMOT 新架构：逐事件 gate 监督对照","",
               "训练：官方 train 45 条；验证：val 45 条。两组均新架构、seed 42、200 分片轮次。",
               "","| 指标 | 原 DIoU 监督 | gate 标签 BCE | 变化 |","|---|---:|---:|---:|"]
        for key in ("HOTA","AssA","DetA","IDF1","MOTA","IDSW"):
            a,b=old["combined"][key],new["combined"][key]
            if key=="IDSW":
                lines.append(f"| {key} | {a:.0f} | {b:.0f} | {b-a:+.0f} |")
            else:
                lines.append(f"| {key} | {a*100:.3f} | {b*100:.3f} | {(b-a)*100:+.3f} pp |")
        (self.output/"comparison.md").write_text("\n".join(lines)+"\n")
        self.state["status"]="complete"
        self.state["comparison"]=report
        self.save()
        print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    args=parser.parse_args()
    experiment=Experiment(args.config,args.output)
    try:
        experiment.run()
    except Exception as error:
        experiment.state.update(status="failed",error=str(error))
        experiment.save()
        raise


if __name__=="__main__":
    main()
