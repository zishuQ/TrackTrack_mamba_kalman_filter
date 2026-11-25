#!/usr/bin/env python3
"""Read-only resource and data preflight for the window training review."""
import json
import os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
heavy={"train_window_groups.py","train_supervision_groups.py","run_sportsmot_g123_experiment.py"}
jobs=[]
for item in Path("/proc").iterdir():
    if not item.name.isdigit() or int(item.name)==os.getpid(): continue
    try: argv=(item/"cmdline").read_bytes().split(b"\0")
    except OSError: continue
    names={Path(x.decode(errors="replace")).name for x in argv if x}
    if names & heavy: jobs.append({"pid":int(item.name),"scripts":sorted(names & heavy)})
mem={}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value=line.split(":",1)
    if key in {"MemAvailable","MemTotal","SwapFree"}:
        mem[key+"_gib"]=int(value.strip().split()[0])/1024**2
root=ROOT/"outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data"
first=sorted(p for p in root.iterdir() if (p/"samples.npz").is_file())[0]
summary=json.loads((first/"summary.json").read_text())
arrays={}
for name in ("event_state_mean","event_state_cov","event_det_measurement","event_det_score","event_warp_next","event_gap","event_frame","event_target_row","event_detection_index","event_has_detection"):
    a=np.load(first/(name+".npy"),mmap_mode="r"); arrays[name]={"shape":a.shape,"dtype":str(a.dtype)}
gpu={}
if torch.cuda.is_available():
    free,total=torch.cuda.mem_get_info()
    gpu={"free_gib":free/2**30,"total_gib":total/2**30,"name":torch.cuda.get_device_name()}
print(json.dumps({"jobs":jobs,"memory":mem,"gpu":gpu,"torch":torch.__version__,"sequence":first.name,"image":[summary["img_width"],summary["img_height"]],"arrays":arrays},indent=2))
