"""Isolated audit/reproduction of the SportsMOT paper baseline. No source patching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "3. track" / "reproduce_paper_checkpoint_legacy080_20261004_retry"
STATE = BASE / "audit_state.json"
LOG = BASE / "run.log"
SUFFIX = "paper_checkpoint_legacy080_20261004_retry"
CKPT = Path("/home/shang/workspace/mamba_kalman_filter/checkpoints/SPORTSMOT_best_model.pth.exp14_3_sportsmot_b128_full_e20_rerun")
TARGET = ROOT / "outputs" / "2. det_feat" / "sportsmot_val_0.80.pickle"
SOURCES = [
    ROOT / "3. Tracker" / "run_mamba.py",
    ROOT / "3. Tracker" / "trackers" / "tracker_mamba.py",
    ROOT / "3. Tracker" / "trackers" / "mamba_kalman_filter_wrapper.py",
    ROOT / "3. Tracker" / "trackers" / "cmc.py",
    ROOT / "3. Tracker" / "utils" / "det_feat_storage.py",
    Path("/home/shang/workspace/mamba_kalman_filter/mamba_kalmanfilter.py"),
    CKPT,
]
CMD = [
    sys.executable, "-u", "run_mamba.py", "--dataset", "SportsMOT", "--mode", "val",
    "--mamba_model_path", str(CKPT), "--target-pickle-override", str(TARGET),
    "--enable-innov", "--enable-diou", "--num-trajectory-tokens", "4", "--window-size", "16",
    "--tracker-suffix", SUFFIX, "--output_dir", str(BASE) + "/",
]

def fingerprints():
    out = {}
    for path in SOURCES:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        out[str(path)] = h.hexdigest()
    return out

def write_state(data):
    BASE.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(STATE)

def relevant_processes():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "run_mamba.py" in raw and "paper_checkpoint_legacy080_20261004" in raw:
                found.append({"pid": int(entry.name), "retry": SUFFIX in raw})
        except (OSError, PermissionError):
            pass
    return found

def status():
    data = json.loads(STATE.read_text()) if STATE.exists() else {"state": "not_started"}
    data["matching_tracking_processes"] = relevant_processes()
    tracker_dir = BASE / ("sportsmot_val_0.80_mamba_" + SUFFIX)
    data["completed_sequence_files"] = len(list(tracker_dir.glob("*.txt")))
    original = ROOT / "outputs" / "3. track" / "reproduce_paper_checkpoint_legacy080_20261004"
    data["previous_attempt_sequence_files"] = len(list((original / "sportsmot_val_0.80_mamba_paper_checkpoint_legacy080_20261004").glob("*.txt")))
    if LOG.exists():
        txt = LOG.read_text(errors="replace")
        data["log_tail"] = re.split(r"[\r\n]", txt)[-12:]
    print(json.dumps(data, indent=2))

def worker():
    before = fingerprints()
    target_stat = TARGET.stat()
    state = {
        "state": "running", "controller_pid": os.getpid(), "command": CMD,
        "cwd": str(ROOT / "3. Tracker"), "started_at": time.time(),
        "fingerprints_before": before,
        "target": {"path": str(TARGET), "size": target_stat.st_size, "mtime_ns": target_stat.st_mtime_ns},
        "mamba_module_resolved": str((ROOT / "mamba_kalman_filter").resolve()),
        "log": str(LOG),
    }
    write_state(state)
    with LOG.open("w") as handle:
        proc = subprocess.Popen(CMD, cwd=ROOT / "3. Tracker", stdout=handle, stderr=subprocess.STDOUT)
        state["tracking_pid"] = proc.pid
        write_state(state)
        code = proc.wait()
    state["exit_code"] = code
    state["ended_at"] = time.time()
    state["fingerprints_after"] = fingerprints()
    state["source_and_checkpoint_unchanged"] = state["fingerprints_before"] == state["fingerprints_after"]
    stat_after = TARGET.stat()
    state["target_metadata_unchanged"] = (target_stat.st_size, target_stat.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
    txt = LOG.read_text(errors="replace")
    match = re.search(r"HOTA\s+MOTA\s+IDF1\s+DetA\s+AssA\s*\n([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    if match:
        state["metrics"] = dict(zip(["HOTA", "MOTA", "IDF1", "DetA", "AssA"], map(float, match.groups())))
    state["completed_sequence_files"] = len(list((BASE / ("sportsmot_val_0.80_mamba_" + SUFFIX)).glob("*.txt")))
    state["state"] = "completed" if code == 0 and match and state["completed_sequence_files"] == 45 else "failed"
    write_state(state)

def start():
    existing = relevant_processes()
    if existing:
        print(json.dumps({"state": "existing_tracking_process", "processes": existing}))
        return
    if STATE.exists():
        status()
        return
    BASE.mkdir(parents=True, exist_ok=True)
    write_state({"state": "starting", "command": CMD, "log": str(LOG)})
    with (BASE / "controller.log").open("w") as handle:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker"],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"state": "started", "controller_pid": proc.pid, "log": str(LOG)}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "start", "worker"])
    action = parser.parse_args().action
    {"status": status, "start": start, "worker": worker}[action]()
