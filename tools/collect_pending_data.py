"""Collect missing formal shards only, with an independent state and log.

Default is a read-only plan. --run --wait-idle waits for existing project jobs;
it waits for the current job to finish before starting collection replay.
Completed data is checked by the chunked data validator.
"""
import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from paired_collection_io import atomic_json, inspect_shard

EXP = ROOT / "outputs/agentguard/experiments/shutrack_collection_fix_20261121"
PROTOCOL = {"schema": 3, "full_write": True, "anchor": False, "column": True,
            "gap3": False, "rank_arrays": True, "teacher_sha256": None}


def groups():
    parent = ROOT / "outputs/agentguard/experiments"
    maps = ROOT / "3. Tracker/trackeval/seqmap/sportsmot"
    sports = set()
    for split in ("train", "val"):
        sports.update(line.strip() for line in (maps / (split + ".txt")).read_text().splitlines()
                      if line.strip() and line.strip() != "name")
    return [
        ("MOT20", parent / "shutrack_bd_mot_20261120/MOT20", "train",
         tuple(f"MOT20-{i:02d}" for i in (1, 2, 3, 5))),
        ("MOT17", parent / "shutrack_bd_mot_20261120/MOT17", "train",
         tuple(f"MOT17-{i:02d}-FRCNN" for i in (2, 4, 5, 9, 10, 11, 13))),
        ("SportsMOT", parent / "shutrack_bd_sportsmot_20261120", "trainval",
         tuple(sorted(sports))),
    ]


def active_project_jobs():
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    needles = (
        "train_supervision_groups.py", "generate_trainval_paired.py", "verify_final_data.py",
    )
    active = []
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        executable = Path(command.split()[0]).name
        if not executable.startswith("python"):
            continue
        if any(name in command for name in needles) or ("run.py" in command and "--dataset" in command):
            active.append({"pid": int(fields[0]), "command": command})
    return active


def wait_idle(enabled, state_path):
    quiet = 0
    while True:
        active = active_project_jobs()
        if active:
            quiet = 0
            if not enabled:
                raise RuntimeError("Project jobs are active; use --wait-idle to queue safely")
            atomic_json(state_path, {"status": "waiting_for_existing_jobs", "jobs": active})
            print("Waiting for existing project jobs: " + ", ".join(str(x["pid"]) for x in active),
                  flush=True)
        else:
            quiet += 1
            if quiet >= 3:
                return
        time.sleep(10)


def plan(selected):
    result = {}
    for dataset, root, split, sequences in selected:
        ready, pending, incompatible = [], [], []
        protocol = dict(PROTOCOL)
        if dataset != "SportsMOT":
            protocol["dataset"] = dataset
        for sequence in sequences:
            directory = root / "data" / sequence
            try:
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("protocol") != protocol:
                    incompatible.append(sequence)
                    continue
                inspect_shard(directory, protocol)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                pending.append(sequence)
            else:
                ready.append(sequence)
        result[dataset] = {"expected": len(sequences), "ready": ready, "pending": pending,
                           "incompatible": incompatible}
    return result


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _proc_rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


class CollectionDriver:
    def __init__(self, wait_for_idle):
        self.wait_for_idle = wait_for_idle
        self.state_path = EXP / "COLLECTION_TASK_STATE.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else {"tasks": {}, "created_at": now()}
        )
        self.logs = EXP / "logs"
        self.logs.mkdir(exist_ok=True)

    def save(self):
        atomic_json(self.state_path, self.state)

    def task_state(self, name):
        return self.state["tasks"].setdefault(name, {"status": "pending"})

    def run_command(self, label, command):
        wait_idle(self.wait_for_idle, EXP / "COLLECTION_STATUS.json")
        status = "validating" if label.startswith("validate_") else "collecting"
        atomic_json(EXP / "COLLECTION_STATUS.json", {"status": status, "task": label})
        log_path = self.logs / f"{label}.log"
        started = now()
        wall = time.time()
        entry = self.task_state(label)
        if entry.get("pid"):
            previous = int(entry["pid"])
            if os.path.isdir(f"/proc/{previous}"):
                try:
                    cmdline = Path(f"/proc/{previous}/cmdline").read_bytes().decode(
                        errors="ignore"
                    )
                except OSError:
                    cmdline = ""
                needles = [str(part) for part in command if str(part).endswith(".py")]
                if needles and any(needle in cmdline for needle in needles):
                    raise RuntimeError(
                        f"{label}: previous instance pid {previous} is still running; "
                        "refusing to start a second copy"
                    )
        entry.update({"command": command, "started_at": started, "status": "running"})
        self.save()
        with log_path.open("ab") as handle:
            handle.write(f"\n===== {started} =====\n{command}\n".encode())
            handle.flush()
            proc = subprocess.Popen(
                command, cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "MALLOC_ARENA_MAX": "2",
                    "MALLOC_TRIM_THRESHOLD_": "131072",
                    "MALLOC_MMAP_THRESHOLD_": "131072",
                },
                start_new_session=True,
            )
            entry["pid"] = proc.pid
            self.save()
            stop = threading.Event()
            watchdog = {"killed": False, "reason": None}
            max_rss = float(os.environ.get("ARCH_MAX_RSS_GB", "16"))
            min_avail = float(os.environ.get("ARCH_MIN_AVAIL_GB", "2.0"))

            def watch():
                low_streak = 0
                while not stop.wait(5):
                    rss = _proc_rss_gb(proc.pid)
                    avail = _mem_available_gb()
                    if rss > max_rss or avail < min_avail:
                        low_streak += 1
                        if low_streak >= 3:
                            watchdog["killed"] = True
                            watchdog["reason"] = f"rss={rss:.1f}GB avail={avail:.1f}GB"
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            break
                    else:
                        low_streak = 0

            thread = threading.Thread(target=watch, daemon=True)
            thread.start()
            code = proc.wait()
            stop.set()
            thread.join(timeout=1)
            if watchdog["killed"]:
                entry["memory_watchdog"] = watchdog["reason"]
        entry.update({
            "finished_at": now(),
            "seconds": round(time.time() - wall, 3),
            "exit_code": code,
            "log": str(log_path),
        })
        entry["status"] = "complete" if code == 0 else "failed"
        self.save()
        return code

def generate_sequences(driver, name, root, dataset, split, sequences):
    protocol = dict(PROTOCOL)
    if dataset != "SportsMOT":
        protocol["dataset"] = dataset
    missing = []
    for seq in sequences:
        directory = Path(root) / "data" / seq
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            missing.append(seq)
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{name}: unreadable summary for {seq}: {exc}") from exc
        if summary.get("protocol") != protocol:
            raise RuntimeError(f"{name}: incompatible replay protocol for {seq}")
        if not summary.get("complete"):
            missing.append(seq)
            continue
        try:
            inspect_shard(directory, protocol, repair_counts=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            missing.append(seq)
    if missing:
        command = [
            sys.executable, "-u", "tools/generate_trainval_paired.py",
            "--root", str(root), "--dataset", dataset, "--split", split,
            "--full-write", "--column-probe", "--lean", "--only", *missing,
        ]
        code = driver.run_command(name, command)
        if code:
            raise RuntimeError(f"{name}: generation failed")
    checked = {}
    for seq in sequences:
        checked[seq] = inspect_shard(Path(root) / "data" / seq, protocol,
                                     repair_counts=True)
    entry = driver.task_state(name)
    entry.update(status="complete", sequences=list(sequences), collection_headers=checked)
    entry["finished_at"] = entry.get("finished_at") or now()
    driver.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=("MOT20", "MOT17", "SportsMOT"))
    args = parser.parse_args()
    selected = [g for g in groups() if not args.datasets or g[0] in args.datasets]
    if not args.run:
        print(json.dumps(plan(selected), indent=2))
        return
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The collection-only controller is already running")
        with (EXP / "controller.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                wait_idle(args.wait_idle, EXP / "COLLECTION_STATUS.json")
                driver = CollectionDriver(args.wait_idle)
                results = {}
                for dataset, root, split, sequences in selected:
                    label = "collect_" + dataset.lower()
                    atomic_json(EXP / "COLLECTION_STATUS.json",
                                {"status": "collecting_or_checking", "dataset": dataset})
                    try:
                        generate_sequences(driver, label, root, dataset, split, sequences)
                        report = EXP / "reports" / (dataset + "_validation.json")
                        command = [
                            sys.executable, "tools/verify_final_data.py", "--data-root", str(root / "data"),
                            "--dataset", dataset, "--split", split, "--expect-full-write",
                            "--require-column", "--sequences", *sequences, "--out", str(report),
                        ]
                        if driver.run_command("validate_" + dataset.lower(), command):
                            raise RuntimeError(f"{dataset}: data validation failed; see collection logs")
                    except Exception as exc:
                        results[dataset] = {"status": "failed", "error": str(exc)}
                        atomic_json(EXP / "COLLECTION_ONLY_RESULT.json", results)
                        atomic_json(EXP / "COLLECTION_STATUS.json",
                                    {"status": "failed", "dataset": dataset, "error": str(exc)})
                        raise
                    results[dataset] = {"status": "complete", "sequences": len(sequences)}
                    atomic_json(EXP / "COLLECTION_ONLY_RESULT.json", results)
                atomic_json(EXP / "COLLECTION_STATUS.json", {"status": "complete", "results": results})
                print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
