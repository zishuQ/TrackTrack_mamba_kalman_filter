"""Collect missing formal shards only, with an independent state and log.

Default is a read-only plan. --run --wait-idle waits for existing project jobs;
it never stops them and never starts model training, tracking or test submission.
Completed data is checked by the chunked data validator.
"""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
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
    needles = ("arch_night_driver.py", "train_supervision_groups.py",
               "generate_trainval_paired.py", "verify_final_data.py",
               "verify_speed4.py")
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
    from paired_collection_io import atomic_json
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
    from paired_collection_io import inspect_shard
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
                from paired_collection_io import atomic_json
                wait_idle(args.wait_idle, EXP / "COLLECTION_STATUS.json")
                # Import the heavy tracker stack only after the existing chain is idle.
                from arch_night_driver import Driver, generate_sequences, now

                class CollectionDriver(Driver):
                    def __init__(self):
                        self.state_path = EXP / "COLLECTION_TASK_STATE.json"
                        self.state = (json.loads(self.state_path.read_text())
                                      if self.state_path.is_file() else {"tasks": {}, "created_at": now()})
                        self.tasks = {}
                        self.logs = EXP / "logs"
                        self.logs.mkdir(exist_ok=True)

                    def save(self):
                        atomic_json(self.state_path, self.state)

                    def run_command(self, *call_args, **call_kwargs):
                        wait_idle(args.wait_idle, EXP / "COLLECTION_STATUS.json")
                        atomic_json(EXP / "COLLECTION_STATUS.json",
                                    {"status": ("validating" if call_args[0].startswith("validate_")
                                                else "collecting"), "task": call_args[0]})
                        return super().run_command(*call_args, **call_kwargs)

                driver = CollectionDriver()
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
