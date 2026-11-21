#!/usr/bin/env python3
"""Full test inference, post-processing, packaging and Codabench submission
for the three formal NEW-architecture checkpoints of the architecture night run.

Resumable per dataset: validated packages and existing submission records are
reused; a dataset that fails is recorded and does not block the others.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/shang/workspace/TrackTrack")
PY = ROOT / ".venv/bin/python"
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
TRACKER = ROOT / "3. Tracker"
CACHE = ROOT / "outputs/agentguard/detection_cache"
MAKE_PACKAGE = ROOT / "outputs/agentguard/experiments/shutrack_a_submit_20261117/scripts/make_package.py"
VALIDATE = ROOT / "tools/codabench_submit.py"
SUBDIR = EXP / "submissions"
RESULTS = SUBDIR / "SUBMISSION_RESULTS.json"

MOT20_SEQS = ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08")
MOT17_SEQS = ("MOT17-01-FRCNN", "MOT17-03-FRCNN", "MOT17-06-FRCNN", "MOT17-07-FRCNN",
              "MOT17-08-FRCNN", "MOT17-12-FRCNN", "MOT17-14-FRCNN")
SM_SEQMAPP = TRACKER / "trackeval/seqmap/sportsmot/test.txt"
SM_SEQS = tuple(
    line.strip() for line in SM_SEQMAPP.read_text().splitlines()
    if line.strip() and line.strip() != "name"
)
SM_LIST = EXP / "logs/sportsmot_test_sequences.txt"

DATASETS = {
    "MOT20": {
        "checkpoint": EXP / "final/MOT20/groups/appearance_row_column/checkpoints/last.pt",
        "epoch": 100, "sequences": MOT20_SEQS, "mode": "test",
        "expected": 4, "seqinfo": Path("/home/shang/datasets/MOT20/test"),
        "competition": "10050", "phase": "16383", "task": "20804",
        "dataset_flag": "mot20",
    },
    "MOT17": {
        "checkpoint": EXP / "final/MOT17/groups/appearance_row_column/checkpoints/last.pt",
        "epoch": 50, "sequences": MOT17_SEQS, "mode": "test",
        "expected": 21, "seqinfo": Path("/home/shang/datasets/MOT17/test"),
        "competition": "10049", "phase": "16382", "task": "20803",
        "dataset_flag": "mot17",
    },
    "SportsMOT": {
        "checkpoint": EXP / "final/SportsMOT/groups/appearance_row_column/checkpoints/epoch_200.pt",
        "epoch": 200, "sequences": SM_SEQS, "mode": "test",
        "expected": 150, "seqinfo": Path("/home/shang/datasets/SportsMOT/dataset/test"),
        "competition": "13077", "phase": "21402", "task": "26486",
        "dataset_flag": "sportsmot",
    },
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}


def save_results(results):
    SUBDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def run_id():
    path = SUBDIR / "run_id.txt"
    if path.is_file():
        return path.read_text().strip()
    value = time.strftime("%Y%m%dT%H%M%S")
    path.write_text(value + "\n")
    return value


def checkpoint_info(dataset, spec):
    path = spec["checkpoint"]
    if not path.is_file():
        raise RuntimeError(f"{dataset}: missing checkpoint {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config") or {}
    if config.get("architecture_variant") != "query_film_bilinear":
        raise RuntimeError(f"{dataset}: checkpoint architecture {config.get('architecture_variant')}")
    if int(config.get("appearance_proj_dim", 0)) != 256:
        raise RuntimeError(f"{dataset}: checkpoint projection {config.get('appearance_proj_dim')}")
    if int(payload.get("epoch", -1)) != spec["epoch"]:
        raise RuntimeError(f"{dataset}: checkpoint epoch {payload.get('epoch')} != {spec['epoch']}")
    return {
        "path": str(path), "sha256": sha256(path),
        "epoch": payload["epoch"], "global_step": payload.get("global_step"),
        "model_config": config,
    }


def tracking_and_post(dataset, spec, log):
    out_dir = EXP / "final" / dataset / "test" / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), "-u", "run.py",
        "--dataset", "SportsMOT" if dataset == "SportsMOT" else dataset,
        "--mode", spec["mode"], "--sequences", *spec["sequences"],
        "--seed", "10000", "--kf-type", "nsa", "--agentguard-mode", "shutrack-v2",
        "--shutrack-checkpoint", str(spec["checkpoint"]), "--agentguard-device", "cpu",
        "--detection-cache-root", str(CACHE),
        "--output_dir", str(out_dir), "--tracker-suffix", "archnight",
        "--use_post", "--skip-eval",
        "--torch-num-threads", "4", "--torch-num-interop-threads", "1",
    ]
    started = time.time()
    with log.open("ab") as handle:
        handle.write(f"\n===== {now()} tracking {dataset} =====\n".encode())
        handle.flush()
        code = subprocess.call(command, cwd=str(TRACKER), stdout=handle, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"{dataset}: tracking failed (exit {code})")
    posts = [p for p in out_dir.iterdir() if p.is_dir() and p.name.endswith("_post")]
    if len(posts) != 1:
        raise RuntimeError(f"{dataset}: expected exactly one post dir, found {posts}")
    post = posts[0]
    txts = sorted(post.glob("*.txt"))
    if dataset == "MOT17":
        frcnn = sorted(post.glob("*-FRCNN.txt"))
        if len(frcnn) != 7:
            raise RuntimeError(f"MOT17: expected 7 FRCNN txt, found {len(frcnn)}")
        for source in frcnn:
            stem = source.name[: -len("-FRCNN.txt")]
            for detector in ("DPM", "SDP"):
                shutil.copyfile(source, post / f"{stem}-{detector}.txt")
        txts = sorted(post.glob("*.txt"))
    if len(txts) != spec["expected"]:
        raise RuntimeError(f"{dataset}: expected {spec['expected']} txt, found {len(txts)}")
    return {
        "post_dir": str(post), "files": len(txts),
        "tracking_seconds": round(time.time() - started, 3),
    }


def package_and_validate(dataset, spec, post_dir, log):
    zip_path = SUBDIR / f"{dataset}.zip"
    package_log = SUBDIR / f"{dataset}_package.log"
    command = [str(PY), str(MAKE_PACKAGE), "--source", str(post_dir),
               "--zip", str(zip_path), "--expected-count", str(spec["expected"])]
    if dataset == "SportsMOT":
        command += ["--expected-names-file", str(SM_LIST)]
    with package_log.open("wb") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"{dataset}: packaging failed")
    validate_command = [
        str(PY), str(VALIDATE), "validate", "--zip", str(zip_path),
        "--dataset", spec["dataset_flag"], "--seqinfo-root", str(spec["seqinfo"]),
    ]
    if dataset == "SportsMOT":
        validate_command += ["--sequences-file", str(SM_LIST)]
    validate_log = SUBDIR / f"{dataset}_validate.log"
    with validate_log.open("wb") as handle:
        code = subprocess.call(validate_command, stdout=handle, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"{dataset}: local zip validation failed")
    return {"zip": str(zip_path), "sha256": sha256(zip_path),
            "package_log": str(package_log), "validate_log": str(validate_log)}


def submit(dataset, spec, package, log):
    name = f"shutrack_archnight_{dataset.lower()}_queryfilm_e{spec['epoch']}_{run_id()}"
    record = SUBDIR / f"{dataset}_submission.json"
    if record.is_file():
        payload = json.loads(record.read_text())
        if payload.get("id") is not None:
            return {"submission_id": payload["id"], "status": payload.get("status"),
                    "scores": payload.get("scores"), "record": str(record),
                    "reused": True}
    command = [
        str(PY), str(VALIDATE), "submit", "--zip", package["zip"],
        "--competition", spec["competition"], "--phase", spec["phase"],
        "--task", spec["task"], "--name", name,
        "--dataset", spec["dataset_flag"], "--seqinfo-root", str(spec["seqinfo"]),
        "--poll", "--poll-interval", "30", "--poll-timeout", "2400",
        "--save-record", str(record),
    ]
    if dataset == "SportsMOT":
        command += ["--sequences-file", str(SM_LIST)]
    submit_log = SUBDIR / f"{dataset}_submit.log"
    with submit_log.open("wb") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    status = None
    submission_id = None
    scores = None
    if record.is_file():
        try:
            payload = json.loads(record.read_text())
            submission_id = payload.get("id") or payload.get("submission_id")
            status = payload.get("status")
            scores = payload.get("scores")
        except json.JSONDecodeError:
            pass
    if code:
        # A created submission may still be scoring; do not resubmit.
        return {"submission_id": submission_id, "status": status or "unknown_after_error",
                "scores": scores, "record": str(record) if record.is_file() else None,
                "error": f"submit exited {code}", "log": str(submit_log)}
    return {"submission_id": submission_id, "status": status,
            "scores": scores, "record": str(record), "log": str(submit_log)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=None,
                        help="Run only the selected datasets, in DATASETS order.")
    args = parser.parse_args()
    selected = args.datasets or list(DATASETS)
    SUBDIR.mkdir(parents=True, exist_ok=True)
    SM_LIST.parent.mkdir(parents=True, exist_ok=True)
    SM_LIST.write_text("\n".join(SM_SEQS) + "\n")
    results = load_results()
    log = EXP / "logs/test_submit.log"
    for dataset in selected:
        spec = DATASETS[dataset]
        entry = results.setdefault(dataset, {})
        if entry.get("status") in ("finished", "scored"):
            continue
        try:
            if "checkpoint" not in entry:
                entry["checkpoint"] = checkpoint_info(dataset, spec)
            if "package" not in entry or not Path(entry["package"]["zip"]).is_file():
                track = tracking_and_post(dataset, spec, log)
                entry["tracking"] = track
                entry["package"] = package_and_validate(dataset, spec, track["post_dir"], log)
                results[dataset] = entry
                save_results(results)
            if entry.get("submission", {}).get("submission_id") is None:
                entry["submission"] = submit(dataset, spec, entry["package"], log)
                results[dataset] = entry
                save_results(results)
            submission = entry.get("submission") or {}
            status = str(submission.get("status") or "").lower()
            entry["status"] = "scored" if status in ("finished", "scored") else (
                "submitted" if submission.get("submission_id") else "failed"
            )
            entry["updated_at"] = now()
            results[dataset] = entry
            save_results(results)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = str(exc)
            entry["updated_at"] = now()
            results[dataset] = entry
            save_results(results)
            print(f"{dataset}: FAILED {exc}", flush=True)
    results["updated_at"] = now()
    save_results(results)
    if any((results.get(dataset) or {}).get("status") == "failed" for dataset in selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
