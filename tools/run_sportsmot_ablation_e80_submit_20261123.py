#!/usr/bin/env python3
"""Test+post+package+submit for the SportsMOT D-supervision ablation e80.

The checkpoint is the train-45 / val-45 ablation run whose 45-sequence val HOTA
peaked at e80 (83.589).  It is *not* the formal trainval checkpoint that was
submitted as 1064222 (official 75.774), so this measures whether the train-only
model transfers better to the official test split.

Stages reuse tools/run_midpoint_test_submit.py (same run.py flags, packaging and
Codabench duplicate protection) with a dedicated experiment directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_midpoint_test_submit as base

ROOT = base.ROOT
EXP = ROOT / "outputs/agentguard/experiments/sportsmot_ablation_e80_test_20261123"
base.EXP = EXP
base.SUBDIR = EXP / "submissions"
base.LOGS = EXP / "logs"
base.STATE_PATH = EXP / "state.json"

DATASETS = {
    "SportsMOT": {
        "checkpoint": ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120/"
                              "ablation/SportsMOT/groups/appearance_row_column/checkpoints/epoch_080.pt",
        "epoch": 80,
        "sequences": tuple(line.strip() for line in base.SM_SEQMAPP.read_text().splitlines()
                           if line.strip() and line.strip() != "name"),
        "expected": 150,
        "seqinfo": Path("/home/shang/datasets/SportsMOT/dataset/test"),
        "competition": "13077", "phase": "21402", "task": "26486",
        "dataset_flag": "sportsmot", "previous_hota": 75.774,
        "suffix": "ablation_e80", "label": "sportsmot_ablation",
    },
}
base.DATASETS = DATASETS


def write_report(driver):
    name = "SportsMOT"
    spec = DATASETS[name]
    entry = driver.state["results"].get(name, {})
    submission = entry.get("submission") or {}
    scores = submission.get("scores") or {}
    if isinstance(scores, list):
        scores = {item.get("column_key"): item.get("score")
                  for item in scores if isinstance(item, dict)}
    hota = scores.get("HOTA")
    delta = None if hota is None else float(hota) - spec["previous_hota"]
    lines = ["# SportsMOT 消融 e80 test 提交", "",
             f"- checkpoint: `{spec['checkpoint']}` (epoch {spec['epoch']})",
             "- val HOTA (45 序列): 83.589",
             f"- 旧官方最佳（正式 trainval e80）: {spec['previous_hota']:.3f}",
             f"- 官方 HOTA: {'-' if hota is None else f'{float(hota):.3f}'}",
             f"- 相对旧最佳: {'-' if delta is None else f'{delta:+.3f}'}",
             f"- 提交: {submission.get('id', '-')} / {submission.get('status', '-')} / "
             f"{submission.get('account', '-')}",
             f"- gate: {json.dumps((entry.get('test') or {}).get('gates'), ensure_ascii=False)}",
             ""]
    (EXP / "report.md").write_text("\n".join(lines) + "\n")
    (EXP / "report.json").write_text(json.dumps(driver.state, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "test", "package", "submit", "report"),
                        default="all")
    args = parser.parse_args()
    driver = base.Driver()
    name = "SportsMOT"
    spec = DATASETS[name]
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    (EXP / "logs/sportsmot_test_sequences.txt").write_text(
        "\n".join(spec["sequences"]) + "\n")
    driver.state["results"].setdefault(name, {})["checkpoint"] = driver.checkpoint_info(name, spec)
    driver.save()
    if args.stage in ("all", "test"):
        post = driver.tracking_and_post(name, spec)
        print(json.dumps({"post": str(post),
                          "gates": driver.state["results"][name]["test"]["gates"]}), flush=True)
    if args.stage in ("all", "package"):
        post = Path(driver.state["results"][name]["test"]["post_dir"])
        package = driver.package(name, spec, post)
        print(json.dumps({"package": package}), flush=True)
    if args.stage in ("all", "submit"):
        package = driver.state["results"][name]["package"]
        payload = driver.submit(name, spec, package)
        print(json.dumps({"id": payload.get("id"), "status": payload.get("status"),
                          "scores": payload.get("scores")}), flush=True)
    if args.stage in ("all", "report"):
        write_report(driver)


if __name__ == "__main__":
    main()
