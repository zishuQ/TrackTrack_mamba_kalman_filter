#!/usr/bin/env python3
"""MOT20 single-shard 25-epoch counterpart: test+post, package, submit, report.

Training runs separately. This driver resolves its checkpoint and every
evaluation output under the selected experiment root, and compares the official
score with 66.637.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_mot_e80_round as base  # noqa: E402

DEFAULT_EXP = ROOT / "outputs/agentguard/experiments/mot20_single_shard_e25_20261124"
TRAIN_GROUP = Path("runs/single_shard/groups/appearance_row_column")
SPEC = {
    "epoch": 25,
    "sequences": ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"),
    "expected": 4, "seqinfo": Path.home() / "datasets/MOT20/test",
    "competition": "10050", "phase": "16383", "task": "20804",
    "dataset_flag": "mot20", "previous_hota": 66.637,
}
LEDGER = ROOT / ".claude/skills/mot-pipeline/experiments.json"


def baseline_scores():
    ledger = json.loads(LEDGER.read_text())
    entry = ledger["datasets"]["MOT20"]["experiments"].get("formal_e100", {})
    return entry.get("official_scores") or {}


def training_stats(experiment):
    metrics_path = experiment / TRAIN_GROUP / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    metrics = [json.loads(line) for line in
               metrics_path.read_text().splitlines() if line.strip()]
    if not metrics:
        return None
    last = metrics[-1]
    return {"epochs": len(metrics), "last_epoch": last.get("epoch"),
            "global_step": last.get("global_step"),
            "epoch_seconds_mean": sum(r["epoch_seconds"] for r in metrics) / len(metrics),
            "epoch_seconds_total": sum(r["epoch_seconds"] for r in metrics),
            "rss_peak_gb": last.get("rss_peak_gb"),
            "cuda_peak_allocated_mb": last.get("cuda_peak_allocated_mb"),
            "samples_per_epoch": last.get("samples")}


def scores_of(payload):
    scores = payload.get("scores")
    if isinstance(scores, list):
        return {item.get("column_key"): item.get("score")
                for item in scores if isinstance(item, dict)}
    return scores or {}


def report(driver, experiment):
    stats = training_stats(experiment)
    results = driver.state["results"].get("MOT20", {})
    submission = results.get("submission") or {}
    scores = scores_of(submission)
    baseline = baseline_scores()
    keys = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag")
    lines = ["# MOT20 单分片 25 全量轮 ↔ 官方 test 提交", "",
             "基准：正式 e100 分片训练，官方 HOTA 66.637（提交 1063086）。", "",
             "## 训练", ""]
    if stats:
        lines.append(f"- 实际轮数 {stats['epochs']}（epoch {stats['last_epoch']}），"
                     f"global_step {stats['global_step']}，每轮样本 {stats['samples_per_epoch']:,}")
        lines.append(f"- 轮时均值 {stats['epoch_seconds_mean']:.1f} s，训练合计 {stats['epoch_seconds_total']:.0f} s")
        lines.append(f"- RSS 峰值 {stats['rss_peak_gb']} GiB，CUDA 峰值 {stats['cuda_peak_allocated_mb']} MiB")
    else:
        lines.append("- 训练指标缺失")
    lines += ["", "## 官方结果", "",
              "| 指标 | e100 基线 | e25 单分片 | Δ |", "|---|---:|---:|---:|"]
    for key in keys:
        old = baseline.get(key)
        new = scores.get(key)
        delta = "-"
        if old is not None and new is not None:
            try:
                delta = f"{float(new)-float(old):+.3f}"
            except (TypeError, ValueError):
                delta = "-"
        lines.append(f"| {key} | {old if old is not None else '-'} | "
                     f"{new if new is not None else '-'} | {delta} |")
    lines += ["", f"- 提交：{submission.get('id', '-')} / {submission.get('status', '-')} / "
                  f"{submission.get('account', '-')}"]
    test = results.get("test") or {}
    if test.get("gates"):
        g = test["gates"]
        lines.append(f"- gate（4 条 test）：motion {g['motion_min']:.4f}/"
                     f"{g['motion_mean']:.4f}/{g['motion_max']:.4f}，appearance "
                     f"{g['appearance_min']:.4f}/{g['appearance_mean']:.4f}/"
                     f"{g['appearance_max']:.4f}")
    package = results.get("package") or {}
    if package.get("zip"):
        lines.append(f"- ZIP：`{package['zip']}`，sha256 {package['sha256']}")
    (experiment / "report.md").write_text("\n".join(lines) + "\n")
    (experiment / "report.json").write_text(json.dumps(
        {"training": stats, "scores": scores, "baseline": baseline,
         "state": driver.state}, indent=2, sort_keys=True) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "test", "package", "submit", "report", "plan"),
                        default="all")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXP)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    checkpoint = experiment / TRAIN_GROUP / "checkpoints/epoch_025.pt"
    if args.stage == "plan":
        print(json.dumps({
            "experiment": str(experiment),
            "checkpoint": str(checkpoint),
            "tracking": str(experiment / "MOT20/test/tracking"),
            "submissions": str(experiment / "submissions"),
            "logs": str(experiment / "logs"),
            "state": str(experiment / "state.json"),
        }, indent=2))
        return
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")

    base.EXP = experiment
    base.SUBDIR = experiment / "submissions"
    base.LOGS = experiment / "logs"
    base.STATE_PATH = experiment / "state.json"
    driver = base.Driver()
    spec = {**SPEC, "checkpoint": checkpoint}
    if args.stage in ("all", "test"):
        result = driver.state["results"].setdefault("MOT20", {})
        result["checkpoint"] = driver.checkpoint_info("MOT20", spec)
        driver.save()
        post = driver.tracking_and_post("MOT20", spec)
        print(json.dumps({"post": str(post)}, indent=2), flush=True)
    if args.stage in ("all", "package"):
        post = Path(driver.state["results"]["MOT20"]["test"]["post_dir"])
        package = driver.package("MOT20", spec, post)
        print(json.dumps({"package": package}, indent=2), flush=True)
    if args.stage in ("all", "submit"):
        package = driver.state["results"]["MOT20"]["package"]
        try:
            payload = driver.submit("MOT20", spec, package)
            print(json.dumps({"id": payload.get("id"), "status": payload.get("status"),
                              "scores": payload.get("scores")}, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            driver.state["results"].setdefault("MOT20", {})["submit_error"] = str(exc)[:500]
            driver.save()
            print(f"submission not completed: {exc}", flush=True)
    if args.stage in ("all", "report"):
        report(driver, experiment)


if __name__ == "__main__":
    main()
