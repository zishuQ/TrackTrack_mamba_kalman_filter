#!/usr/bin/env python3
"""Reports for the SHUTrack architecture night run (ablation and final)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/shang/workspace/TrackTrack")
EXP = ROOT / "outputs/agentguard/experiments/shutrack_architecture_night_20261120"
OLD_MOT20 = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_mot20_20261120"
OLD_SM = ROOT / "outputs/agentguard/experiments/shutrack_d_proj256_sportsmot_20261120"
MOT20_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20/data"
MOT17_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17/data"
SM_DATA = ROOT / "outputs/agentguard/experiments/shutrack_bd_sportsmot_20261120/data"
METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")
FORMAT = {"HOTA": ".4f", "MOTA": ".4f", "IDF1": ".4f", "DetA": ".4f", "AssA": ".4f",
          "IDSW": ".0f", "Frag": ".0f"}


def load(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary_rows(path):
    payload = load(path)
    return payload.get("combined") or {}, payload.get("per_sequence") or {}


def table(name, row):
    return "| " + name + " | " + " | ".join(format(row.get(k, 0), FORMAT[k]) for k in METRICS) + " |"


def delta(name, new, old):
    cells = []
    for key in METRICS:
        value = new[key] - old[key]
        cells.append(f"{value * 100:+.2f} pp" if key not in ("IDSW", "Frag") else f"{value:+.0f}")
    return "| " + name + " | " + " | ".join(cells) + " |"


def train_stats(root):
    root = Path(root)
    complete_path = root / "complete.json"
    config_path = root / "config.json"
    metrics = [r for r in read_jsonl(root / "metrics.jsonl") if "epoch" in r]
    complete = load(complete_path) if complete_path.is_file() else {}
    config = load(config_path) if config_path.is_file() else {}
    tail = metrics[-20:]
    ms_step = None
    if tail:
        per_step = [r["epoch_seconds"] / max(1, r["batches"]) for r in tail]
        ms_step = sum(per_step) / len(per_step) * 1000
    return {"complete": complete, "config": config, "metrics": metrics,
            "ms_step": ms_step, "epochs": len(metrics)}


def eval_speed(state, task):
    entry = (state.get("tasks") or {}).get(task) or {}
    log = entry.get("log")
    if not log or not Path(log).is_file():
        return None
    values = []
    for line in Path(log).read_text().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return values[-1] if values else None


def dataset_samples(root):
    total = 0
    per = {}
    for summary in sorted(Path(root).glob("*/summary.json")):
        payload = load(summary)
        per[summary.parent.name] = payload.get("samples", 0)
        total += payload.get("samples", 0)
    return total, per


def ablation_report():
    state = load(EXP / "TASK_STATE.json") if (EXP / "TASK_STATE.json").is_file() else {}
    sources = {
        "MOT20": {
            "old_eval": OLD_MOT20 / "evals/val_d256/summary.json",
            "new_eval": EXP / "ablation/MOT20/evals/new/summary.json",
            "old_train": OLD_MOT20 / "groups/appearance_row_column",
            "new_train": EXP / "ablation/MOT20/groups/appearance_row_column",
            "old_task": None, "new_task": "ablation_mot20_new_eval",
            "val": "MOT20-01, MOT20-05",
        },
        "SportsMOT": {
            "old_eval": OLD_SM / "evals/val_d256/summary.json",
            "new_eval": EXP / "ablation/SportsMOT/evals/new/summary.json",
            "old_train": OLD_SM / "groups/appearance_row_column",
            "new_train": EXP / "ablation/SportsMOT/groups/appearance_row_column",
            "old_task": None, "new_task": "ablation_sportsmot_new_eval",
            "val": "official val 45 sequences",
        },
        "MOT17": {
            "old_eval": EXP / "ablation/MOT17/evals/old/summary.json",
            "new_eval": EXP / "ablation/MOT17/evals/new/summary.json",
            "old_train": EXP / "ablation/MOT17/groups_old/appearance_row_column",
            "new_train": EXP / "ablation/MOT17/groups_new/appearance_row_column",
            "old_task": "ablation_mot17_old_eval", "new_task": "ablation_mot17_new_eval",
            "val": "MOT17-09-FRCNN, MOT17-10-FRCNN",
        },
    }
    lines = ["# SHUTrack OLD-D-256 vs NEW-D-256 架构消融", ""]
    lines.append("D 监督与数据固定；唯一变量为模型架构（legacy vs query_film_bilinear）。")
    lines.append("表中指标为 0–1 比例；Δ 为百分点（pp），IDSW/Frag 为计数差。")
    lines.append("")
    for dataset, source in sources.items():
        old_comb, old_per = summary_rows(source["old_eval"])
        new_comb, new_per = summary_rows(source["new_eval"])
        old_train = train_stats(source["old_train"])
        new_train = train_stats(source["new_train"])
        lines.append(f"## {dataset}（验证：{source['val']}）")
        lines.append("")
        lines.append("| 组 | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(table("OLD-D-256", old_comb))
        lines.append(table("NEW-D-256", new_comb))
        lines.append(delta("NEW − OLD", new_comb, old_comb))
        lines.append("")
        lines.append("| 参数量 OLD/NEW | 训练轮数 OLD/NEW | 稳定 ms/step OLD/NEW | 训练耗时 OLD/NEW (s) | 评估速度 OLD/NEW (fps) |")
        lines.append("|---|---|---|---|---|")
        old_params = old_train["config"].get("appearance_proj_dim")
        speed_old = eval_speed(state, source["old_task"]) if source["old_task"] else None
        speed_new = eval_speed(state, source["new_task"])
        lines.append(
            f"| 1,324,194 / 1,208,220 | {old_train['epochs']} / {new_train['epochs']} | "
            f"{old_train['ms_step']:.1f} / {new_train['ms_step']:.1f} | "
            f"{old_train['complete'].get('seconds', 0):.0f} / {new_train['complete'].get('seconds', 0):.0f} | "
            f"{speed_old if speed_old else '历史日志'} / {speed_new if speed_new else '—'} |"
        )
        lines.append("")
        lines.append("| 序列 | HOTA OLD | HOTA NEW | ΔHOTA (pp) | IDF1 OLD | IDF1 NEW | AssA OLD | AssA NEW | IDSW OLD/NEW | Frag OLD/NEW |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for sequence in sorted(set(old_per) & set(new_per)):
            o, n = old_per[sequence], new_per[sequence]
            rows.append((sequence, o, n, n["HOTA"] - o["HOTA"]))
        rows.sort(key=lambda item: item[3], reverse=True)
        for sequence, o, n, diff in rows:
            lines.append(
                f"| {sequence} | {o['HOTA']:.4f} | {n['HOTA']:.4f} | {diff * 100:+.2f} | "
                f"{o['IDF1']:.4f} | {n['IDF1']:.4f} | {o['AssA']:.4f} | {n['AssA']:.4f} | "
                f"{int(round(o['IDSW']))}/{int(round(n['IDSW']))} | "
                f"{int(round(o['Frag']))}/{int(round(n['Frag']))} |"
            )
        lines.append("")
        lines.append("- 单 seed、既定划分；不宣称统计显著。评估速度为跟踪总帧数/总耗时；OLD 数值来自历史日志，非严格同条件测速。")
        lines.append("")
    path = EXP / "reports/ABLATION_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("saved", path)


def final_report():
    recipe = load(EXP / "FINAL_RECIPE.json") if (EXP / "FINAL_RECIPE.json").is_file() else {}
    results = load(EXP / "submissions/SUBMISSION_RESULTS.json") if (EXP / "submissions/SUBMISSION_RESULTS.json").is_file() else {}
    lines = ["# SHUTrack NEW 架构正式训练与提交报告", ""]
    data_roots = {"MOT20": MOT20_DATA, "MOT17": MOT17_DATA, "SportsMOT": SM_DATA}
    for dataset in ("MOT20", "MOT17", "SportsMOT"):
        root = EXP / "final" / dataset / "groups/appearance_row_column"
        stats = train_stats(root)
        samples, per_sequence = dataset_samples(data_roots[dataset])
        config = stats["config"]
        complete = stats["complete"]
        entry = results.get(dataset) or {}
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"- 数据：`{data_roots[dataset]}`，{len(per_sequence)} 条序列，{samples:,} 样本")
        lines.append(
            f"- 配置：架构={config.get('architecture_variant')}，proj={config.get('appearance_proj_dim')}，"
            f"epochs={config.get('epochs')}，steps={config.get('total_steps')}，"
            f"shards={config.get('shards')}，sampling={config.get('sequence_sampling')}，"
            f"lr={config.get('lr')}，init={config.get('init_mode')}"
        )
        lines.append(
            f"- 训练：{complete.get('seconds', 0):.0f} s，末 20 轮 {stats['ms_step']:.1f} ms/step，"
            f"checkpoint `{root / 'checkpoints/last.pt'}`"
        )
        if entry:
            lines.append(
                f"- test/打包：zip `{entry.get('package', {}).get('zip')}`，"
                f"sha256 `{entry.get('package', {}).get('sha256')}`"
            )
            submission = entry.get("submission") or {}
            lines.append(
                f"- Codabench：submission id={submission.get('submission_id')}，"
                f"status={submission.get('status')}，scores={submission.get('scores')}"
            )
            if entry.get("error"):
                lines.append(f"- 错误：{entry['error']}")
        lines.append("")
    lines.append("## 配方出处与已知差异")
    lines.append("")
    lines.append(f"- FINAL_RECIPE.json：`{EXP / 'FINAL_RECIPE.json'}`")
    for item in recipe.get("known_differences", []):
        lines.append(f"- {item}")
    lines.append("")
    path = EXP / "reports/FINAL_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("saved", path)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("ablation", "final"):
        raise SystemExit("usage: arch_night_reports.py ablation|final")
    if sys.argv[1] == "ablation":
        ablation_report()
    else:
        final_report()


if __name__ == "__main__":
    main()
