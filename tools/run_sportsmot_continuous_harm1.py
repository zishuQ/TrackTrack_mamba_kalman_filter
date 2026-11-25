#!/usr/bin/env python3
"""One missing 2x2 control: three-frame continuous supervision, harm=1.

Same D_e20 recipe, seed and 50-epoch schedule, stopped/evaluated only at e20.
Reuses the existing train45 state data and D/E/G1-3-harm2 outputs.
There is no formal training, test, postprocessing or submission stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sportsmot_g123_experiment as mechanisms
from paired_collection_io import atomic_json

EXP = ROOT / "outputs/agentguard/experiments/sportsmot_continuous_harm1_e20_20261124"
PREVIOUS = mechanisms.MECH_EXP
NAME = "G1-3-harm1"
SPEC = {"window_frames": 3, "window_supervision": "full", "augment": "none"}


def setup():
    mechanisms.MECH_EXP = EXP
    mechanisms.COMMON = {**mechanisms.COMMON, "motion_harm_weight": 1.0}
    abc = mechanisms.abc
    abc.EXP = EXP
    abc.LOGS = EXP / "logs"
    abc.EVALS = EXP / "evals"
    abc.GATES = EXP / "gate_targets"
    abc.STATE_PATH = EXP / "state.json"
    abc.PREFLIGHT_PATH = EXP / "preflight.json"
    return abc


def inspect_baseline(pre):
    import torch
    from paired_training_input import official_sequences
    expected = official_sequences("train", "SportsMOT")
    assert {p.name for p in mechanisms.DATA.iterdir() if (p / "samples.npz").is_file()} == expected
    assert pre["sequences"] == 45 and pre["samples"] == 305631
    assert pre["actual_steps"] == 6420 and pre["total_steps"] == 16050
    checkpoint = mechanisms.BASELINE_EXP / "runs/D/groups/appearance_row_column/checkpoints/epoch_020.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["training_config"]
    assert payload["epoch"] == 20 and payload["global_step"] == 6420
    for key, value in mechanisms.COMMON.items():
        if key in cfg:
            assert cfg[key] == value, (key, cfg[key], value)
    assert cfg["epochs"] == 50 and cfg["warmup_steps"] == 80
    assert set(cfg["sequence_names"]) == expected
    assert cfg["from_scratch"] and cfg["architecture_variant"] == "query_film_bilinear"
    assert cfg["appearance_proj_dim"] == 256 and cfg["dropout"] == .05
    return {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config": cfg}


def analyze(pre, baseline_info, state):
    import torch
    abc = mechanisms.abc
    entries = {
        "one_step_harm1_D": mechanisms.load_epoch(mechanisms.BASELINE_EXP, "D", 20),
        "continuous3_harm1": mechanisms.load_epoch(EXP, NAME, 20),
        "one_step_harm2_E": mechanisms.load_epoch(mechanisms.BASELINE_EXP, "E", 20),
        "continuous3_harm2": mechanisms.load_epoch(PREVIOUS, "G1-3", 20),
    }
    verified = mechanisms.verify(NAME, SPEC, pre)
    checkpoint = mechanisms.checkpoint(NAME, 20)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    group = mechanisms.group_dir(NAME)
    expected_init = None
    for line in (mechanisms.BASELINE_EXP / "runs/D/groups/appearance_row_column/training.log").read_text().splitlines():
        if line.startswith("verification: "):
            expected_init = json.loads(line.split(": ", 1)[1])["initial_parameter_sha256"]
    assert payload["training_config"]["initial_parameters_sha256"] == expected_init
    verification = [json.loads(line.split(": ", 1)[1]) for line in
                    (group / "training.log").read_text().splitlines()
                    if line.startswith("verification: ")][-1]
    assert verification["model_param_numel"] == verification["optimizer_param_numel"] == 1208220
    baseline_cfg = baseline_info["config"]
    for key in ("rank_weight", "column_weight", "rank_alpha", "rank_tau", "batch_size", "seed",
                "dropout", "lr", "weight_decay", "epochs", "total_steps", "warmup_steps", "taus"):
        assert payload["training_config"][key] == baseline_cfg[key], key
    assert payload["scheduler_state_dict"]["last_epoch"] == 6420
    assert verified["last"]["samples"] == 305631
    expected_val = set(abc.VAL_SEQS)
    for label, entry in entries.items():
        assert set(entry["per_sequence"]) == expected_val, label
        assert entry["gate_target"]["valid_events"] == 281687
    comparisons = {}
    for label, a, b in (("continuous_effect_harm1", "continuous3_harm1", "one_step_harm1_D"),
                        ("continuous_effect_harm2", "continuous3_harm2", "one_step_harm2_E"),
                        ("harm1_vs_harm2_continuous", "continuous3_harm1", "continuous3_harm2")):
        comparisons[label] = {
            "delta": mechanisms.delta(entries[a]["combined"], entries[b]["combined"]),
            "per_sequence": mechanisms.per_change(entries[a]["per_sequence"], entries[b]["per_sequence"]),
        }
    result = {"generated_at": abc.now(), "status": "awaiting_user_decision", "preflight": pre,
              "baseline": baseline_info, "entries": entries, "comparisons": comparisons,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "initial_parameters_sha256": expected_init, "training": verified,
              "interpretation": "Overall full-parameter training effects; motion-only HOTA contribution is unmeasured."}
    baseline_metrics = [json.loads(line) for line in
                        (mechanisms.BASELINE_EXP / "runs/D/groups/appearance_row_column/metrics.jsonl").read_text().splitlines()
                        if line.strip() and json.loads(line)["epoch"] <= 20]
    assert len(baseline_metrics) == 20
    baseline_seconds = sum(row["epoch_seconds"] for row in baseline_metrics)
    new_seconds = sum(row["epoch_seconds"] for row in verified["metrics"])
    result["cost_comparison"] = {"D_e20_seconds": baseline_seconds,
                                 "continuous3_harm1_seconds": new_seconds,
                                 "ratio": new_seconds / baseline_seconds}
    rows = ["# SportsMOT 三帧连续监督 × harm：补齐 e20 对照", "",
            "官方train45/val45，seed42、dropout0.05；相同初始化和全参数训练。50轮余弦计划，"
            "warmup80，在e20/6420步停止，最终LR约6.59e-5。新组仅完整评最终e20一次。", "",
            "| 配置 | HOTA | AssA | DetA | IDF1 | IDSW | Frag |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in entries.items():
        c = entry["combined"]
        rows.append(f"| {label} | " + " | ".join(f"{c[k]*100:.3f}" for k in ("HOTA", "AssA", "DetA", "IDF1"))
                    + f" | {c['IDSW']:.0f} | {c['Frag']:.0f} |")
    rows += ["", "## 同 harm 的连续监督效果", ""]
    for label in ("continuous_effect_harm1", "continuous_effect_harm2"):
        d = comparisons[label]["delta"]
        rows.append(f"- {label}: HOTA {d['HOTA']*100:+.3f}pp，AssA {d['AssA']*100:+.3f}pp，"
                    f"DetA {d['DetA']*100:+.3f}pp，IDSW {d['IDSW']:+.0f}。")
    h1_delta = comparisons["continuous_effect_harm1"]["delta"]["HOTA"] * 100
    new_hota = entries["continuous3_harm1"]["combined"]["HOTA"] * 100
    rows += ["", f"补齐的三帧连续harm1结果为{new_hota:.3f}，相对D_e20为{h1_delta:+.3f}pp。"]
    if h1_delta > 0:
        rows += ["本次单seed/e20下，连续监督在harm1上也高于对应一步基线；是否稳定仍需后续决定。"]
    else:
        rows += ["本次单seed/e20下，三帧连续未超过更强的一步harm1基线。harm2上的+0.223pp追回，"
                 "未在harm1对照中形成同方向提升。−0.028pp差值很小，当前按接近看待，不据此判断稳定退化。"]
    rows += ["", "## 固定 val 回放的运动门控诊断", "",
             "| 配置 | 有害gate | 有益gate | 总节省 | 总误伤 | 净额外DIoU loss/事件 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, entry in entries.items():
        t = entry["gate_target"]["new"]
        rows.append(f"| {label} | {t['harmful_gate_mean']:.5f} | {t['useful_gate_mean']:.5f} | "
                    f"{t['harmful_savings_total']:.3f} | {t['useful_damage_total']:.3f} | {t['net_loss_change']:+.6g} |")
    rows += ["", "正的净额外loss表示固定回放上比完整写入更差。这是同一批val有效事件的一步代理指标。", "",
             "## 在线双门控与逐序列变化", "",
             "gate均值为45个序列各自平均gate的等权均值。", "",
             "| 配置 | motion gate均值 | appearance gate均值 |",
             "|---|---:|---:|"]
    for label, entry in entries.items():
        g = entry["gates"]
        rows.append(f"| {label} | {g['motion_mean']:.5f} | {g['appearance_mean']:.5f} |")
    changes = comparisons["continuous_effect_harm1"]["per_sequence"]
    rows += ["", f"新组相对D_e20，HOTA改善超过0.5pp的序列{changes['improved']}条，"
             f"下降超过0.5pp的序列{changes['degraded']}条，其余{changes['unchanged']}条在阈值内。", "",
             "| 新组相对D_e20 | 序列 | ΔHOTA/pp |", "|---|---|---:|"]
    for kind in ("best", "worst"):
        for sequence, change in changes[kind][:3]:
            rows.append(f"| {kind} | {sequence} | {change*100:+.3f} |")
    rows += ["",
             "## 训练和解释范围", "",
             f"新组20轮训练计算共{sum(r['epoch_seconds'] for r in verified['metrics']):.1f}秒；"
             f"CUDA峰值{verified['last'].get('cuda_peak_allocated_mb')}MiB。", "",
             f"相同harm1下，D_e20为{baseline_seconds/20:.2f}秒/轮，三帧连续为{new_seconds/20:.2f}秒/轮，"
             f"训练计算耗时为{new_seconds/baseline_seconds:.2f}倍。两组优化器步数和锚点数相同，模型调用及递推次数不同。", "",
             "连续监督和harm改变的是整套全参数训练方案。运动与外观gate、共享表征都可能变化；"
             "指标改善不能直接归因于运动分支。有害事件识别能力结合gate分离、总误伤/节省一并解释。"
             "单seed结果不包含重复训练稳定性结论。", "",
             "## 增强诊断与结束状态", "",
             "小样本train诊断见augmentation_diagnostic.md/.json。诊断分别隔离坐标扰动与模拟漏检，"
             "训练配方未据此自动修改。", "",
             "本轮到此停止，等待用户决定；未启动延长训练、正式trainval、test/post或提交。"]
    diagnostic_path = EXP / "augmentation_diagnostic.json"
    if diagnostic_path.is_file():
        diagnostic = json.loads(diagnostic_path.read_text())
        fixed = diagnostic["summary"]["frozen_prior_events"]
        result["augmentation_diagnostic"] = {
            "path": str(diagnostic_path), "unique_anchors": diagnostic["unique_anchors"],
            "original": fixed["original"], "box_noise_only": fixed["box_noise_only"],
            "missing_only": fixed["missing_only"],
        }
        rows += ["", "诊断摘要：检测中心误差从"
                 f"{fixed['original']['center_error_px']['mean']:.3f}px变为"
                 f"{fixed['box_noise_only']['center_error_px']['mean']:.3f}px，固定先验有害率从"
                 f"{fixed['original']['harmful_fraction_among_observed_valid']*100:.2f}%变为"
                 f"{fixed['box_noise_only']['harmful_fraction_among_observed_valid']*100:.2f}%。"
                 "保留检测的分数完全不变；这是观测分布变化的证据，尚未证明保守门控的因果机制。"]
    atomic_json(EXP / "report.json", result)
    (EXP / "report.md").write_text("\n".join(rows) + "\n")
    state["completion"] = {"status": "awaiting_user_decision", "report": str(EXP / "report.md"),
                           "checkpoint_sha256": result["checkpoint_sha256"], "at": abc.now()}
    abc.save_state(state)
    print("\n".join(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "train", "eval", "analyze", "plan"), default="all")
    args = parser.parse_args()
    abc = setup()
    if args.stage == "plan":
        print(json.dumps({"experiment": str(EXP), "name": NAME, "spec": SPEC,
                          "common": mechanisms.COMMON,
                          "command": list(map(str, mechanisms.train_command(NAME, SPEC, {})))}, indent=2))
        return
    EXP.mkdir(parents=True, exist_ok=True)
    abc.LOGS.mkdir(parents=True, exist_ok=True)
    pre = mechanisms.data_steps()
    baseline_info = inspect_baseline(pre)
    atomic_json(EXP / "preflight.json", {**pre, "baseline": baseline_info["path"],
                                        "window_implementation": "packed_replay_v2"})
    state = abc.load_state()
    if args.stage in ("all", "train"):
        mechanisms.train_run(NAME, SPEC, pre, state)
        verified = mechanisms.verify(NAME, SPEC, pre)
        state["runs"].setdefault(NAME, {})["verification"] = {
            "epoch": verified["last"]["epoch"], "global_step": verified["last"]["global_step"],
            "reload_ok": verified["reload_ok"]}
        abc.save_state(state)
    if args.stage in ("all", "eval"):
        mechanisms.verify(NAME, SPEC, pre)
        abc.eval_run(NAME, 20, state)
        abc.gate_run(NAME, 20, state)
    if args.stage in ("all", "analyze"):
        analyze(pre, baseline_info, state)


if __name__ == "__main__":
    main()
