#!/usr/bin/env python3
"""Audit and interpret existing SportsMOT e20 artifacts without running models.

Reads completed checkpoints, metrics and val outputs. Never starts training,
tracking, gate inference, formal trainval or submissions. Existing artifacts
remain intact; writes completion_audit.json and analysis_report.md only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sportsmot_g123_experiment as driver
from shutrack.training.checkpoint import load_shutrack_checkpoint
from shutrack.training.motion_targets import SOURCE_KEYS, source_fingerprint


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verification_log(group):
    entries = [json.loads(line.split(": ", 1)[1])
               for line in (group / "training.log").read_text().splitlines()
               if line.startswith("verification: ")]
    assert entries, f"Missing parameter verification: {group}"
    result = entries[-1]
    assert result["model_param_numel"] == result["optimizer_param_numel"] == 1208220
    return result


def online_gates(path, expected):
    rows = read_lines(path)
    assert len(rows) == 45 and {r["sequence"] for r in rows} == expected
    result = {}
    for channel in ("motion", "appearance"):
        means = [r[f"shutrack_avg_{channel}_gate"] for r in rows]
        counts = [r[f"shutrack_{channel}_gate_observations"] for r in rows]
        assert all(count > 0 for count in counts)
        result[channel] = {
            "event_count": int(sum(counts)),
            "event_weighted_mean": sum(m * n for m, n in zip(means, counts)) / sum(counts),
            "sequence_mean": sum(means) / len(means),
            "event_min": min(r[f"shutrack_min_{channel}_gate"] for r in rows),
            "event_max": max(r[f"shutrack_max_{channel}_gate"] for r in rows),
        }
    return rows, result


def comparison(a, b):
    changes = {}
    for sequence in a["per_sequence"]:
        changes[sequence] = {
            key: (a["per_sequence"][sequence][key] - b["per_sequence"][sequence][key])
                 * (1 if key in ("IDSW", "Frag") else 100)
            for key in driver.SCORE_KEYS
        }
    ordered = sorted(changes, key=lambda name: changes[name]["HOTA"])
    return {
        "combined_delta": driver.delta(a["combined"], b["combined"]),
        "improved_over_0_5pp": sum(v["HOTA"] > .5 for v in changes.values()),
        "degraded_over_0_5pp": sum(v["HOTA"] < -.5 for v in changes.values()),
        "worst_five": ordered[:5], "best_five": ordered[-5:][::-1],
        "per_sequence_delta_pp_or_count": changes,
    }


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
                "| " + " | ".join(map(str, row)) + " |" for row in rows]


def main():
    torch.set_num_threads(4)
    exp = driver.MECH_EXP
    pre = read(exp / "preflight.json")
    assert driver.data_steps() == {k: pre[k] for k in driver.data_steps()}
    train = driver.abc.official_sequences("train", "SportsMOT")
    val = driver.abc.official_sequences("val", "SportsMOT")
    assert len(train) == len(val) == 45 and not train & val
    assert {p.name for p in driver.DATA.iterdir() if (p / "samples.npz").exists()} == train
    state = driver.abc.load_state()
    baseline_group = driver.BASELINE_EXP / "runs/E/groups/appearance_row_column"
    baseline_checkpoint = baseline_group / "checkpoints/epoch_020.pt"
    baseline_payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    baseline_config = baseline_payload["training_config"]
    baseline_log = verification_log(baseline_group)
    baseline_init = baseline_log["initial_parameter_sha256"]
    baseline = driver.load_epoch(driver.BASELINE_EXP, "E", 20)
    baseline["per_sequence"] = read(driver.BASELINE_EXP / "evals/E_e020/summary.json")["per_sequence"]
    assert baseline_payload["epoch"] == 20 and baseline_payload["global_step"] == 6420
    assert set(baseline_config["sequence_names"]) == train
    assert set(baseline["per_sequence"]) == val
    _, baseline_online = online_gates(driver.BASELINE_EXP / "evals/E_e020/gates.jsonl", val)

    # Verify reuse of the same one-step supervision rows despite the new state dumps.
    for sequence in sorted(train):
        assert read(driver.DATA / sequence / "summary.json")["complete"]
        with np.load(driver.DATA / sequence / "samples.npz", allow_pickle=False) as new, \
             np.load(Path(baseline_config["data_root"]) / sequence / "samples.npz",
                     allow_pickle=False) as old:
            for key in ("frame", "track_id", "detection_index", "rank_rows", "column_det"):
                np.testing.assert_array_equal(new[key], old[key], err_msg=f"{sequence}: {key}")
            # State-rich collection can reorder the event table. Canonicalize
            # its storage offsets to sample row ids before comparing sources.
            canonical = np.arange(len(new["frame"]), dtype=np.int64)
            def source(samples):
                return {key: canonical if key == "current_event_index" else samples[key]
                        for key in SOURCE_KEYS}
            assert source_fingerprint(source(new)) == source_fingerprint(source(old)), sequence
            for field in ("event_evidence", "event_detection_index", "event_has_detection"):
                left = np.load(driver.DATA / sequence / f"{field}.npy", mmap_mode="r")
                right = np.load(Path(baseline_config["data_root"]) / sequence / f"{field}.npy",
                                mmap_mode="r")
                np.testing.assert_array_equal(left[new["current_event_index"]],
                                              right[old["current_event_index"]],
                                              err_msg=f"{sequence}: aligned {field}")

    result = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "verification_status": "ANALYZED",
        "verification_method": "artifact integrity and numerical recomputation; no model reruns",
        "phase": "awaiting_user_decision", "preflight": pre,
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": digest(baseline_checkpoint),
        "same_training_motion_source_rows": True,
        "baseline": {"combined": baseline["combined"], "online_gates": baseline_online,
                     "motion_target": baseline["gate_target"]["new"]},
        "strong_baseline_D_e20": read(driver.BASELINE_EXP / "evals/D_e020/summary.json")["combined"],
        "runs": {}, "comparisons": {},
    }
    current = {}
    for name, spec in driver.RUNS.items():
        verified = driver.verify(name, spec, pre)
        group = driver.group_dir(name)
        checkpoint = driver.checkpoint(name, 20)
        model, payload = load_shutrack_checkpoint(checkpoint, map_location="cpu")
        last = torch.load(group / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        cfg = payload["training_config"]
        assert payload["epoch"] == last["epoch"] == 20
        assert payload["global_step"] == last["global_step"] == 6420
        assert all(torch.equal(v, last["model_state_dict"][k])
                   for k, v in payload["model_state_dict"].items())
        assert sum(p.numel() for p in model.parameters()) == 1208220
        assert cfg["initial_parameters_sha256"] == baseline_init
        for key in ("architecture_variant", "appearance_proj_dim", "dropout", "batch_size",
                    "lr", "weight_decay", "seed", "shards", "rank_weight", "rank_alpha",
                    "rank_tau", "column_weight", "column_hinge", "motion_harm_weight",
                    "epochs", "total_steps", "warmup_steps", "taus", "preload_features"):
            assert cfg[key] == baseline_config[key], (name, key)
        assert cfg["micro_batch"] == 1024 and set(cfg["sequence_names"]) == train
        assert payload["scheduler_state_dict"]["last_epoch"] == 6420
        assert math.isclose(payload["optimizer_state_dict"]["param_groups"][0]["lr"],
                            pre["lr_at_stop"], abs_tol=1e-12)
        verification_log(group)
        metrics = verified["metrics"]
        assert [r["epoch"] for r in metrics] == list(range(1, 21))
        assert all(r["samples"] == 305631 and r["micro_batches"] == 321
                   and r["global_step"] == 321 * r["epoch"] for r in metrics)
        assert all(math.isfinite(r[k]) for r in metrics
                   for k in ("loss", "motion_loss", "appearance_loss"))
        ev = exp / "evals" / f"{name}_e020"
        summary = read(ev / "summary.json")
        assert set(summary["per_sequence"]) == val
        raw_dirs = [p for p in ev.iterdir() if p.is_dir() and not p.name.endswith("_post")]
        assert len(raw_dirs) == 1 and {p.stem for p in raw_dirs[0].glob("*.txt")} == val
        gates, online = online_gates(ev / "gates.jsonl", val)
        assert all(Path(row["shutrack_checkpoint"]).resolve() == checkpoint.resolve() for row in gates)
        log = (exp / "logs" / f"eval_{name}_e020.log").read_text()
        assert log.count("===== ") == 1, f"Unexpected repeated full val: {name}"
        target = read(exp / "gate_targets" / f"{name}_e020.json")
        assert target["valid_events"] == 281687
        t = target["new"]
        assert t["harmful_count"] + t["useful_count"] == target["valid_events"]
        net = t["useful_damage_total"] - t["harmful_savings_total"]
        assert math.isclose(net / target["valid_events"], t["net_loss_change"], abs_tol=1e-7)
        for key in ("train",):
            assert state["runs"][name][key]["status"] == "complete"
        for key in ("evals", "gates"):
            assert state["runs"][name][key]["e020"]["status"] == "complete"
        current[name] = summary
        result["runs"][name] = {
            "checkpoint": str(checkpoint), "sha256": digest(checkpoint),
            "epoch": 20, "global_step": 6420, "initial_sha256": baseline_init,
            "parameters": 1208220, "val_sequences": 45, "one_full_val_invocation": True,
            "combined": summary["combined"], "online_gates": online,
            "motion_target": t, "net_excess_loss_total": net,
            "damage_to_savings_ratio": t["useful_damage_total"] / t["harmful_savings_total"],
            "epoch_seconds_mean": sum(r["epoch_seconds"] for r in metrics) / 20,
            "epoch_seconds_total": sum(r["epoch_seconds"] for r in metrics),
            "rss_peak_gib": max(r["rss_peak_gb"] for r in metrics),
            "cuda_peak_mib": max(r["cuda_peak_allocated_mb"] for r in metrics),
            "sample_anchors_total": sum(r["samples"] for r in metrics),
            "motion_transitions_total": sum(r["transitions"] for r in metrics),
            "comparison_vs_baseline": comparison(summary, baseline),
            "gradient_probe_e20": {k: metrics[-1][k] for k in
                ("grad_norm_motion_shared", "grad_norm_appearance_shared")},
        }
    for a, b in (("G1-5", "G1-3"), ("G3-3", "G1-3"), ("G3-3", "G2-3")):
        result["comparisons"][f"{a}_vs_{b}"] = comparison(current[a], current[b])
    result["fallacy_scan"] = {
        "coverage": "11/11",
        "aggregation": "COMBINED HOTA preserved; per-sequence changes are descriptive, not contributions",
        "ecological": "No inference from sequence averages to individual harmful events",
        "selection": "Official val45 complete; motion proxy restricted to fixed GT-valid matched events",
        "collider": "No post-treatment adjustment or filtered winning sequences",
        "base_rate": "30938 harmful vs 250749 useful; total savings and damage both reported",
        "regression_to_mean": "No repeated-seed stability or persistence claims",
        "survivorship": "All four groups and all 45 val sequences included",
        "look_elsewhere": "All planned groups reported; winner treated as exploratory",
        "forking_paths": "Protocol fixed to e20; no intermediate val selection",
        "causal": "Overall model comparison; motion-only causal contribution unmeasured",
        "reverse_causality": "Training precedes val; proxy differences not asserted as causes of HOTA",
    }
    (exp / "completion_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (exp / "analysis_report.md").write_text(render(result, current, baseline) + "\n")
    print(json.dumps({"phase": result["phase"], "runs": list(result["runs"]),
                      "report": str(exp / "analysis_report.md"),
                      "audit": str(exp / "completion_audit.json")}, ensure_ascii=False, indent=2))


def render(result, current, baseline):
    runs = result["runs"]
    lines = ["# SportsMOT 四组 e20：接手续验与机制分析", "",
             "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: validate",
             f"- Origin Date: {result['generated_at']}",
             "- Verification Status: ANALYZED（核验现有产物，未重跑模型）",
             "- Version Label: sportsmot_e20_completion_v1", "",
             "## 结果与当前状态", "",
             "四组 e20 训练、最终 val45 raw 和门控分析均已完成。复用既有产物，接手后未新增训练、"
             "val 或门控推理。宿主机检查未发现正在运行的相关重任务。当前阶段为 awaiting_user_decision。", "",
             "G1-3 在本轮四组机制实验中最高，HOTA 83.517，相对同harm2的 E_e20 +0.223pp。"
             "5 帧较 3 帧慢约 70%，HOTA 低 0.199pp；加噪两组的 HOTA 均低于 E_e20。", "",
             f"已有更强的同预算一步harm1基线D_e20，HOTA {result['strong_baseline_D_e20']['HOTA']*100:.3f}，"
             "高于G1-3-harm2。三帧连续harm1已单独立项补测，输出在sportsmot_continuous_harm1_e20_20261124。"
             "本报告不把harm2配置上的追回等同于超过D_e20，也不把整套训练的变化归因于运动分支。", "",
             "## 协议和完整性", "",
             "官方 train45/val45；四组同 seed42、同初始参数哈希、query_film_bilinear、"
             "1,208,220 参数且全部进入优化器；dropout0.05、harm2、batch=micro_batch1024、"
             "shards1、CPU 预载、rank18/column50/tau0.1/alpha0.95。", "",
             "每组均有20条连续轮次记录，每轮305,631锚点、321步；合计6,112,620锚点和6420步。"
             "50轮计划为16050步，warmup80，e20末LR=6.5899171e-5。最终快照与last.pt权重一致，"
             "调度器为6420步；45条raw输出、summary序列集合、gate统计与checkpoint指向一致。"
             "每组日志中完整val命令恰好一次。G1-3门控分析有一次归一化尾差断言失败后重启，"
             "最终分析已完整覆盖45序列；失败发生在分析循环前，没有重复完整val。", "",
             "与 E_e20 核对了共同训练配置、初始化哈希及45条训练序列的运动监督源指纹。"
             "新旧缓存的事件表存储顺序不同，按样本绑定对齐后当前运动证据、检测索引和mask完全一致；"
             "源指纹用样本行号替代事件表偏移后比较。"
             "本轮窗口状态数据来自既有完整缓存，未重新采集。", "",
             "## 同预算最终 val", ""]
    rows = []
    for name, entry in [("E_e20", baseline)] + list(current.items()):
        c = entry["combined"]
        rows.append([name] + [f"{c[k]*100:.3f}" for k in ("HOTA", "AssA", "DetA", "IDF1")]
                    + [int(c["IDSW"]), int(c["Frag"]),
                       f"{(c['HOTA']-baseline['combined']['HOTA'])*100:+.3f}"])
    lines += table(["组", "HOTA", "AssA", "DetA", "IDF1", "IDSW", "Frag", "ΔHOTA/pp"], rows)
    lines += ["", "E_e100的83.680为100轮历史参考；其预算不同，独立保留。", "",
              "## 三项机制对照", "",
              "连续3帧：G1-3相对E_e20，HOTA +0.223pp、AssA +0.263pp、DetA +0.170pp，"
              "IDSW少11次。运动在线均值更接近1；这组整体结果有小幅收益，"
              "运动抑制带来的独立HOTA贡献尚未测量。", "",
              "窗口长度：G1-5相对G1-3，HOTA −0.199pp、AssA −0.189pp、DetA −0.209pp，"
              "IDSW多7次。增加两步递推提高训练成本，本次20轮结果没有取得整体收益。", "",
              "单步加噪：G2-3用最多两步固定全写入预热，只在锚点调用模型并监督。相对E_e20，"
              "HOTA −0.324pp、DetA −0.386pp。它同时重建带噪历史/状态，本次对照反映整套增强接入的效果。", "",
              "连续3帧加噪：G3-3相对G1-3，HOTA −0.737pp、AssA −0.583pp、DetA −0.911pp；"
              "相对G2-3，HOTA −0.189pp，IDF1却 +0.153pp，显示不同指标方向并不完全一致。", "",
              "## 运动门控：总节省和总误伤", "",
              "固定val回放的有效运动事件281,687个，其中有害30,938（10.98%）、有益250,749（89.02%）。"
              "误差以DIoU loss表示。总节省为有害事件完整写入损失减门控损失；总误伤为有益事件门控损失减完整写入损失。"
              "净额外损失=(总误伤−总节省)/281687，正值表示比完整写入更差。", ""]
    rows = []
    targets = [("E_e20", result["baseline"]["motion_target"])] + [
        (name, r["motion_target"]) for name, r in runs.items()]
    for name, t in targets:
        rows.append([name, f"{t['harmful_gate_mean']:.4f}", f"{t['useful_gate_mean']:.4f}",
                     f"{t['useful_gate_mean']-t['harmful_gate_mean']:.4f}",
                     f"{t['harmful_savings_total']:.3f}", f"{t['useful_damage_total']:.3f}",
                     f"{t['net_loss_change']:+.6f}"])
    lines += table(["组", "有害gate", "有益gate", "分离度", "总节省", "总误伤", "净额外损失/事件"], rows)
    lines += ["", "G2-3/G3-3的分离度增加到0.203/0.228，但总误伤约为总节省的4.76/5.17倍。"
              "有益事件的数量约为有害事件8.1倍，较小的逐事件误伤仍会累积成明显净损失。"
              "这一结果与加噪组DetA回退方向一致；该代理诊断不能单独确定在线HOTA变化的因果来源。", "",
              "G1-3的总节省8.301、总误伤29.475，整体接近全写入；"
              "分离度仅0.0012。因此本轮尚未显示连续监督解决了运动门控趋近1的问题。", "",
              "## 在线gate统计口径", "",
              "下表均值按实际应用gate的匹配事件数加权。min/max为事件级极值；原report.md中的"
              "min/mean/max取45个序列各自的平均gate，不能当作事件级极值。", ""]
    rows = []
    for name, gates in [("E_e20", result["baseline"]["online_gates"])] + [
            (name, r["online_gates"]) for name, r in runs.items()]:
        rows.append([name, f"{gates['motion']['event_weighted_mean']:.5f}",
                     f"{gates['motion']['event_min']:.5f}/{gates['motion']['event_max']:.5f}",
                     f"{gates['appearance']['event_weighted_mean']:.5f}"])
    lines += table(["组", "motion事件加权均值", "motion事件min/max", "appearance事件加权均值"], rows)
    lines += ["", "## 逐序列变化与大幅翻转", "",
              "改善/退化阈值为±0.5pp，其余归入阈值内；这不表示指标完全相同。"
              "COMBINED HOTA保留TrackEval原始聚合值，不用45条序列简单平均代替。", ""]
    lines += table(["相对E_e20", "改善>0.5pp", "退化>0.5pp", "阈值内"], [
        [name, r["comparison_vs_baseline"]["improved_over_0_5pp"],
         r["comparison_vs_baseline"]["degraded_over_0_5pp"],
         45-r["comparison_vs_baseline"]["improved_over_0_5pp"]
           -r["comparison_vs_baseline"]["degraded_over_0_5pp"]] for name, r in runs.items()])
    focus = ("v_9MHDmAMxO5I_c006", "v_9MHDmAMxO5I_c009", "v_0kUtTtmLaJA_c004",
             "v_0kUtTtmLaJA_c005", "v_0kUtTtmLaJA_c006")
    lines += [""] + table(["序列HOTA", "E_e20", *current], [
        [seq] + [f"{entry['per_sequence'][seq]['HOTA']*100:.3f}"
                 for entry in [baseline, *current.values()]] for seq in focus])
    lines += ["", "G1-5相对G1-3在v_9MHDmAMxO5I_c006提高4.755pp，但在同视频c009降低4.752pp。"
              "前者AssA +9.427pp、DetA −1.661pp；后者AssA −8.531pp、DetA约−0.020pp。"
              "明显翻转主要体现为序列关联质量变化，不能仅凭总IDSW数或平均gate解释。", "",
              "加噪两组在v_0kUtTtmLaJA_c006均下降约6.4–6.5pp，同时在v_9MHDmAMxO5I_c006"
              "提高约5.6–5.9pp。G2-3/G3-3分别有15/18条序列下降超过0.5pp，"
              "表明回退覆盖多条序列。未启动额外可视化推理；异常帧级原因尚待后续任务定位。", "",
              "## 计算成本", ""]
    lines += table(["组", "平均秒/轮", "20轮计算秒", "CUDA峰值/MiB", "RSS峰值/GiB", "日志transitions/20轮"], [
        [name, f"{r['epoch_seconds_mean']:.2f}", f"{r['epoch_seconds_total']:.1f}",
         r["cuda_peak_mib"], r["rss_peak_gib"], r["motion_transitions_total"]] for name, r in runs.items()])
    a, b = runs["G1-5"], runs["G1-3"]
    lines += ["", f"G1-5轮时为G1-3的{a['epoch_seconds_mean']/b['epoch_seconds_mean']:.2f}倍，"
              f"20轮多{a['epoch_seconds_total']-b['epoch_seconds_total']:.1f}秒（约9.1分钟），"
              f"CUDA峰值多{a['cuda_peak_mib']-b['cuda_peak_mib']:.1f}MiB。"
              "各组优化器步数/锚点预算相同，递推次数和计算量不同。"
              "transitions在full组统计窗口有效位置，G2-3只统计锚点；"
              "G2-3的最多两步KF预热不计入此列，因此该列不是KF实际操作总次数。", "",
              "## 解释范围与统计核查", "",
              "单seed探索性比较，未计算p值或置信区间；重复训练的稳定性尚未知。"
              "本次仅核验已有数值/产物，没有重新训练或重新推理，因此Verification Status为ANALYZED。"
              "窗口内KF/EMA/证据和mask动态递推，离散关联候选仍为固定回放。"
              "外观rank/column只在锚点计算，3帧/5帧表示状态递推长度，HCE历史容量仍为5事件。", "",
              "统计推断误区核查11/11：聚合差异、生态推断、样本筛选、碰撞变量、基率忽略、"
              "均值回归、幸存偏差、多重探索、分析分支选择、相关/因果混淆、反向因果。"
              "重点限制为单seed、val选优、固定回放有效事件筛选，以及代理指标与在线指标的差别；"
              "逐项记录见completion_audit.json。", "",
              "## 下一步边界与可复查文件", "",
              "本轮已结束，等待用户决定。后续延长训练、正式trainval、test/post和提交均未启动。", "",
              "- 既有主报告：report.md / report.json",
              "- 接手续验和全量逐序列差值：completion_audit.json",
              "- 训练/评估状态：state.json",
              "- 复查命令：.venv/bin/python tools/analyze_sportsmot_mechanisms_e20.py",
              "- 原始证据：runs/*/groups/appearance_row_column、evals/*_e020、gate_targets/*_e020.json。"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
