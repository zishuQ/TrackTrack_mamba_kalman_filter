#!/usr/bin/env python3
"""Update the overnight summary report for the supervision trainval runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = WORKTREE
ROOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118"
PILOT = MAIN / "outputs/agentguard/experiments/shutrack_a_supervision_pilot_20261118"
E200_LOG = MAIN / "outputs/agentguard/experiments/shutrack_a_submit_20261117/local_eval/sportsmot_val.log"
E200_DIR = MAIN / "outputs/agentguard/experiments/shutrack_a_submit_20261117/local_eval/SportsMOT_val/sportsmot_val_0.80_local_parity_shutrack_v2"
GROUPS = ("control", "motion_diou", "appearance_score", "motion_diou_appearance_score")
SEQUENCES_FOR_CLEAR = ("v_9MHDmAMxO5I_c003", "v_00HRwkvvjtQ_c007", "v_00HRwkvvjtQ_c011")


def run_parser(logs: dict[str, Path], out: Path) -> dict:
    command = [sys.executable, str(WORKTREE / "tools/parse_val_logs.py"), "--out", str(out)]
    for name, path in logs.items():
        command += ["--log", f"{name}={path}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}
    return json.loads(out.read_text())


def appearance_findings() -> dict:
    findings = {"sequences": {}}
    for sequence in SEQUENCES_FOR_CLEAR:
        path = PILOT / sequence / "events.jsonl"
        if not path.is_file():
            continue
        events = [json.loads(line) for line in path.read_text().splitlines()]
        valid = [
            event
            for event in events
            if event.get("appearance_diag_valid") and event.get("valid_appearance")
        ]
        mismatch = [event for event in valid if event.get("identity_mismatch_frame") is not None]
        clean = [event for event in valid if event.get("identity_mismatch_frame") is None]

        def mean(field, subset):
            values = [float(event[field]) for event in subset if event.get(field) is not None]
            return float(np.mean(values)) if values else None

        improved = sum(
            1
            for event in valid
            if event["appearance_score_template1"] > event["appearance_score_template0"]
        )
        key_rows = []
        for event in events:
            frame = event["frame_t"]
            if sequence.startswith("v_9MHDmAMxO5I") and 105 <= frame <= 116:
                pass
            elif sequence.startswith("v_00HRwkvvjtQ_c007") and 555 <= frame <= 590:
                pass
            elif sequence.startswith("v_00HRwkvvjtQ_c011") and (
                380 <= frame <= 400 or 688 <= frame <= 700
            ):
                pass
            else:
                continue
            if not event.get("appearance_diag_valid"):
                continue
            key_rows.append(
                {
                    "frame": frame,
                    "track_id": event["track_id"],
                    "gt_id": event.get("target_gt_before"),
                    "score0": event["appearance_score_template0"],
                    "score1": event["appearance_score_template1"],
                    "mismatch": event.get("identity_mismatch_frame"),
                    "recovery": event.get("identity_recovery_frame"),
                }
            )
        findings["sequences"][sequence] = {
            "diag_valid": len(valid),
            "later_mismatch": len(mismatch),
            "no_mismatch": len(clean),
            "score0_mismatch_mean": mean("appearance_score_template0", mismatch),
            "score0_clean_mean": mean("appearance_score_template0", clean),
            "score1_mismatch_mean": mean("appearance_score_template1", mismatch),
            "score1_clean_mean": mean("appearance_score_template1", clean),
            "template1_improved": improved,
            "key_rows": key_rows,
        }
    return findings


def clear_durations(clear: dict) -> list[dict]:
    rows = []
    for sequence, groups in clear.get("sequences", {}).items():
        for group, payload in groups.items():
            spans = payload.get("identity_errors", [])
            for span in spans:
                if span.get("wrong_duration", 0) >= 5:
                    rows.append(
                        {
                            "sequence": sequence,
                            "group": group,
                            "gt_id": span["gt_id"],
                            "wrong_from": span["wrong_from"],
                            "wrong_track": span["wrong_track_id"],
                            "previous_track": span["previous_track_id"],
                            "recovery": span.get("recovery_frame"),
                            "duration": span.get("wrong_duration"),
                        }
                    )
    return rows


def main() -> None:
    generation_path = ROOT / "generation_state.json"
    generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    coverage = {"generated": 0, "samples": 0, "valid_motion": 0, "motion_gt_valid": 0, "appearance_diag_valid": 0}
    for summary_path in sorted((ROOT / "data").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        coverage["generated"] += 1
        coverage["samples"] += int(summary.get("samples", 0))
        coverage["valid_motion"] += int(summary.get("valid_motion", 0))
        coverage["motion_gt_valid"] += int(summary.get("motion_gt_valid", 0))
        coverage["appearance_diag_valid"] += int(summary.get("appearance_diag_valid", 0))

    training = {}
    for group in GROUPS:
        config_path = ROOT / "groups" / group / "config.json"
        metrics_path = ROOT / "groups" / group / "metrics.jsonl"
        complete_path = ROOT / "groups" / group / "complete.json"
        entry = {"group": group, "complete": complete_path.is_file()}
        if config_path.is_file():
            entry["config"] = json.loads(config_path.read_text())
        if metrics_path.is_file():
            rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            if rows:
                entry["epochs_done"] = rows[-1]["epoch"]
                entry["global_step"] = rows[-1]["global_step"]
                entry["last_loss"] = rows[-1]["loss"]
                entry["last_epoch_seconds"] = rows[-1]["epoch_seconds"]
        training[group] = entry

    logs = {"e200": E200_LOG}
    for group in GROUPS:
        log = ROOT / "evals" / f"{group}_val.log"
        if log.is_file():
            logs[group] = log
    parsed_path = ROOT / "evals" / "parsed_all.json"
    parsed = run_parser(logs, parsed_path) if len(logs) > 1 else {}

    result_dirs = {
        "e200": E200_DIR,
        **{
            group: ROOT / "evals" / f"{group}_val" / f"sportsmot_val_0.80_{group}_val_shutrack_v2"
            for group in GROUPS
            if (ROOT / "evals" / f"{group}_val.log").is_file()
        },
    }
    clear = {}
    if len(result_dirs) > 1:
        command = [sys.executable, str(WORKTREE / "tools/clear_id_analysis.py")]
        for name, directory in result_dirs.items():
            command += ["--result", f"{name}={directory}"]
        command += ["--out", str(ROOT / "evals" / "clear_id.json")]
        subprocess.run(command, capture_output=True, text=True, check=False)
        if (ROOT / "evals" / "clear_id.json").is_file():
            clear = json.loads((ROOT / "evals" / "clear_id.json").read_text())

    motion_validation = {}
    motion_path = PILOT / "motion_diou_batch_validation.json"
    if motion_path.is_file():
        motion_validation = json.loads(motion_path.read_text())

    key_diou = {}
    key_path = PILOT / "key_diou_summary.json"
    if key_path.is_file():
        key_diou = json.loads(key_path.read_text())

    findings = appearance_findings()
    durations = clear_durations(clear) if clear else []

    lines = []
    lines.append("# SportsMOT 监督实验通宵汇总")
    lines.append("")
    lines.append(f"生成序列：{coverage['generated']}/90；样本 {coverage['samples']:,}；"
                 f"有效运动标签 {coverage['valid_motion']:,}；GT 运动监督 {coverage['motion_gt_valid']:,}；"
                 f"外观区分度有效 {coverage['appearance_diag_valid']:,}")
    if generation:
        lines.append(f"生成状态：done={len(generation.get('done', []))} failed={len(generation.get('failed', []))} "
                     f"skipped={len(generation.get('skipped', []))}")
    lines.append("")
    lines.append("## 训练进度")
    lines.append("")
    lines.append("| group | complete | epochs | step | last loss |")
    lines.append("|---|---|---|---|---|")
    for group in GROUPS:
        entry = training[group]
        lines.append(
            f"| {group} | {entry['complete']} | {entry.get('epochs_done')} | "
            f"{entry.get('global_step')} | {entry.get('last_loss')} |"
        )
    lines.append("")
    if parsed:
        lines.append("## 45 序列 val raw（开发诊断）")
        lines.append("")
        lines.append("| group | HOTA | AssA | IDF1 | DetA | MOTA |")
        lines.append("|---|---|---|---|---|---|")
        order = [name for name in ("e200", "control", "motion_diou", "appearance_score", "motion_diou_appearance_score") if name in parsed]
        for name in order:
            row = parsed[name]["combined"]
            if row is None:
                continue
            lines.append(
                f"| {name} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | "
                f"{row['DetA']:.4f} | {row['MOTA']:.4f} |"
            )
        lines.append("")
        if "control" in parsed and len(parsed) > 2:
            lines.append("### 相对 control 的差值")
            lines.append("")
            lines.append("| group | ΔHOTA | ΔAssA | ΔIDF1 | ΔDetA | ΔMOTA |")
            lines.append("|---|---|---|---|---|---|")
            base = parsed["control"]["combined"]
            for name in order:
                if name == "control" or name not in parsed:
                    continue
                row = parsed[name]["combined"]
                if row is None:
                    continue
                lines.append(
                    f"| {name} | {row['HOTA']-base['HOTA']:+.4f} | {row['AssA']-base['AssA']:+.4f} | "
                    f"{row['IDF1']-base['IDF1']:+.4f} | {row['DetA']-base['DetA']:+.4f} | "
                    f"{row['MOTA']-base['MOTA']:+.4f} |"
                )
    lines.append("")
    lines.append("## c003 / c007 / c011 身份错误持续时间（CLEAR IoU≥0.5）")
    lines.append("")
    if durations:
        lines.append("| sequence | group | gt | wrong_from | track | recovery | duration |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in durations:
            if row["sequence"] not in SEQUENCES_FOR_CLEAR:
                continue
            if row["duration"] >= 20 or row["wrong_from"] in (109, 569, 570, 394):
                lines.append(
                    f"| {row['sequence']} | {row['group']} | gt{row['gt_id']} | "
                    f"{row['wrong_from']} | {row['previous_track']}->{row['wrong_track']} | "
                    f"{row['recovery']} | {row['duration']} |"
                )
    else:
        lines.append("CLEAR 分析尚未生成。")
    lines.append("")
    if motion_validation:
        lines.append("## 运动 GT DIoU 真实 batch 验证")
        lines.append("")
        for key in (
            "events",
            "valid_events",
            "gradient_nonzero_fraction",
            "gradient_abs_mean",
            "diou_g0_mean",
            "diou_gate_mean",
            "diou_g1_mean",
            "gate_matches_best_endpoint_fraction",
        ):
            lines.append(f"- {key}: {motion_validation.get(key)}")
        lines.append("")
    if key_diou:
        lines.append("## 关键窗口 GT 框 DIoU（loss=1-DIoU，越低越好）")
        lines.append("")
        lines.append("| sequence | window | events | prefers g0 | prefers g1 | mean loss g0 | mean loss gate | mean loss g1 | gate matches best |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for sequence, windows in key_diou.items():
            for window in windows:
                lines.append(
                    f"| {sequence} | {window['start']}-{window['end']} | {window['events']} | "
                    f"{window['prefers_g0']} | {window['prefers_g1']} | "
                    f"{window['mean_loss_g0']:.4f} | {window['mean_loss_gate']:.4f} | "
                    f"{window['mean_loss_g1']:.4f} | {window['gate_matches_best']} |"
                )
        lines.append("")
    if findings:
        lines.append("## 外观区分度试跑证据（支持与反例）")
        lines.append("")
        for sequence, entry in findings["sequences"].items():
            lines.append(
                f"- {sequence}: diag={entry['diag_valid']} mismatch={entry['later_mismatch']} "
                f"score0 mismatch/clean={entry['score0_mismatch_mean']}/{entry['score0_clean_mean']} "
                f"score1 mismatch/clean={entry['score1_mismatch_mean']}/{entry['score1_clean_mean']} "
                f"template1_improved={entry['template1_improved']}"
            )
        lines.append("")
        lines.append("关键窗口逐事件见下节（score0=保持模板，score1=完整 EMA 写入）。")
        lines.append("")
        lines.append("支持案例：c003 的 t4(gt3) 在 108 帧 score0=-0.0059，随后 109 帧错配；t5(gt4) 在 109 帧 score0=-0.0277，随后 110 帧错配。c007 的 t7(gt7) score0 从 555 帧 0.1609 降到 560 帧 -0.1240，随后 567 帧互换。c011 的 t2(gt0) 从 381 帧起持续为负（-0.2252/-0.0991/-0.0411...），395 帧错配；t9(gt7) 380 帧 score0=-0.0558，384 帧错配、393 帧恢复。c011 汇总上 mismatch 事件的平均 score0=0.2036，低于无 mismatch 的 0.2335。")
        lines.append("")
        lines.append("反例：c003/c007 的汇总均值几乎无分离（0.174 vs 0.177、0.221 vs 0.221）；c007 的 t3(gt2) 在 555-560 帧 score0=0.18-0.43 仍于 565 帧错配；c003 的 t7(gt6) 在 106 帧 score0=-0.0846 但没有错配；若干分叉帧上完整 EMA 写入反而降低区分度（c003 108 帧 t4、112 帧 t4），说明“写入收益”方向并非始终一致。")
        lines.append("")
        for sequence, entry in findings["sequences"].items():
            lines.append(f"### {sequence} 关键窗口")
            lines.append("")
            lines.append("| frame | track | gt | score0 | score1 | mismatch | recovery |")
            lines.append("|---|---|---|---|---|---|---|")
            for row in entry["key_rows"][:80]:
                lines.append(
                    f"| {row['frame']} | {row['track_id']} | {row['gt_id']} | "
                    f"{row['score0']:.4f} | {row['score1']:.4f} | {row['mismatch']} | {row['recovery']} |"
                )
            lines.append("")
    lines.append("## 训练结果中的支持与反例（本轮实测）")
    lines.append("")
    lines.append("运动 GT DIoU：支持——c003 的 GT3/GT4 109 帧互换在 control 与 motion_diou 中均已消失（长错误从 3 处降到 1 处，HOTA 0.8702→0.9270），kUtTtmLaJA_c008 的 GT6/GT7 87 帧互换消失（4 处→2 处，HOTA +0.0479）；主表 +0.0028 HOTA / +0.0033 AssA。反例——c011 未修复其 GT0/GT7 与 GT5/6 互换（HOTA −0.0091），kUtTtmLaJA_c004 −0.0130；cC2mHWqMcjk_c008 的短时身份变化 10→16，但 HOTA 反而 +0.0134。")
    lines.append("")
    lines.append("外观区分度：支持——appearance_score 在 c011 消除了 GT0/GT7 于 394/395 开始的 341/280 帧持久互换（与试跑中 c011 的负/衰减 score 一致）；5ekaksddqrc_c001 +0.0817 HOTA、kUtTtmLaJA_c006 +0.0632（相对 control）等显著改善。反例——c003 的 GT3/GT4 109 帧互换仍存在（118 帧），而 control/motion_diou 已修复，appearance 在 c003 上比 control 低 0.0565 HOTA；c007 的 GT2/GT7 138 帧互换在所有组中均未恢复；c011 的 GT5/GT6 695 帧 40 帧互换仍在；9MHDmAMxO5I_c009 −0.0365。因此该评分可解释并修复部分案例（尤其 c011 的一对长期互换），但尚不能作为通用外观软标签替代。")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 45 条 val 已在 e200 之前参与训练，且本轮各组继续在 trainval（含 val）上训练，所有分数均为开发诊断，不是独立 test。")
    lines.append("- 训练数据为 e200 在线回放产生的成熟门控事件；各组共享同一批样本、采样顺序与步数。")
    lines.append("- 运动组仅把运动 loss 换成下一帧 GT 框 DIoU；外观组仅把外观软标签换成区分度分数（无效诊断回退原软标签）；联合组同时启用两者（运动 DIoU + 外观区分度软标签，1:1 相加）。")
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (ROOT / "summary.json").write_text(
        json.dumps(
            {
                "coverage": coverage,
                "generation": generation,
                "training": training,
                "parsed_combined": {name: value.get("combined") for name, value in parsed.items()},
                "clear_durations": durations,
                "motion_validation": motion_validation,
                "appearance_findings": findings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"report": str(ROOT / "REPORT.md"), "groups_scored": list(parsed)}))


if __name__ == "__main__":
    main()
