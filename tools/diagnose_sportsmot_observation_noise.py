#!/usr/bin/env python3
"""Small paired train-only observation audit, CPU-only and without a model.

Sample up to 128 anchor windows in each official train sequence, evaluate noise
streams at epochs 1/10/20. Separate center, size, missing and combined effects.
Read source caches in place; no training, tracking or cache changes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools", ROOT / "3. Tracker", ROOT / "agentguard/src"):
    sys.path.insert(0, str(path))
import train_supervision_groups as base
from sportsmot_supervision_pilot import load_gt
from paired_training_input import official_sequences
from paired_collection_io import atomic_json
from shutrack.training.observation_noise import ObservationNoise, corrupt_observations
from shutrack.training.window_replay import window_index
from shutrack.training.recurrent_motion import kf_update, warp_predict
from shutrack.training.motion_targets import numpy_diou

DATA = ROOT / "outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data"
OUT = ROOT / "outputs/agentguard/experiments/sportsmot_continuous_harm1_e20_20261124"
PROFILE = ObservationNoise.for_dataset("SportsMOT")
MODES = {
    "original": None,
    "center_only": replace(PROFILE, size_std=0, missing=0),
    "size_only": replace(PROFILE, position_std=0, missing=0),
    "box_noise_only": replace(PROFILE, missing=0),
    "missing_only": replace(PROFILE, position_std=0, size_std=0),
    "full_augmentation": PROFILE,
}


def xyxy(boxes):
    return np.concatenate((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), axis=-1)


def cxcywh(boxes):
    return np.concatenate(((boxes[..., :2] + boxes[..., 2:]) / 2,
                           boxes[..., 2:] - boxes[..., :2]), axis=-1)


def loss(mean, gt):
    return 1 - numpy_diou(xyxy(mean[..., :4]), gt)


def endpoint_step(mean, cov, measurement, score, present, warp):
    """Actual g=0/1 endpoints; a missing detection has identical endpoints."""
    safe = np.where(present[:, None], measurement, mean[:, :4])
    m, p = kf_update(torch.from_numpy(np.ascontiguousarray(mean)),
                     torch.from_numpy(np.ascontiguousarray(cov)),
                     torch.from_numpy(np.ascontiguousarray(safe)),
                     torch.from_numpy(np.ascontiguousarray(score)))
    full = np.where(present[:, None], m.numpy(), mean)
    full_cov = np.where(present[:, None, None], p.numpy(), cov)
    w = torch.from_numpy(np.ascontiguousarray(warp))
    no_write, _ = warp_predict(torch.from_numpy(np.ascontiguousarray(mean)),
                               torch.from_numpy(np.ascontiguousarray(cov)), w)
    update, update_cov = warp_predict(torch.from_numpy(np.ascontiguousarray(full)),
                                     torch.from_numpy(np.ascontiguousarray(full_cov)), w)
    return no_write.numpy(), update.numpy(), update_cov.numpy()


def describe(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "p50": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def summarize(rows):
    arrays = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    valid, present = arrays["valid"], arrays["present"]
    observed = valid & present
    benefit = arrays["l0"] - arrays["l1"]
    harmful = observed & (benefit < -1e-6)
    det_ok = arrays["gt_now_valid"] & arrays["active"] & present
    error = arrays["measurement"] - cxcywh(arrays["gt_now"])
    center = np.linalg.norm(error[:, :2], axis=1)
    size = np.linalg.norm(error[:, 2:], axis=1)
    det_loss = 1 - numpy_diou(xyxy(arrays["measurement"]), arrays["gt_now"])
    score_bins = []
    for lo, hi in ((0, .8), (.8, .9), (.9, .95), (.95, 1.00001)):
        hit = det_ok & (arrays["score"] >= lo) & (arrays["score"] < hi)
        score_bins.append({"range": [lo, min(hi, 1.)], "n": int(hit.sum()),
                           "center_error_px": describe(center[hit]),
                           "measurement_diou_loss": describe(det_loss[hit])})
    gt_synthetic_error = arrays["gt_synthetic_error"][det_ok]
    result = {
        "active_events": int(arrays["active"].sum()),
        "missing_events": int((arrays["active"] & ~present).sum()),
        "missing_rate": float((arrays["active"] & ~present).sum() / arrays["active"].sum()),
        "gt_valid_next_events": int(valid.sum()), "observed_valid_next_events": int(observed.sum()),
        "harmful_events": int(harmful.sum()),
        "harmful_fraction_among_observed_valid": float(harmful.sum() / max(1, observed.sum())),
        "skip_loss_all_valid": describe(arrays["l0"][valid]),
        "write_loss_all_valid": describe(arrays["l1"][valid]),
        "write_benefit_observed": describe(benefit[observed]),
        "harmful_write_damage": describe(-benefit[harmful]),
        "beneficial_write_gain": describe(benefit[observed & ~harmful]),
        "center_error_px": describe(center[det_ok]), "size_error_px": describe(size[det_ok]),
        "measurement_diou_loss": describe(det_loss[det_ok]),
        "synthetic_gt_plus_same_noise_center_error_px": describe(np.linalg.norm(gt_synthetic_error[:, :2], axis=1)),
        "synthetic_gt_plus_same_noise_size_error_px": describe(np.linalg.norm(gt_synthetic_error[:, 2:], axis=1)),
        "retained_score": describe(arrays["score"][det_ok]), "confidence_bins": score_bins,
    }
    return result


def collect_sequence(sequence, sample_count, seed, epochs):
    data = base.SequenceData(sequence, "train", DATA, "SportsMOT")
    rng = np.random.RandomState(seed + int.from_bytes(hashlib.sha256(sequence.encode()).digest()[:4], "little") % 1000000)
    rows = np.sort(rng.choice(data.n, min(sample_count, data.n), replace=False))
    all_events, all_rows, all_active = window_index(data, 3)
    events, step_rows, active = all_events[rows], all_rows[rows], all_active[rows]
    n, length = active.shape
    measurements = np.asarray(data.event_det_measurement[events], np.float64)
    scores = np.asarray(data.event_det_score[events], np.float64)
    has = np.asarray(data.event_has[events], bool) & active
    means = np.asarray(data.event_state_mean[events], np.float64)
    covs = np.asarray(data.event_state_cov[events], np.float64).reshape(n, length, 8, 8)
    warp = np.asarray(data.event_warp_next[events], np.float64).reshape(n, length, 2, 3)
    warp_ok = np.any(warp[..., :2] != 0, axis=(-1, -2))
    warp = np.where(warp_ok[..., None, None], warp, np.eye(2, 3))
    gt_next = data.small["gt_box_t1"][step_rows].astype(np.float64)
    valid = data.small["motion_gt_valid"][step_rows] & active & warp_ok
    frame = np.asarray(data.event_frame[events])
    gt_ids = data.samples["gt_id"][step_rows]
    gt = load_gt(sequence, "train", "SportsMOT")
    gt_now = np.zeros((n, length, 4), np.float64)
    gt_now_valid = np.zeros((n, length), bool)
    for i, t in np.argwhere(active):
        box = gt.get(int(frame[i, t]), {}).get(int(gt_ids[i, t]))
        if box is not None:
            gt_now[i, t] = box
            gt_now_valid[i, t] = True
    meta = json.loads((data.dir / "summary.json").read_text())
    accumulated = {scope: {mode: [] for mode in MODES} for scope in ("frozen_prior_events", "dynamic_anchor")}
    sanity = 0.
    first = np.argmax(active, axis=1)
    for epoch in epochs:
        for mode, cfg in MODES.items():
            if cfg is None:
                z, score, present = measurements.copy(), scores.copy(), has.copy()
            else:
                z, score, present = corrupt_observations(measurements, scores, has, active, rows,
                    seed=seed, epoch=epoch, width=meta["img_width"], height=meta["img_height"], config=cfg)
                z, score = z.astype(np.float64), score.astype(np.float64)
            assert np.array_equal(score[present], scores[present]), "Unexpected score change on retained observations"
            # The GT+noise reference uses the same Gaussian displacement. It is
            # only an observation-error comparison, not a replacement training run.
            delta = z - measurements
            frozen_l0 = np.zeros((n, length)); frozen_l1 = np.zeros_like(frozen_l0)
            dynamic_l0 = np.zeros_like(frozen_l0); dynamic_l1 = np.zeros_like(frozen_l0)
            mu = means[np.arange(n), first].copy()
            pc = covs[np.arange(n), first].copy()
            for t in range(length):
                no, full, _ = endpoint_step(means[:, t], covs[:, t], z[:, t], score[:, t], present[:, t], warp[:, t])
                frozen_l0[:, t] = loss(no, gt_next[:, t])
                frozen_l1[:, t] = loss(full, gt_next[:, t])
                if mode == "original":
                    cached = data.small["pred_box_g1"][step_rows[:, t]]
                    if valid[:, t].any():
                        sanity = max(sanity, float(np.abs(xyxy(full[:, :4])[valid[:, t]] - cached[valid[:, t]]).max()))
                no, full, next_cov = endpoint_step(mu, pc, z[:, t], score[:, t], present[:, t], warp[:, t])
                dynamic_l0[:, t] = loss(no, gt_next[:, t])
                dynamic_l1[:, t] = loss(full, gt_next[:, t])
                mu = np.where(active[:, t, None], full, mu)
                pc = np.where(active[:, t, None, None], next_cov, pc)
            fields = {"valid": valid, "active": active, "present": present, "measurement": z,
                      "gt_now": gt_now, "gt_now_valid": gt_now_valid, "score": score,
                      "gt_synthetic_error": delta}
            for scope, l0, l1 in (("frozen_prior_events", frozen_l0, frozen_l1),
                                  ("dynamic_anchor", dynamic_l0, dynamic_l1)):
                packed = {**fields, "l0": l0, "l1": l1}
                packed = {k: (v[:, -1] if scope == "dynamic_anchor" else v.reshape((-1,) + v.shape[2:]))
                          for k, v in packed.items()}
                accumulated[scope][mode].append(packed)
    assert sanity < .001, (sequence, sanity)
    manifest = {"sequence": sequence, "rows": rows.tolist(),
                "frames": data.samples["frame"][rows].tolist(),
                "track_ids": data.samples["track_id"][rows].tolist(),
                "baseline_endpoint_max_abs_error_px": sanity}
    data.samples.close()
    return accumulated, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-sequence", type=int, default=128)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    merged = {scope: {mode: [] for mode in MODES} for scope in ("frozen_prior_events", "dynamic_anchor")}
    manifests = []; per_sequence = {}
    for sequence in sorted(official_sequences("train", "SportsMOT")):
        observations, manifest = collect_sequence(sequence, args.samples_per_sequence, args.seed, args.epochs)
        manifests.append(manifest)
        per_sequence[sequence] = {}
        for scope, modes in observations.items():
            per_sequence[sequence][scope] = {}
            for mode, records in modes.items():
                per_sequence[sequence][scope][mode] = summarize(records)
                merged[scope][mode].extend(records)
        print(json.dumps({"sequence": sequence, "anchors": len(manifest["rows"]),
                          "endpoint_error_px": manifest["baseline_endpoint_max_abs_error_px"]}), flush=True)
    summary = {scope: {mode: summarize(records) for mode, records in modes.items()}
               for scope, modes in merged.items()}
    result = {"scope": "official train only; no model inference", "seed": args.seed,
              "epochs": args.epochs, "samples_per_sequence": args.samples_per_sequence,
              "unique_anchors": sum(len(m["rows"]) for m in manifests),
              "window_frames": 3, "profile": vars(PROFILE),
              "score_policy": "Retained detections keep original score; dropped detections receive zero score",
              "summary": summary, "sample_manifest": manifests, "per_sequence": per_sequence}
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "augmentation_diagnostic.json", result)
    (args.out / "augmentation_diagnostic.md").write_text(render(result) + "\n")
    print(json.dumps({"out": str(args.out), "unique_anchors": result["unique_anchors"]}))


def render(result):
    lines = ["# SportsMOT 现有train数据：小样本增强诊断", "",
             f"官方train45，每序列固定抽样{result['samples_per_sequence']}个锚点，"
             f"共{result['unique_anchors']}个独立锚点索引；3帧窗口，噪声取epoch={result['epochs']}、seed42。"
             "同一批窗口逐模式配对，重复噪声和相邻窗口重叠不视为独立样本；未做显著性检验。", "",
             "SportsMOT参考实现先读取GT框再加高斯噪声；USE_REAL_DET_OBSERVATION默认False。"
             "SHUTrack增强读取已有检测框，保留检测的分数不变；缺失时mask=False、score=0。"
             "本诊断未修改任何训练配方。", "",
             "## 观测误差和固定先验端点", "",
             "冻结每个事件采集时的KF先验，单独改变观测。检测误差使用当前同身份GT，"
             "端点损失使用下一帧同身份GT，单位为像素/DIoU loss。"
             "目标身份沿用原样本gt_id（当前写入前的身份投票），不重新关联。", "",
             "| 模式 | 漏检率 | 中心误差px | 尺寸误差px | 观测DIoU loss | 有害率/有观测有效事件 | 全写下一步loss | 平均写入收益 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, s in result["summary"]["frozen_prior_events"].items():
        lines.append(f"| {mode} | {s['missing_rate']*100:.3f}% | {s['center_error_px']['mean']:.4f} | "
                     f"{s['size_error_px']['mean']:.4f} | {s['measurement_diou_loss']['mean']:.6f} | "
                     f"{s['harmful_fraction_among_observed_valid']*100:.3f}% | {s['write_loss_all_valid']['mean']:.6f} | "
                     f"{s['write_benefit_observed']['mean']:+.6f} |")
    lines += ["", "平均写入收益=不写loss−完整写入loss；正值表示写入有益。有害率只在真实保留的观测上计算。"
              "漏检事件两端点相同，纳入总体预测误差，但不归类为有益更新。"
              "missing_only不会改变保留检测的框/分数，保留子集均值的小幅变化来自样本筛选。", "",
              "## 三帧固定全写入递推后的锚点", "",
              "以同一窗口起始先验递推；最多前两步按gm=1预热，比较锚点g=0/1。"
              "用于隔离增强的历史状态效应，不代替G3的模型门控历史。", "",
              "| 模式 | 锚点漏检率 | 有害率/有观测有效事件 | 锚点g=0 loss | 锚点g=1 loss | 平均写入收益 |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode, s in result["summary"]["dynamic_anchor"].items():
        lines.append(f"| {mode} | {s['missing_rate']*100:.3f}% | "
                     f"{s['harmful_fraction_among_observed_valid']*100:.3f}% | "
                     f"{s['skip_loss_all_valid']['mean']:.6f} | {s['write_loss_all_valid']['mean']:.6f} | "
                     f"{s['write_benefit_observed']['mean']:+.6f} |")
    lines += ["", "## GT加噪与真实检测叠加噪声", ""]
    s = result["summary"]["frozen_prior_events"]["box_noise_only"]
    orig = result["summary"]["frozen_prior_events"]["original"]
    lines += [f"在配对样本上，真实检测的平均中心误差为{orig['center_error_px']['mean']:.4f}px；"
              f"检测加噪后为{s['center_error_px']['mean']:.4f}px。同样噪声直接加在GT上，"
              f"平均中心误差为{s['synthetic_gt_plus_same_noise_center_error_px']['mean']:.4f}px。", "",
              "GT加噪列只比较观测来源，不复刻MoTTrack模型、损失、数据分段或训练结果；"
              "此处未用GT观测启动新训练。", "",
              "## 分数与误差", "",
              "代码已断言所有保留观测分数与原值完全一致；每个分数分箱的误差记录在JSON的confidence_bins中。"
              "分箱按左闭右开划分，最后一箱包含1。", "",
              "| 分数段 | 原始中心误差px | 加框噪中心误差px | 原始观测loss | 加框噪观测loss | 样本数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for a, b in zip(orig["confidence_bins"], s["confidence_bins"]):
        def fmt(v): return "—" if v is None else f"{v:.5f}"
        lines.append(f"| {a['range']} | {fmt(a['center_error_px']['mean'])} | {fmt(b['center_error_px']['mean'])} | "
                     f"{fmt(a['measurement_diou_loss']['mean'])} | {fmt(b['measurement_diou_loss']['mean'])} | {a['n']} |")
    miss = result["summary"]["frozen_prior_events"]["missing_only"]
    lines += ["", "## 解释范围", "",
              f"本小样本中，完整坐标扰动使固定先验的有害率从{orig['harmful_fraction_among_observed_valid']*100:.3f}%"
              f"变为{s['harmful_fraction_among_observed_valid']*100:.3f}%，"
              f"平均写入收益从{orig['write_benefit_observed']['mean']:.6f}变为{s['write_benefit_observed']['mean']:.6f}；"
              f"坐标扰动/单独模拟漏检使总体下一步loss分别变化"
              f"{s['write_loss_all_valid']['mean']-orig['write_loss_all_valid']['mean']:+.6f}/"
              f"{miss['write_loss_all_valid']['mean']-orig['write_loss_all_valid']['mean']:+.6f}。"
              "这里描述的是该固定样本的观测变化，不能直接外推全量数据或新训练结果。", "",
              "这些配对统计量描述增强对观测和端点的改变，不证明学得保守的因果机制；"
              "需将几何变化大小、模拟漏检导致的预测损失和分数保持不变一起判断。"
              "本轮未重标分数、修改miss概率或另启增强训练。各序列等量抽样，汇总不是全量train分布的无偏估计。", "",
              "原始端点重建与缓存的最大误差逐序列写入sample_manifest，要求小于0.001像素。"
              "样本行号/帧号/track id、各模式完整分布和逐序列统计均保留在augmentation_diagnostic.json。"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
