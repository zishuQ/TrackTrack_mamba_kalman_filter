# SHUTrack A 四维速度实验复现入口（2026-11-19）

四维 KF 速度（vx, vy, vw, vh）实验的全部入口，均从主目录运行：

    cd /home/shang/workspace/TrackTrack
    NEW=outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119

## 数据（90 条 trainval，同一 e200 回放）

    .venv/bin/python tools/generate_trainval_paired.py --root "$NEW" --skip-rank-arrays

每个样本新增：

- `kf_velocity4_g0/g1`：t+1 的 KF 速度四维端点（按图像宽/高归一化）；
- `gt_velocity4`：相邻差分 GT 标签 [vx, vy, vw, vh]，mask 为 `size_adj_valid`；
- `gt_size_velocity_central`：t 与 t+2 GT 在 t+1 计算坐标下的中心差分宽高标签，
  独立 mask `size_central_valid`（仅真实帧号连续、GT 与 warp 都有效时）；
- 原有 `kf_velocity_g0/g1`、`gt_velocity`、`speed_valid` 保持不变。

本轮训练只用相邻差分标签；中心差分留给下一项独立比较。

## 验证

    .venv/bin/python tools/verify_speed4.py

输出 `$NEW/speed4_validation.json`：与上一轮数据逐位对齐、g=0/0.5/1 与线上 KF
（含 GMC 帧）抽查、真实 batch 梯度、size_weight=0 与中心速度组逐位一致。

## 训练与评估

    SIZE_WEIGHT=14 bash tools/run_train_speed4.sh
    bash tools/run_eval_speed4.sh

`SIZE_WEIGHT` 由真实 batch 梯度选定：宽高 L1 的贡献取中心 L1（权重 50）的 50%
（实测建议 12.8–14.7，取 14）。其余配置与联合组/中心速度组一致：90 序列、
batch 1024、seed 42、lr 1e-4、wd 1e-4、warmup 185、4 分片 ×10 循环、
40 轮 ×185 batch = 7,400 步。

## 配对漏检诊断与汇总

    .venv/bin/python tools/paired_gap_diagnostic.py \
      --checkpoint joint=outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118/groups/motion_diou_appearance_score/checkpoints/epoch_040.pt \
      --checkpoint motion_diou_speed=outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119/groups/motion_diou_speed/checkpoints/epoch_040.pt \
      --checkpoint motion_diou_speed4="$NEW/groups/motion_diou_speed4/checkpoints/epoch_040.pt" \
      --output "$NEW/evals/paired_gap.json"
    .venv/bin/python tools/speed4_analysis.py

配对诊断从三组共同拥有的最后一个匹配事件出发，使用检测层面相同的无检测区间
（该 GT 在区间内没有被 IoU>=0.5 的检测命中），沿线上 GMC + KF 递推，比较第
1、3、5 帧的中心误差、宽高误差与框 IoU。

结果入口：`$NEW/REPORT.md`、`$NEW/summary.json`、`$NEW/speed4_validation.json`、
`$NEW/evals/clear_id_all.json`、`$NEW/evals/paired_gap.json`。
