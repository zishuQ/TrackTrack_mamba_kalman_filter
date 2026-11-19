# SHUTrack A 成对监督实验复现入口（2026-11-19）

原 `TrackTrack-shutrack-a-opt` worktree 的开发已接回主目录：

    /home/shang/workspace/TrackTrack        # 分支 feature/shutrack-v2（含 A-opt 提交）
    /home/shang/workspace/TrackTrack/.venv  # 运行环境

所有 `tools/*.py` 与 `tools/*.sh` 的仓库路径现在按脚本位置解析
（`Path(__file__).resolve().parents[1]`），不再依赖旧 worktree；实验产物仍在
主目录的 `outputs/agentguard/experiments/` 下。

## 数据与实验目录

    OLD=outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118
    NEW=outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119
    PILOT=outputs/agentguard/experiments/shutrack_a_supervision_pilot_20261118

- `OLD/groups/{control,motion_diou,appearance_score,motion_diou_appearance_score}`
  为昨晚四组（40 分片轮次 / 7,400 步）。
- `NEW/data` 为同一 e200 回放扩展后的 90 序列数据（样本行与 `OLD/data`
  逐行一致，新增外观排序特征与 KF 速度字段）。
- `NEW/groups/{appearance_rank,motion_diou_speed}` 为本轮两组。

## 复现命令

```bash
cd /home/shang/workspace/TrackTrack
OLD=outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118
NEW=outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119

# 1) 数据生成（90 条 trainval，可断点续跑；写入 $NEW/data）
.venv/bin/python tools/generate_trainval_paired.py --root "$NEW"

# 2) 真实 batch 验证（模板三档 / 排序梯度 / 速度梯度）
.venv/bin/python tools/verify_rank_speed.py

# 3) 串行训练两组（默认超参与系数见下）
bash tools/run_train_rank_speed.sh

# 4) 45 条 SportsMOT val raw 评估（seed 10000, NSA, 共享检测缓存）
bash tools/run_eval_rank_speed.sh

# 5) 漏检后中心误差 + CLEAR 逐序列汇总
.venv/bin/python tools/gap_center_error.py \
  --checkpoint joint="$OLD/groups/motion_diou_appearance_score/checkpoints/epoch_040.pt" \
  --checkpoint motion_diou_speed="$NEW/groups/motion_diou_speed/checkpoints/epoch_040.pt" \
  --output "$NEW/evals/gap_center_error.json"
.venv/bin/python tools/rank_speed_analysis.py
```

单独训练某一组：

```bash
.venv/bin/python tools/train_supervision_groups.py --group appearance_rank \
  --data-root "$NEW/data" --groups-root "$NEW/groups" \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95
.venv/bin/python tools/train_supervision_groups.py --group motion_diou_speed \
  --data-root "$NEW/data" --groups-root "$NEW/groups" --speed-weight 50
```

## 超参与可选性

- `appearance_rank`：可微外观 EMA 模板 + 身份排序损失
  （`rank_weight=18`、`tau=0.1`、`alpha=0.95`）。权重按真实 batch 参数梯度
  范数对齐联合组外观 BCE（实测比值 17.5）。
- `motion_diou_speed`：GT 框 DIoU + KF 中心速度 L1（`speed_weight=50`，
  按运动门控梯度约为框 DIoU 的 25% 选取）。**速度 L1 保留为可选实验**：
  主训练入口 `shutrack.cli train` 的默认配方不变（不带速度项/排序项）。
- 两组的其它配置（90 序列、batch 1024、seed 42、lr 1e-4、wd 1e-4、
  warmup 185、4 分片 ×10 循环、40 轮 ×185 batch = 7,400 步）与联合组完全一致。

## 结果入口

- `$NEW/REPORT.md`：主表、逐序列差值、45 条 CLEAR 汇总、gap 误差；
- `$NEW/summary.json`：机读汇总；
- `$NEW/rank_speed_validation.json`：真实 batch 验证记录；
- `$NEW/evals/clear_id_all.json`：TrackEval CLEAR 前帧匹配优先规则下的
  全部错误段；
- `$NEW/evals/gap_center_error.json`：漏检后第 1/3/5 帧中心误差。
