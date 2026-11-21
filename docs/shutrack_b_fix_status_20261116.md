# SHUTrack-B 修复 / A 训练状态记录（2026-11-16 晚）

> 用户离开前要求：记录结果后停止。明天回来继续跑剩余结果。

## 主工作树（feature/shutrack-b）— 已实现、未提交

修改/新增文件（均未暂存、未提交）：

- `shutrack/data/gmc.py`（新）— GMC 读取、命名映射、逐帧 warp 组合
- `shutrack/data/b_dataset.py` — 逐事件 warp/next_warp、last_match_frame、clip 起点身份规则、序列本地采样、`stats()`
- `shutrack/training/b_train.py` — warp→predict 递推、gap 语义、外观缓冲、CFG 边界、设备端统计、profile 窗口、每 epoch 计数器
- `shutrack/integration/tracktrack_b.py` — `_observation_box`（上一次真实观测，CMC 单独累积）
- `3. Tracker/trackers/cmc.py` — apply_cmc 对 `_observation_box` 的守卫钩子（CRLF 保持，仅 +9 行）
- `shutrack/cli_b.py` — `--sequence-split/--gmc-dir/--profile-batches`、`stats` 子命令
- `shutrack/schema_b.py` — B_ARCHITECTURE_VERSION → `shutrack_b_v2`
- `shutrack/configs/shutrack_b_sportsmot.yaml`、`shutrack/README_B.md`
- 测试：`tests/test_b_motion_parity.py`（新）、`tests/test_b_clip_sampling.py`（新）、`tests/test_b_recurrence.py`（更新）

建议 commit message：
`fix(shutrack-b): replay CMC, match-based observation gap, and sequence-local batching`

## 已完成的验证

- `pytest shutrack/tests`：98 passed, 1 skipped
- `pytest agentguard/tests`：206 passed, 1 failed —— 既有失败
  `test_iwg_rg_cma_runtime.py::test_adapter_context_size_one_does_not_reuse_history_as_current_window`
  （模型只支持 context_size ∈ {6,8}，与本次改动无关；`agentguard/` 未修改）
- 固定索引复现（测试内断言）：
  - 修复前：38,524 clips，1,359 个起点身份不符（1,277 valid-GT + 82 gt=-1）
  - 修复后：37,266 clips，起点错配 0；matched 且 `frame-last_match>1` 的步：319 → 284
  - GMC：90 序列共 55,544 行，其中 9,704 行平移 >10px
- `cli_b stats`（90 序列全部）：
  - train：45 序列 / 19,176 clips / 284,452 steps / matched 280,024 / valid_current 279,598 / valid_next 278,700 / appearance 274,358
  - val：45 序列 / 18,090 clips / 268,474 steps / matched 263,858 / valid_current 263,515 / valid_next 262,652 / appearance 258,298
  - 合计：37,266 clips / 552,926 steps / matched 543,882 / valid_current 543,113 / valid_next 541,352 / appearance 532,656；起点错配 0
- 10-step CPU smoke：整个 run 只 1 次 detection cache open、1 次 GMC load、1 次 candidate map build、1 次 index sequence load；loss 有限
- 在线复跑（新语义 + smoke 权重）：`v_00HRwkvvjtQ_c001` / `v_00HRwkvvjtQ_c003` 正常完成（HOTA≈0.04，仅验证管线）
- 验收轨迹测试：非零 CMC + 连续失配 + 重匹配，训练与在线逐步一致（预测框/写回框/速度/观测间隔/外观模板/最近检测缓冲）；默认接管阈值 6 下"3 匹配 + 2 连失 + 第 6 帧重匹配、10px/帧 → 速度=10"回归通过

## 已停止（用户关机前要求全部停止）

- **B 完整 1 epoch（GPU）**：21:49 启动，21:56 用户要求停止，已 kill（SIGTERM，exit 143）
  - 目录：`outputs/agentguard/experiments/shutrack_b_postfix_full_epoch_20261116/checkpoints`
  - 停止时进度：epoch 1, step 500/约 1459（`training.log` 有记录；`metrics.jsonl` 为空，**无 epoch 指标、无 last.pt/best.pt**）
  - 明天重跑：可直接复用同一目录（无 last.pt 不会触发拒绝），或换新目录名（如 `..._rerun_20261117`）
  - 命令（batch 32, clip 16, num-workers 0, seed 42, recent 2, profile-batches 20, epochs 1）：
    `./.venv/bin/python -u -m shutrack.cli_b train --index-dir outputs/agentguard/shutrack_b_index_trainval_fixed_20261116 --checkpoint-dir outputs/agentguard/experiments/shutrack_b_postfix_full_epoch_20261116/checkpoints --detection-cache-root outputs/agentguard/detection_cache --gt-root /home/shang/datasets/SportsMOT/dataset --split train --device cuda --epochs 1 --batch-size 32 --clip-length 16 --num-workers 0 --seed 42 --recent-detections 2 --profile-batches 20 --log-interval 100`

## 明天待办

1. 重跑 B 的完整 1 epoch（用户关机前已停止，未产生 epoch 指标/checkpoint；命令见上节），读 `metrics.jsonl`，把测速写进 `shutrack/README_B.md` 的 Measured smoke。
2. 用该 run 的 `best.pt` 复跑两条 SportsMOT val 片段（`v_00HRwkvvjtQ_c001 v_00HRwkvvjtQ_c003`，`--agentguard-mode shutrack-b`，CPU）。
3. 提交上述 B 修复（精确暂存，不带无关未跟踪文件）。
4. A 版 val raw 评估：见下节阻塞。

## A 版状态

- worktree：`/home/shang/workspace/TrackTrack-shutrack-v2-a`，分支 `shutrack-v2-a-51ed7b6`，commit 51ed7b6
- 训练完成 20 epochs：577,972 样本/epoch，12,220 步，best training loss 0.404108，总耗时 1,443.6 s（≈24 min）
  - epoch 1：100.97 s，5,724 samples/s，data 58.6 s，compute 37.5 s，CUDA 峰值 343.4/398.0 MB
  - epoch 20：68.0 s，8,494 samples/s，data 37.4 s，CUDA 峰值 343.4 MB
- checkpoint：`outputs/agentguard/experiments/shutrack_a_v3_sportsmot_trainval_competition_bs1024_20e_seed42_20261116/checkpoints/{best.pt,last.pt}`（best 按训练 loss 选）
- **val raw 评估未完成**：
  - worktree 缺 `outputs`，已建 symlink：`TrackTrack-shutrack-v2-a/outputs -> /home/shang/workspace/TrackTrack/outputs`
  - 仍失败：找 `../outputs/2. det_feat/sportsmot_val_0.95.pickle` 时 FileNotFoundError；主仓库 `outputs/2. det_feat/` 只有 `sportsmot_val_0.80.pickle`（另有 `detection_cache/` 目录）
  - 明天需二选一：找到/恢复 0.95 紧凑缓存；或从 mmap detection cache 重建该 pickle（scripts/ 下有 `compact_det_feat_pickles.py`）
  - 评估命令（在 worktree 的 `3. Tracker` 下）：
    `../.venv/... run.py --dataset SportsMOT --mode val --agentguard-mode shutrack-v2 --shutrack-checkpoint <best.pt> --agentguard-device cpu --detection-cache-root /home/shang/workspace/TrackTrack/outputs/agentguard/detection_cache --tracker-suffix shutrack_a_sportsmot_val_e20 --print-per-sequence-metrics`
  - 注意：worktree 的 `.venv` 不存在，用主仓库 `/home/shang/workspace/TrackTrack/.venv/bin/python`
