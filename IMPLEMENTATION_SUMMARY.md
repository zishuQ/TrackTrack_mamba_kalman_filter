# AgentGuard 完整代码实施总结

## 概述

为 TrackTrack（CVPR 2025）追踪系统完整实现了 AgentGuard 门控架构——一个运动/外观门控网络，在 TrackTrack 接受检测后独立决定写入多少运动信息和外观信息。

**核心约束**：AgentGuard 不改变 TrackTrack 的匹配决策、不重跑 Hungarian 分配、不输出新 Track ID/Detection ID、不修改检测框、不修改 NMS/TPA/TAI。

---

## 一、修改的现有文件（4 个）

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `3. Tracker/trackers/track.py` | +95 行 | 新增 `snapshot_state()`（状态冻结）、`restore_state()`（状态恢复）、`update_with_gates(frame_id, detection, motion_gate, appearance_gate)`（门控更新）。运动门控在 KF 预测状态与完整更新之间插值；外观门控在旧 ReID 与完整 EMA 之间插值。全 1 门等价于原始 `update()`，误差 < 1e-7。 |
| `3. Tracker/trackers/utils.py` | +61 行 | `iterative_assignment()` 新增 `return_meta=False` 参数。`True` 时额外返回关联元数据字典（IOU、余弦、角度、raw_cost、final_cost、匹配轮次、阈值、检测来源）。`False` 时行为完全不变。 |
| `3. Tracker/trackers/tracker.py` | +120 行 | 5 个最小插入点引入 AgentGuard：(1) 帧开始保存 frame_start 快照；(2) CMC+predict 后保存 pre_update 快照；(3) 第一阶段关联用 `return_meta=True`；(4) 匹配/未匹配 track 通过 adapter 构建事件、调 IWG、应用门控；(5) `finalize_first_stage` 触发 TGR 重放。不成熟轨迹、AgentGuard 关闭时完全走原始路径。 |
| `3. Tracker/run.py` | +24 行 | 新增 `--agentguard-mode {off,iwg,full}`、`--iwg-checkpoint`、`--tgr-checkpoint`、`--agentguard-device` 参数。mode=off 保持原名；mode=iwg 追加 `_agentguard_iwg`；mode=full 追加 `_agentguard_full`。 |

---

## 二、新增文件总数：124

### 2.1 集成层 (`3. Tracker/integrations/agentguard/`, 5 文件)

唯一同时了解 TrackTrack 内部对象和 AgentGuard 协议的模块：

| 文件 | 职责 |
|------|------|
| `converters.py` | Track ↔ TrackStateSnapshot、检测 ↔ DetectionObservation、关联元数据 ↔ AssociationPairFeatures 的转换 |
| `adapter.py` | `AgentGuardTrackerAdapter`：构建 matched/unmatched 事件、调用 IWG、应用门控、管理缓冲区、触发 TGR |
| `config_bridge.py` | 从 argparse 提取 AgentGuard 配置 |
| `hooks.py` | 生命周期 hook 桩 |

### 2.2 数据协议 (`agentguard/src/agentguard/contracts/`, 6 文件)

| 文件 | 内容 |
|------|------|
| `enums.py` | `DetectionSource`（HIGH/LOW/NMS_DELETED_HIGH）、`TrackLifecycle`（NEW/TRACKED/LOST/REMOVED）、`WritePolicy`（FULL_WRITE/MOTION_ONLY/APPEARANCE_ONLY/HOLD_BOTH/SOFT_CAUTION）及其原型矩阵 P(5×2)、`EventType`（CLEAN_OBSERVATION 到 INSUFFICIENT_EVIDENCE 共 10 类） |
| `states.py` | `TrackStateSnapshot`（冻结 dataclass，所有 NumPy 数组深拷贝）、`DetectionObservation`（冻结，含 x1y1x2y2/x1y1wh/cxcywh 只读属性）、`AssociationPairFeatures`（冻结，含 IOU/余弦/置信度/角度/raw_cost/final_cost/匹配轮次/阈值/检测来源） |
| `events.py` | `TrackEvent`（可变 dataclass，含 frame_start_state、pre_update_state、detection、association、warp_matrix、63 维标量、ReID 特征、IWG 输出） |
| `outputs.py` | `GateDecision`（冻结，motion_gate、appearance_gate 裁剪到 [0,1]） |
| `serialization.py` | JSON 序列化/反序列化 |

### 2.3 特征工程 (`agentguard/src/agentguard/features/`, 4 文件)

| 文件 | 内容 |
|------|------|
| `scalar.py` | `compute_scalar_features()`：63 维标量特征 → 关联特征 0-26（IOU 距离、余弦距离、代价统计、行/列熵）、几何与运动特征 27-54（检测-预测残差、归一化位置、KF 协方差对角线、速度展平）、轨迹上下文 55-62（分数、历史长度、间隙、状态 one-hot）。无检测事件前 26 维置零，检测几何置零。 |
| `normalization.py` | `NormalizationStats`：fit/transform/inverse_transform/save/load |
| `builder.py` | `EventFeatureBuilder`：在线/离线共用同一个 `build()` 方法，输出标量(63)、track ReID、detection ReID |

### 2.4 网络模型 (`agentguard/src/agentguard/models/`, 3 文件)

| 文件 | 内容 |
|------|------|
| `event_encoder.py` | `EventEncoder`：共享 ReID 投影 Linear(D,64)+LayerNorm → 256 维交互特征（track、det、差、积）→ 128+64 维融合 → 最终 128 维 event embedding |
| `iwg.py` | `IWG`：6 事件序列（当前+5 历史）经 Transformer（2 层、128 维、4 头）→ policy_head（5 维 softmax）+ gate_residual_head（2 维 tanh）+ event_head（10 类）+ cue_head（3 维 sigmoid）。最终门 = clip(policy @ P + 0.15·tanh(residual), 0, 1) |
| `tgr.py` | `TGR`：4 事件窗口经 EventEncoder → 拼接 IWG 策略/门控投影 → Transformer → 逐位置残差 head → 修正门 = clip(iwg_gate + 0.5·tanh(residual), 0, 1)。无检测位置强制置 [0,0]。 |

### 2.5 在线运行时 (`agentguard/src/agentguard/runtime/`, 6 文件)

| 文件 | 内容 |
|------|------|
| `buffers.py` | `EventBuffer`（IWG 历史，最多 6，左 padding）、`WindowBuffer`（TGR 窗口，固定 4） |
| `checkpoint.py` | `CheckpointManager`：track_id → event_id → TrackStateSnapshot |
| `replay.py` | `ReplayEngine`：从 checkpoint 重放事件序列（warp→predict→gated update），支持有/无检测事件 |
| `manager.py` | `AgentGuardRuntime`：mode=off/iwg/full 三态、成熟轨迹判定（state∈{Tracked,Lost} ∧ history≥6）、run_iwg_inference、_run_tgr_inference、finalize_first_stage |
| `statistics.py` | `RuntimeStatistics`：事件计数、IWG/TGR 调用数、重放次数、平均门控值 |

### 2.6 纯 NumPy 运动模型 (`agentguard/src/agentguard/motion/`, 2 文件)

| 文件 | 内容 |
|------|------|
| `nsa_numpy.py` | `NSAKalmanFilter`：8 维状态（cxcywh+速度）、4 维观测、完全匹配 TrackTrack 的 NSA KF（Q 依赖于 mean[2]/mean[3] 缩放后逐元素平方；R 依赖于 projected_mean[2]/projected_mean[3] 缩放后逐元素平方再乘 (1-confidence)；Cholesky 增益计算）。含 predict/project/update/initiate/mean_to_bbox/bbox_to_measurement/apply_warp。与 TrackTrack KF 误差 < 1e-7。 |

### 2.7 离线缓存系统 (`agentguard/src/agentguard/data/`, 9 文件)

| 文件 | 内容 |
|------|------|
| `cache_schema.py` | `CacheManifest` dataclass |
| `cache_writer.py` | `EventCacheWriter`：按序列分 shard 写 .pt 文件（events_00000.pt 等） |
| `cache_reader.py` | `EventCacheReader`：完整回读 |
| `gt_reader.py` | `GTReader`：解析 MOT 格式 gt.txt，提供逐帧 GT 框 |
| `gt_matching.py` | `GTMatching`：Hungarian IoU 匹配（阈 0.5），-1 表示不匹配 |
| `identity_prototype.py` | `IdentityPrototypeBuilder`：收集 score≥0.6 ∧ IoU≥0.7 的 ReID → L2 归一化 → 初始均值 → 保留余弦相似度 top 70% → 再均值归一化 |
| `candidate_builder.py` | `CandidateBuilder`：A（真实匹配）、B（错误身份，最低 final_cost 的其他 GT）、C（同身份低质量，NMS 删除 > 低分 > 其他高分） |
| `future_oracle.py` | `FutureOracleBuilder`：保存当前 GT 框 + 未来 5 帧 GT 框/Oracle 检测/warp 矩阵 |
| `split_manager.py` | `SplitManager`：从 YAML 读取训练/验证序列划分 |

### 2.8 Rollout 标签 (`agentguard/src/agentguard/rollout/` + `agentguard/src/agentguard/labels/`, 6 文件)

| 文件 | 内容 |
|------|------|
| `rollout/motion.py` | 运动 Rollout：write(1) 与 skip(0) 两分支 → 未来 5 帧用 Oracle 检测 KF 更新 → 每帧损失 1-IoU+0.25·L1_normalized → 收益 Bm = L_skip - L_write |
| `rollout/appearance.py` | 外观 Rollout：write/skip 两分支 EMA → 未来 5 帧用 Oracle ReID 继续 EMA → 每帧损失 1-cos_sim(feat, prototype) → 收益 Ba = L_skip - L_write |
| `rollout/window.py` | TGR 窗口标签：穷举 16 种二元序列 → 重放 4 帧 + Oracle 预测 3 帧 → 选最低运动/外观总损失序列 |
| `rollout/losses.py` | IoU、L1 归一化、运动帧损失、外观损失、sigmoid |
| `labels/__init__.py` | `compute_dataset_stats()`（tau 从中位数裁剪 1e-3~0.1）、`compute_soft_target()`（y=sigmoid(B/tau)）、`build_rollout_labels()`、`compute_policy_soft_target()` |

### 2.9 数据集 (`agentguard/src/agentguard/datasets/`, 4 文件)

| 文件 | 内容 |
|------|------|
| `iwg_dataset.py` | `IWGDataset`：当前事件 + 同一轨迹前 5 帧真实历史，序列 padding，collate_fn |
| `tgr_dataset.py` | `TGRDataset`：4 事件连续窗口，collate_fn |
| `samplers.py` | `BalancedSampler`：A/B/C/不匹配样本平衡采样 |
| `split_validation.py` | 训练/验证序列交叉污染检测 |

### 2.10 训练 (`agentguard/src/agentguard/training/`, 7 文件)

| 文件 | 内容 |
|------|------|
| `train_iwg.py` | IWG 训练循环：BCE gate + KL policy + Brier = 总损失；AdamW lr=3e-4、cosine scheduler、AMP、梯度裁剪 5、early stop patience=5 |
| `train_tgr.py` | TGR 训练循环：MaskedBCE + 0.02·Δg²；冻结 EventEncoder 和 IWG，仅训练 TGR；lr=2e-4 |
| `optim.py` / `scheduler.py` | AdamW + cosine warmup scheduler |
| `checkpointing.py` | 完整 checkpoint 保存/恢复（含 reid_dim、config、tau 等元数据） |
| `metrics.py` / `logging_utils.py` | MetricsTracker、TensorBoard 日志、JSONL/CSV/JSON 输出 |

### 2.11 Teacher 标注 (`agentguard/src/agentguard/teacher/`, 7 文件)

| 文件 | 内容 |
|------|------|
| `event_selector.py` | 多条件筛选（标签接近 0.5、运动/外观冲突、低 oracle 覆盖、候选竞争、Dlow/Ddel、Student 预测误差、熵高），最大 10000 事件 |
| `evidence_packet.py` | `EvidencePacketBuilder`：为每个事件生成 event.json + contact_sheet.jpg + appearance_gallery.jpg + prompt.json |
| `response_schema.py` | `TeacherResponse` Pydantic 模型：event_types、cue_reliability、policy_prior（5 项概率和 ∈ [0.99,1.01]）、request_temporal_revision、confidence、abstain、evidence_frames |
| `bailian_client.py` | `BailianClient`：百炼 API 调用（Flash → Plus 升级）、base64 图像编码、JSON 响应解析 |
| `request_cache.py` | `RequestCache`：SHA256 哈希去重、token 用量统计、费用计算、CSV/JSON 导出 |
| `runner.py` | `TeacherRunner`：完整标注流程编排 |

### 2.12 Verifier 验证与标签融合 (`agentguard/src/agentguard/verifier/`, 7 文件)

| 文件 | 内容 |
|------|------|
| `local_replay.py` | `LocalReplayVerifier`：5 种策略本地重放 → Delta Rm 评分 |
| `event_descriptor.py` | `EventDescriptor`：14 维跨事件描述符（候选 margin、熵、检测重叠、来源 one-hot、轨迹年龄、间隙、归一化速度、label、oracle 覆盖） |
| `event_grouping.py` | `SimilarEventGrouper`：余弦相似度找 k=32 最近邻，排除本序列，要求 ≥12 事件 ∧ ≥3 序列 |
| `cross_event.py` | 跨事件评分：Sm = Mean(Rm) - 0.5·Std(Rm) - 0.5·HarmRate |
| `scoring.py` | Verifier 分布 softmax(S/0.1) → 与 Teacher 几何平均融合 → 策略门控 = q·P |
| `label_fusion.py` | Rollout 可靠性（GT/Or/收益/一致性加权）+ Teacher 置信度 → 加权融合 → Teacher abstain 时回退 Rollout |

### 2.13 评估 (`agentguard/src/agentguard/evaluation/`, 1 文件)

TrackEval 封装：HOTA、AssA、DetA、IDF1、IDs、Frag。Oracle 阻塞检查（AssA/IDF1/IDs 均无改善 → oracle_blocked.json）。

### 2.14 CLI 工具 (`agentguard/src/agentguard/cli/`, 2 文件，~2100 行)

12 个子命令：`cache_events`、`validate_cache`、`build_rollout_labels`、`evaluate_oracle`、`build_student_v0_data`、`train_student_v0`、`select_teacher_events`、`build_evidence_packets`、`run_bailian_teacher`、`verify_and_fuse`、`build_student_v1_data`、`train_student_v1`。

### 2.15 运行脚本 (`scripts/agentguard/`, 14 文件)

| 脚本 | 用途 |
|------|------|
| `00_verify_environment.sh` | 环境检查（数据集、GT、检测缓存、ReID 维度、CMC、输出目录） |
| `01_run_baseline.sh` | 运行 Baseline 追踪 |
| `02_cache_events.sh` | 单序列/全量事件缓存 |
| `03_validate_cache.sh` | 缓存校验（生成 3 个统计 JSON） |
| `04_build_rollout_labels.sh` | Rollout 运动/外观标签构造 |
| `05_run_oracle.sh` | Oracle IWG/Full 评估 |
| `06_build_student_v0_data.sh` | Student-V0 数据集构造 |
| `07_train_student_v0.sh` | Student-V0 训练 |
| `08_select_teacher_events.sh` | 困难事件筛选 |
| `09_build_evidence_packets.sh` | Evidence Packet 生成 |
| `10_run_bailian_teacher.sh` | 百炼 Teacher 标注 |
| `11_verify_and_fuse_teacher.sh` | Verifier 验证与标签融合 |
| `12_build_student_v1_data.sh` | Student-V1 数据集构造 |
| `13_train_student_v1.sh` | Student-V1 训练 |

### 2.16 配置文件 (`agentguard/configs/`, 7 文件)

`runtime.yaml`（模式/缓冲/重放）、`data.yaml`（缓存分片/GT 匹配/身份原型）、`training.yaml`（IWG/TGR/V1 超参）、`bailian.yaml.example`、`splits/{MOT17,MOT20,SportsMOT}.yaml`。

### 2.17 测试 (`agentguard/tests/`, 21 文件, 260 测试)

所有测试通过，0 失败。覆盖：快照/恢复、全 1 门等价、全 0 门行为、关联元数据等价、特征维度、NSA NumPy KF 一致性、缓存往返、GT 投票无泄漏、A/B/C 候选、运动/外观 Rollout、TGR 窗口标签、Teacher Schema 校验、请求缓存、升级逻辑、Verifier 拒绝、跨序列分组、标签融合、在线关闭等价。

---

## 三、用户运行流程

```bash
# 环境变量
export DATASET=MOT17
export DATA_DIR=/path/to/datasets
export PICKLE_DIR="../outputs/2. det_feat"
export BAILIAN_API_KEY=your_key  # 仅在 step 10 需要

# 1. 基线验证
bash scripts/agentguard/01_run_baseline.sh

# 2. 事件缓存（先单序列测试再全量）
MODE=train_custom SEQUENCE=MOT17-02 MAX_FRAMES=200 bash scripts/agentguard/02_cache_events.sh
MODE=train_custom bash scripts/agentguard/02_cache_events.sh

# 3. 缓存检查
bash scripts/agentguard/03_validate_cache.sh

# 4. Rollout 标签
bash scripts/agentguard/04_build_rollout_labels.sh

# 5. Oracle 评估
MODE=val_custom bash scripts/agentguard/05_run_oracle.sh

# 6-7. Student-V0
bash scripts/agentguard/06_build_student_v0_data.sh
DEVICE=cuda bash scripts/agentguard/07_train_student_v0.sh

# 8-10. Teacher
bash scripts/agentguard/08_select_teacher_events.sh
bash scripts/agentguard/09_build_evidence_packets.sh
bash scripts/agentguard/10_run_bailian_teacher.sh

# 11-13. Student-V1
bash scripts/agentguard/11_verify_and_fuse_teacher.sh
bash scripts/agentguard/12_build_student_v1_data.sh
DEVICE=cuda bash scripts/agentguard/13_train_student_v1.sh
```

---

## 四、未实现

- **可求解困难事件生成器**（hard events generator）：需等 Student-V1 真实失败统计完成后单独开发。

---

## 五、关键数字

| 指标 | 值 |
|------|-----|
| Commit | `26b88dc` (branch: `agent`) |
| 修改文件 | 4 |
| 新增文件 | 124 |
| 代码行数 | +20,651 |
| 测试数量 | 260 (全部通过) |
| ReID 维度 | 从数据/checkpoint 元数据读取，不写死 |
| 63 维标量特征 | 在线/离线共用同一构建器 |
| KF 对等误差 | < 1e-7 |
| 全 1 门等价误差 | < 1e-7 |
| Snapshot/Restore 误差 | < 1e-7 |
| 在线/离线特征误差 | < 1e-6 |
