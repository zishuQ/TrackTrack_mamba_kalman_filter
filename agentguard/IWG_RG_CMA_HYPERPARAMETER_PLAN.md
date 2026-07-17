# IWG + RG-CMA 三数据集调参与实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-17
- Verification Status: ANALYZED
- Version Label: iwg_rg_cma_tuning_plan_v1
- Scope: MOT17-FRCNN、MOT20、SportsMOT
- Excluded Scope: 原 TrackTrack 检测、关联、Kalman、GMC、阈值和后处理参数
- Primary Metric: 验证集 final raw HOTA
- Guard Metrics: base raw HOTA、MOTA、IDF1、DetA、AssA、gate 校准与修正诊断

## 目标与基本原则

本计划只优化 IWG、RG-CMA、标签、数据采样和训练过程，不调整原 TrackTrack 的任何参数，也不使用 AFLink 或线性插值结果选择训练超参数。

所有调参结论必须满足以下约束：

1. 验证阶段只比较 raw，test 阶段才使用 post。
2. 同组实验必须对齐样本暴露量、optimizer steps 和有效完整数据遍历次数。
3. 同时记录 base 和 final；若 final 长期低于 base，说明 CMA 没有提供有效增益。
4. 训练 loss、gate accuracy 和 gate MAE 只能用于诊断，不能代替 tracker HOTA。
5. 首轮使用 seed=42 筛选；最终候选至少使用 3 个训练 seed 复核。
6. 不做全笛卡尔积搜索；先单因素筛选，再做少量交互实验。

## 当前基线配置

| 模块 | 当前配置 |
|---|---|
| 输入 | track ReID 2048、detection ReID 2048、scalar63 |
| 历史 | 最近 6 个真实在线事件，包含 unmatched 历史 |
| EventEncoder | ReID 2048 -> 64；appearance 256 -> 128 -> 64；scalar 63 -> 128 -> 64 |
| Base IWG | d_model=128，2 层，4 heads，FFN=256，dropout=0.1 |
| RG-CMA | temporal attention 1 层；cross-modal 1 层；4 heads；FFN=256 |
| 输出 | base motion/app gate；final correction 上限为 +/-0.05 |
| 参数量 | 总计约 811,086；Base IWG 约 501,580；RG-CMA 约 309,506 |
| 优化 | AdamW，lr=1e-4，weight_decay=1e-4，batch=1024 |
| 调度 | warmup=1 epoch，cosine decay to zero，grad clip=1.0 |
| 标签 | future horizon=5，policy temperature=0.1 |
| 损失 | base=1.0，policy=0.1，cue=0.2，risk=0.1，final=1.0，residual=0.5，revision=0.01 |

### 梯度边界

- `L_base` 和 `L_policy` 训练 Base IWG。
- cue/risk 使用 detached event embedding，只训练对应 heads。
- RG-CMA 输入 detached；`L_final`、residual 和 revision 只训练 RG-CMA。
- reliability 输入中的 base/policy/cue/risk/scalar 也 detached。
- 当前唯一间接耦合是一次 backward 后的全局 gradient clipping。

因此，单纯提高 `L_final` 不会让 Base IWG 学得更好；过大的 CMA 或辅助损失反而可能通过全局裁剪压低 Base IWG 的实际更新。

## 数据规模与已观察到的注意力行为

| 数据集 | 训练样本量 | motion entropy | appearance entropy | 主要风险 |
|---|---:|---:|---:|---|
| MOT17-FRCNN | 94,052 | 约 1.59 | 约 1.67 | 小数据、过拟合、迁移后遗忘 |
| MOT20 | 1,049,088 | 约 1.52-1.56 | 约 1.64-1.68 | 序列不平衡、分片顺序、训练量混淆 |
| SportsMOT | 577,972 | 约 1.32-1.39 | 约 1.44-1.56 | 长期运动关系和 unmatched 历史利用不足 |

6 个位置的最大熵为 `ln(6)=1.792`。MOT17/MOT20 的注意力接近均匀，说明当前问题不一定是 attention 容量不足；SportsMOT 的选择性更强，更适合优先尝试更长历史或适度增宽模型。

## 优先级 P0-P1：先解决监督、数据暴露与优化

| 优先级 | 调整对象 | 当前值 | 建议候选 | 可能提升机制 | 主要退化风险 | 敏感数据集 |
|---|---|---:|---|---|---|---|
| P0 | 有效完整遍历 | 随 shard 数变化 | 固定 full passes 或总 samples/steps | 消除名义 epoch 混淆 | 不对齐时结论无效 | 全部 |
| P0 | shard 顺序 | 固定轮转 | 每 cycle 随机打乱，记录顺序 | 避免固定数据段总吃到较高 LR | 随机性增加，需要固定 seed | MOT20 |
| P0 | checkpoint 选择 | 部分只评末轮 | 25/50/75/100 或按 full pass 对齐 | 识别平台期和过拟合 | tracker 评测成本高 | 全部 |
| P1 | future horizon | 5 | 3 / 5 / 8 | 直接改变 safe gate 的时间尺度 | 太短近视；太长归因噪声大 | Sports 可试 8；MOT17 3/5 |
| P1 | safe tau | 数据自适应值 | 0.5x / 1x / 2x | 调整软标签区分度和安全样本比例 | 太小趋向全关；太大趋向全开 | 全部 |
| P1 | policy temperature | 0.1 | 0.05 / 0.1 / 0.2 | 调整 policy target 峰度 | 太低不稳定；太高近似均匀 | MOT20、Sports |
| P1 | 序列采样 | 按样本 | sample-proportional / sqrt-size / sequence-balanced | 避免大序列支配更新 | 完全均匀会过采短序列 | MOT20 |
| P1 | 历史长度 | 6 | 4 / 6 / 8；Sports 再试 10 | 调整冲突判断时间范围 | 长历史稀释证据并增加 padding | Sports 8 优先 |
| P1 | base:CMA LR | 1:1 | 1:0.5 / 1:1 / 1:2 | 匹配两个 detached 学习问题的收敛速度 | CMA 太快追噪声；太慢退化为 base | MOT17 先试 1:0.5 |
| P1 | batch / LR | 1024 / 1e-4 | 512、1024、2048；5e-5、1e-4、2e-4 | 改变梯度噪声和更新次数 | batch 翻倍不代表 LR 必须线性翻倍 | MOT17 对 batch 更敏感 |
| P1 | scheduler | warmup1 + cosine | warmup 1/3/5；cosine 或 constant+decay | 改善早期稳定性和后期有效学习 | warmup 太长浪费有限更新 | MOT20、Sports |
| P1 | gradient clip | global 1.0 | global 0.5/1/2；base/CMA 分组裁剪 | 防止 CMA 大梯度压缩 base 更新 | clip 太小欠拟合；太大不稳定 | 全部 |
| P1 | loss 权重 | 见基线 | final 0.5/1；residual 0.25/0.5/1；revision 0/0.01/0.05 | 控制修正频率、方向和幅度 | 不训练 base；可能加剧全局裁剪 | MOT17 最需保守 |

## 优先级 P2-P4：输入、容量和低优先级路线

| 优先级 | 调整对象 | 当前值 | 建议候选 | 可能提升机制 | 主要退化风险 | 敏感数据集 |
|---|---|---:|---|---|---|---|
| P2 | reliability 输入 | gate+policy+cue+risk+scalar63 | full / 去 scalar / 仅 gate+policy / 仅 cue+risk | 检查 reliability token 是否过度主导 cross attention | 去除有效信息会降低冲突识别 | MOT17、MOT20 |
| P2 | scalar63 编码 | 单路 MLP | association/motion/context 分块后融合 | 减少异质语义互相干扰 | 参数增加，削弱跨块交互 | MOT20、Sports |
| P2 | correction bound | 0.05 | 0.02 / 0.05 / 0.10 | 为真实大 residual 提供修正空间 | 破坏 safe base prior | Sports 可试 0.10 |
| P2 | base residual scale | 0.15 | 0.10 / 0.15 / 0.20 | 调整 base 相对 policy prior 的自由度 | 太大不稳；太小不适应实例 | MOT17 偏 0.10/0.15 |
| P2 | dropout | 0.1 | 0 / 0.1 / 0.2 | 控制过拟合 | 0.2 可能欠拟合 | MOT17 |
| P2 | weight decay | 1e-4 | 0 / 1e-5 / 1e-4 / 1e-3 | 控制泛化 | 1e-3 可能压制小 gate head | MOT17 |
| P3 | modality token | 64 | 64 / 96 / 128 | 增加模态表达能力 | 小数据过拟合、checkpoint 不兼容 | Sports |
| P3 | d_model | 128 | 128 / 192 / 256 | 增加融合容量 | 计算量与过拟合增长 | Sports 先试 192 |
| P3 | Base IWG 层数 | 2 | 1 / 2 / 3 | 增强事件组合能力 | 可能只是更复杂的平均池化 | Sports 才优先 3 |
| P3 | attention heads | 4 | 2 / 4 / 8 | 改变子空间划分 | d_model=128 时 8 heads 仅 16 维/head | 先试 2/4 |
| P3 | FFN | 256 | 256 / 512 | 增加非线性容量 | 参数增加但不改善时序选择 | Sports |
| P3 | CMA 层数 | temporal1 + cross1 | temporal 1/2；cross 1/2 | 增强模态冲突建模 | 放大 reliability token 的错误主导 | 最后才试 |
| P4 | optimizer | AdamW | AdamW；AdamW beta2=0.95；Lion | 可能改变响应速度和泛化 | Lion 对软标签 gate 可能振荡 | AdamW param groups 优先 |
| P4 | 跨数据集训练 | 分数据集 | MOT17+MOT20 或三数据集平衡混合 | 增加运动和拥挤场景覆盖 | 域冲突；小数据被大数据吞没 | 必须 dataset-balanced |
| P4 | Mamba 教师标签 | 已实验 | NSA / Mamba / 混合教师 | 迁移运动教师能力 | 已有 Mamba-native 未显示优势 | 暂停投入 |

## 公平分片换算

名义 epoch 不能直接跨 shard 数比较。当前调度中，一个完整数据遍历需要访问全部 shard：

```text
effective full passes = nominal epochs / memory_shards
```

已知实验换算：

| 配置 | 名义 epochs | 有效 full passes |
|---|---:|---:|
| SportsMOT 4-shard | 200 | 50 |
| SportsMOT 5-shard | 200 | 40 |
| MOT20 4-shard | 100 | 25 |
| MOT20 10-shard | 100 | 10 |

若比较 shard 粒度本身，应例如比较：

```text
MOT20:     4-shard e100  vs 5-shard e125  vs 10-shard e250  （均 25 full passes）
SportsMOT: 4-shard e200  vs 5-shard e250  vs 10-shard e500  （均 50 full passes）
```

同时必须对齐总样本数和 optimizer steps，并在每个完整 cycle 随机化 shard 顺序。

## 三数据集专属策略

| 数据集 | 第一阶段 | 第二阶段 | 暂不优先 |
|---|---|---|---|
| MOT17-FRCNN | horizon 3/5、history 4/6、base:CMA LR=1:0.5、dropout/WD | MOT20 warm-start 后小 LR 迁移 | d_model=256、3 层 CMA、10 events |
| MOT20 | 对齐 full passes、sequence/sqrt-size balance、随机 shard 顺序 | horizon 5/8、history 4/6/8、分组 LR | 固定 shard 顺序下比较 4/10 shard |
| SportsMOT | history 6/8/10、horizon 5/8、bound 0.05/0.10 | d_model 128/192、FFN 256/512 | 只围绕 batch 反复搜索 |

## 分阶段实验顺序

### Stage 0：建立可比较基线

- 固定每个数据集的 full passes、总样本暴露和 optimizer steps。
- 统一保存按 full pass 对齐的 checkpoint。
- 验证只跑 base raw 和 final raw；不使用 post 选择模型。
- 首轮 seed=42，tracker seed=10000。

### Stage 1：监督和采样

依次比较 horizon、tau、policy temperature 和 sequence-balanced sampling。改变 horizon、tau 或 policy temperature 必须重新生成 labels，并写入新的 label schema/hash。

### Stage 2：上下文与优化

- MOT17：history 4/6。
- MOT20：history 4/6/8。
- SportsMOT：history 6/8/10。
- 固定 batch=1024，比较 base/CMA LR：`1e-4/5e-5`、`1e-4/1e-4`、`1e-4/2e-4`。
- 完成分组 LR 后再比较 batch，不把 batch 和 LR 同时改变。

### Stage 3：reliability 与修正机制

先做 reliability 输入消融，再调整 loss、correction bound 和 base residual scale。若 reliability token 持续支配 cross attention，应先解决输入和训练目标，再考虑堆叠 cross-modal 层。

### Stage 4：模型容量

只让 Stage 1-3 的最优配置进入宽度和层数实验。优先 SportsMOT 的 `d_model=192` 或 history=8；MOT17 原则上保持小模型。

### Stage 5：复核与交互实验

- 30% 预算初筛，保留前两名跑满。
- 只测试有机制依据的两因素交互，例如 `history x horizon`、`batch x LR`、`bound x residual loss`。
- 最终候选使用至少 3 个训练 seed。

## 必须记录的指标和诊断

### Tracker 指标

- base raw 与 final raw 的 HOTA、MOTA、IDF1、DetA、AssA。
- 每序列指标，避免总体指标掩盖短序列退化。
- test 才记录 post；验证阶段不使用 post。

### Gate 与优化诊断

- gate accuracy、motion MAE、appearance MAE、Brier score、ECE。
- `final-base` 平均绝对修正量、修正率、正负比例。
- correction 达到 `+/- bound` 的比例。
- `safe-base.detach()` 被 correction bound 截断的比例。
- 每个 loss 对 Base、cue/risk、temporal attention、cross attention、correction heads 的梯度范数。
- gradient clipping 前后的分模块梯度范数。
- motion/appearance 各历史位置权重和 attention entropy。
- cross-modal 3x3 attention，特别是 reliability token 占比。
- 每个序列和 shard 的样本数、optimizer steps、平均 LR 和访问顺序。

只有当大量目标 residual 被 0.05 截断、CMA 修正方向正确且实际 correction 接近边界时，才应把 correction bound 提高到 0.10。若 correction 很少接近边界，扩大 bound 不会解决问题。

## Checkpoint 与数据兼容性

- 修改 token 维度、d_model、层数、heads、FFN、history positional embedding 或输入结构时，必须使用新的 model schema 和 checkpoint 目录。
- 结构不兼容的 checkpoint 不能 strict load；为公平比较，第一轮不使用 partial warm-start。
- 修改 label horizon、tau、policy temperature 或 safe fallback 时，必须使用新的 label schema、dataset hash 和规范化元数据。
- 只改变 LR、batch、weight decay、scheduler 或 loss 权重时，可以保持模型 schema，但实验目录必须独立。

## 实验审计记录：MOT20 4-shard e100（2026-07-17）

实验目录：

```text
outputs/agentguard/experiments/
  iwg_rg_cma_v1_mot20_interleaved_shard4x1_seed42_bs1024_100e
```

该实验为 4 shards、每 shard 连续训练 1 个名义 epoch、循环 25 次，共 100 个名义 epochs，即 25 次有效 full passes。单个名义 epoch 只包含一个 shard，不能直接横向比较；下表使用 checkpoint 前最近 4 个 epochs 的样本加权聚合，确保每个窗口包含全部 4 个 shards。

| Checkpoint 窗口 | Total loss | Gate acc | Motion MAE | Appearance MAE | 结论 |
|---|---:|---:|---:|---:|---|
| e22-e25 | 1.229384 | 0.855407 | 0.187033 | 0.158107 | 仍在明显学习 |
| e47-e50 | 1.192207 | 0.867479 | 0.183083 | 0.137377 | 持续改善 |
| e72-e75 | 1.176401 | 0.872892 | 0.180940 | 0.128885 | 已保存节点中训练代理指标最均衡 |
| e97-e100 | 1.177380 | 0.872375 | 0.180518 | 0.129786 | 平台期；motion 略好，其余轻微回退 |

全周期扫描结果：

- 最低 total/base/final loss 和最低 appearance MAE：cycle 21，e81-e84。
- 最高 gate accuracy：cycle 22，e85-e88。
- 最低 motion MAE：cycle 23，e89-e92。
- e75-e100 整体处于平台，没有训练崩溃，也没有明显继续拟合收益。

现有 tracker 评测只覆盖 `epoch100 final raw`：

```text
HOTA 0.792449 | MOTA 0.934741 | IDF1 0.921734
DetA 0.806825 | AssA 0.778890
```

epoch25/50/75 尚未跑对应的 tracker raw 评测。因此：训练代理指标倾向 e75/约 e84-e92，但当前唯一经过 HOTA 验证、且 test 表现已知的 checkpoint 仍是 e100。迁移到 MOT17 时默认使用 e100；若要严谨确认 MOT20 自身峰值，只需补跑 e75 final raw，必要时再跑 e50，不需要优先评测 e25。

### 下一轮推荐：MOT20 4-shard e100 -> MOT17

初始化 checkpoint：

```text
outputs/agentguard/experiments/
  iwg_rg_cma_v1_mot20_interleaved_shard4x1_seed42_bs1024_100e/
  checkpoints/iwg_rg_cma_epoch100.pt
```

推荐主实验保持现有网络和标签不变，仅做低学习率域适配：

| 参数 | 推荐值 | 原因 |
|---|---:|---|
| MOT17 数据 | 7 个 FRCNN 序列，A-only，全量 | 与现有 94,052 样本数据集和历史实验对齐 |
| epochs | 50 | 现有 MOT20 e50 -> MOT17 实验在 50 epochs 仍有训练收益 |
| batch size | 1024 | MOT17 已验证优于 2048，且与旧迁移实验一致 |
| lr | 1e-5 | 避免破坏 MOT20 学到的拥挤运动先验；已有迁移实验有效 |
| optimizer | AdamW | 保持唯一变量为初始化 checkpoint |
| weight decay | 1e-4 | 与现有 finetune contract 一致 |
| warmup | 1 epoch | 与当前训练实现和历史结果一致 |
| scheduler | cosine to zero | 与历史迁移实验对齐 |
| grad clip | 1.0 global | 首轮不引入额外代码变量 |
| memory shards | 1 | MOT17 数据可全量加载，避免分片混淆 |
| workers / AMP | 4 / false | 保持正式配置 |
| seed | 42 | 保持可比性 |

当前 warm-start 训练会保存 e5、e10、e25、e50。选择时至少对 e25、e50 跑 MOT17 `all` 的 base raw 和 final raw；若二者接近，再补 e10。主判据为 final raw HOTA，约束为 final 不低于同 checkpoint 的 base。验证/训练集诊断阶段不加 post。

不建议首轮把 LR 提到 2e-5 或 5e-5，因为 MOT17 只有 94,052 个样本，源模型已经在 1,049,088 个 MOT20 样本上形成稳定先验；提高 LR 会同时加速域适配和灾难性遗忘。若 1e-5 的 e25 与 e50 几乎没有变化，再单独开第二实验测试差分 LR，而不是修改本次主实验。

## 结论边界

本文中所有“可能提升”都是由当前代码结构、梯度路径、已有训练日志和注意力统计得到的实验假设，不是保证的 HOTA 增益。最终判断以同一数据划分、同一 TrackTrack 配置、对齐训练预算后的 raw tracker 指标为准。
