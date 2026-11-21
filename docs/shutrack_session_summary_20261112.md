# SHUTrack / AgentGuard 会话总结（2026-11-12）

> 用途：交给外部高级模型审查与讨论。本文自包含，包含仓库事实、实验证据、设计分析、已执行动作和待决问题。

## 0. 背景

- 主仓库：`/home/shang/workspace/TrackTrack`，git 分支 `main` 与 `agentguard`。
- `agentguard` 分支是当前论文项目，论文中命名为 **SHUTrack**，两个组件分别叫 **HUC** 和 **RUC**。
- 参照项目：`/home/shang/workspace/mamba_kalman_filter`（文档中称 MoTTrack / MTUTrack），其两个组件是 **MCP**（Mamba Covariance Predictor）和 **TTA**（Trajectory Token Aggregator），作者认为这两者"耦合得更完美、论文故事更好写"。
- 本会话要回答四个问题：
  1. HUC、RUC 在代码里分别是什么？cue/risk 辅助头是干什么的？
  2. 为什么要加 RUC 做有界修正？HUC 不能单独做到吗？设计理由是什么？
  3. MoTTrack 的 MCP 与 TTA 是如何耦合、如何被论文叙述的？
  4. SHUTrack 能否照此改造或表述？如何改造？并准备重训验证。

---

## 1. AgentGuard / SHUTrack 代码事实

### 1.1 两个组件的对应关系

代码中没有 `HUC`、`RUC`、`SHUTrack` 字样，以下映射是按架构语义推断（论文作者未否认）：

- **HUC ↔ Safe-Direct IWG（Instance-wise Gate network）**
  - `agentguard/src/agentguard/models/iwg.py:18`（基类 `IWG`）
  - `agentguard/src/agentguard/models/iwg_rg_cma.py:467`（实际使用的 `SafeDirectIWG`）
  - 输入：6 个事件的历史上下文（当前 + 5 历史），每个事件由 track ReID、det ReID、63 维标量特征（IoU、角度、KF 几何、匹配可靠性等）编码为 128 维事件向量。
  - 结构：EventEncoder → 2 层 Transformer（d=128, 4 heads）→ policy head（5 类原型）+ gate residual head（2 维）。
  - 输出：motion gate、appearance gate，两个 [0,1] 连续值，决定"接受匹配后往轨迹状态里写多少运动/外观信息"。
  - 运行时：`--agentguard-mode iwg-rg-cma --iwg-rg-cma-output base` 使用它的原始输出。

- **RUC ↔ RG-CMA（Reliability-Guided Cross-Modal Attention）**
  - `agentguard/src/agentguard/models/iwg_rg_cma.py:655`（`IWGRGCMA`），核心在 `_refine_flat`（`:881`）
  - 输入：6 帧因果窗口的 motion / appearance token，加一个 reliability token（由 base gate、policy 概率、cue、risk、63 维标量拼接投影而成）。
  - 结构：逐模态时序注意力 + 跨模态注意力（motion/appearance/reliability 三个 token）→ 两个 correction head。
  - 输出：有界修正 `correction = bound * tanh(raw)`，`refined_gate = clip(base_gate + correction, 0, 1)`；无检测位置强制为 0。`correction_bound` 旧配置 0.05，新配置 0.10。
  - 运行时：`--iwg-rg-cma-output final`；正式 test 默认 `final`。

- **关键契约（Safe-Direct）**
  - base gate 在 RUC 中是 `detach()` 的；correction head 最后一层零初始化（`iwg_rg_cma.py:825-826`）；修正被 bound 截断并 clip（`:1026`, `:1040-1041`）。
  - 描述符里写明 `"gradient_contract": "safe_direct_base_detached_rg_cma"`。
  - 初始化时 `final == base`；RUC 只能微调，不能覆盖 HUC。
  - runtime 支持 `rg_cma_alpha` 缩放修正（`runtime/manager.py:263-268`）和 `disable_kf_gate` / `disable_ema_gate`（`:274-276`）。

### 1.2 cue / risk 辅助头

定义在 `agentguard/src/agentguard/data/label_schema.py:70-92`，头在 `iwg.py:42-43`：

- **cue（3 维）** = `[motion_label_confidence, appearance_label_confidence, max(两者)]`
  - 置信度 = `min(|benefit| / (2τ), 1) * coverage`，衡量 rollout 标签的确定程度。
- **risk（4 维）** = `[motion_harm, appearance_harm, insufficient_evidence, cross_modal_conflict]`
  - `motion_harm = 1 − motion_soft_target`（写入运动信息的伤害概率）；
  - `appearance_harm` 同理；
  - `insufficient_evidence = 1 − min(两个置信度)`；
  - `cross_modal_conflict = |motion_target − appearance_target|`。

用途：

1. 辅助 BCE 监督（总损失权重 0.2 / 0.1，见 `training/loss_iwg_rg_cma.py:268`）。
2. 在 Safe-Direct 版本中吃 **detached** embedding（`iwg_rg_cma.py:537-538`），不向 gate 特征通路回传梯度。
3. 作为 RUC 的 reliability token 输入（`_build_reliability_input`, `:1078`）。
4. 无检测事件用 sentinel 填充：cue=0，policy=`hold_both`，risk=`[0,0,1,0]`（证据不足=1）。
5. 运行时作为诊断输出（`runtime/manager.py:13-19` 打包 base/final/correction/cue/risk）。

### 1.3 仓库内实验证据（重要）

**(a) 同一 checkpoint 只切 base/final**（`outputs/agentguard/experiments/iwg_rg_cma_best_base_final_raw_20260725/base_final_delta.tsv`）：

| 数据集 | base HOTA | final HOTA | Δ HOTA |
|---|---:|---:|---:|
| MOT17/all | 77.6202 | 77.7277 | +0.11 |
| MOT20/all | 79.2380 | 79.1448 | **−0.09** |
| SportsMOT/val | 83.2286 | 83.2326 | +0.004 |

**(b) SportsMOT val α sweep**（`outputs/agentguard/overnight/iwg_rg_cma_1x6_sportsmot_val_alpha_sweep_20260725/`）：
α = 0 → HOTA 83.1554；α = 1 → 83.1551。修正幅度对结果几乎无影响。

**(c) MOT17 02/09/10 子集消融**（`outputs/agentguard/ablation/summary.csv`，epoch 65）：

| 方法 | HOTA | Δ vs TrackTrack |
|---|---:|---:|
| TrackTrack reference | 62.15 | — |
| AgentGuard baseline（带 CMA 训练，final gate） | 66.62 | +4.48 |
| AgentGuard without-CMA 架构 | 64.91 | +2.76 |
| IWG policy-only | 65.44 | +3.29 |
| IWG direct-2gate | 64.58 | +2.44 |
| CMA without reliability token | 65.71 | +3.57 |
| CMA without cross evidence | 66.18 | +4.03 |

注意：baseline 与 without-CMA 是不同训练配置/架构，差距不能全部归因于推理期修正。

**(d) 通道级消融**（同表，MOT17 02/09/10）：

| 配置 | HOTA | Δ vs baseline |
|---|---:|---:|
| baseline | 66.62 | — |
| disable Gate_EMA（外观门控关） | 65.37 | −1.25 |
| disable Gate_KF（运动门控关） | 62.18 | −4.44 |

**结论**：官方 checkpoint 上性能几乎全部来自 HUC；RUC 的推理期边际贡献在 ±0.1 HOTA 量级（MOT20 略负）。系统里真正"两个都不可或缺、可独立消融"的轴是**运动门控通道 vs 外观门控通道**，而不是 base/corrector 两阶段。

### 1.4 为什么当前设计仍这样写

代码与实验记录支撑的解释（工程性而非性能必需）：

1. **安全契约**：base detached + 修正零初始化 + bound + clip，保证 RUC 不破坏 HUC，最坏每事件偏 bound；运行时可退回 base。
2. **残差学习/信用分配**：HUC 用直接标签训练，RUC 只学 `clip(safe_target − base, ±bound)`，并用 revision/no-harm 损失鼓励不确定时弃权（correction→0）。
3. **表达增量**：HUC 把运动/外观压成融合事件向量 + 固定 5 原型；RUC 显式保留两条模态 token 序列做时序/跨模态注意力。
4. **可消融性**：同一 checkpoint 切换 base/final、α、bound、去 reliability token、去 cross evidence，每个元素都可单独量化。

但若论文把 RUC 写成主要性能贡献，现有证据不支持；更稳的叙事是"HUC 主体 + RUC 有界安全守卫"。

---

## 2. MoTTrack 的 MCP 与 TTA

（注：Q/R 是同一个 MCP+TTA 模块的两个实例——Q 分支预测过程噪声、R 分支预测观测噪声；作者强调的"两个组件"是 **MCP 与 TTA**。）

### 2.1 代码级耦合

`/home/shang/workspace/mamba_kalman_filter/mamba_kalmanfilter.py`：

- **MCP（Mamba Covariance Predictor）= 整条协方差预测管线**
  ```
  f_x → input_proj → (x + TTA 摘要) → Mamba2 → OutputHead(Linear→LeakyReLU→Linear) → Softplus → Q_t / R_t
  ```
  Mamba2 维护每 track 的 conv/SSM 状态（`_conv_state_bank` / `_ssm_state_bank`）。
- **TTA（Trajectory Token Aggregator）= 管线内的残差侧枝**（`:41`）
  - 4 个可学习 query token 对历史特征序列做 cross-attention，mean pool 成轨迹级摘要。
  - 训练：`forward_causal` 因果掩码注意力，每个时刻只用过去（`:336-360`）。
  - 推理：`batch_step` 维护每 track 最近 16 帧特征缓冲区，批量重算摘要并 `detach()`（`:281-334`）。
  - **耦合点**：`x = x + summary`——摘要加在 MCP 的输入表示上，MCP 的 Mamba 再递推处理。
- **开关**：`use_tta` / `use_mamba` 在构造时可分别关闭，这是组件消融的基础。
- Q、R 两个分支各实例化一个完整的 TTA+MCP（`:405-420`），只有输入/输出维度不同。

### 2.2 论文叙述方式

`diagrams/gpt_image_prompts.md`、`diagrams/gpt_image2_mcp_3col_prompt.md` 等文件显示：

1. **MCP 管线图**：主路径竖直画 `Linear → ⊕ → Mamba → Linear → ReLU → Linear → Softplus`；TTA 从 InputProj 后横向分叉，曲线箭头指回 `⊕`，明确画成残差/skip；TTA 用暖橙色突出。
2. **TTA 细节图**：可学习 query tokens + cross-attention + causal mask 热力图（左下三角、近处更亮 = recency bias）。
3. **Sequence Maintenance 图**：缓冲区随帧增长，每行 `TTA + Mamba → h_t`，绿色虚线画状态 carry-over `h₁→h₂→…`，每步输出 Q_t/R_t——把 MCP 描述成流式递推预测器、TTA 描述成对增长缓冲区的摘要器。
4. **消融协议**（`run_component_multiseed.py:38-44`）：`full / tta_only / mcp_only / no_imc`，配对差值 `TTA = no_imc − mcp_only`、`MCP = no_imc − tta_only`，多种子统计。

### 2.3 为什么这种耦合"好写"

1. **生产者–消费者，不是竞争者**：TTA 输出中间表示，MCP 输出最终协方差，不在同一空间抢同一个输出。
2. **残差注入**：`x + summary`，主干信号永远保留，侧枝只能调制。
3. **同一特征空间**：TTA 在 input_proj 之后的 d_model 空间工作，与主干类型兼容。
4. **训练/推理语义一致**：训练因果注意力看整条序列，推理用缓冲区 + 每 track 状态。
5. **构造时可插拔**：两个开关让每个组件独立去掉，消融表自然成立。
6. **角色一句话**：TTA 负责"从历史里摘要出上下文"，MCP 负责"沿时间递推预测协方差"。

---

## 3. 诊断与改造方案

### 3.1 诊断

HUC/RUC 与 TTA/MCP 在形式上都是"主干 + 残差侧枝"，差别在**注入位置**：

- MoTTrack：侧枝（TTA）注入**主干的输入表示**，主干（MCP）产生最终输出 → 空间不同 → 互补。
- SHUTrack：主干（HUC）产生最终 gate，侧枝（RUC）注入**输出** → 同一空间 → 竞争。
- 这解释了为什么 RUC 必须有 bound/abstention 压住，且边际贡献只有 ±0.1 HOTA。

真正的"双组件轴"证据：运动门控 −4.44 HOTA、外观门控 −1.25 HOTA，远大于 base/final 的差异。

### 3.2 讨论过的三条路线

- **A. 只改表述**：写成 "propose–guard"（HUC 提议、RUC 有界守卫）。不重训，诚实但不对称，RUC 只能定位为安全机制。
- **B. 换轴改造**：把组件换成运动增益分支 / 外观增益分支，两个对称估计器共享同一条门控更新规则（`x_t=(1−m)x̂+mz`，`f_t=(1−a)f+a f̃`），完全对应 Q/R 式对称结构。需重训；`clean-cross-modal` 变体已完成约 80%（已有 motion/appearance 两条 token 流）。
- **C. 乘性可靠性**：`g_final = s ⊙ g_base`，s 为可靠性估计；`selective-correction` 已有 `reliability_gate_head`。改动小，但仍是层级结构。
- **不推荐 D**：把 MoTTrack 的 Q 分支并进来，novelty 与另一篇论文重叠。

### 3.3 最终选定方向：RUC 改为输入侧残差摘要器（TTA 式）

```
旧:  event → HUC(transformer+heads) → base_gate
     RUC(attention) → correction → gate = clip(base_gate + δ)

新:  event, motion_token, appearance_token ← EventEncoder
     summary = RUC(motion/appearance 因果窗口 + 上游可靠性证据)
     event'  = event + summary                ← 零初始化投影
     gate    = HUC(event')                    ← 主干
```

关键设计决策：

1. **可靠性 token 不能再用 base gate**（新结构里 RUC 在 HUC 之前，会循环依赖）。改用上游证据：motion/appearance token 一致性/冲突度量、标量特征中的可靠性子集（`CLEAN_RELIABILITY_SCALAR_INDICES` 思路）、原始事件特征。cue/risk 头保留在 HUC 输出端做辅助监督与诊断，不再进入 RUC 输入。
2. **summary 投影零初始化**，训练起点严格等于纯 HUC（残差哲学）。
3. **训练目标**：复用 `compute_iwg_without_cma_loss`（gate BCE+MSE + 0.1 policy KL + 0.2 cue BCE + 0.1 risk BCE），无 correction target / bound / revision loss；RUC 通过 HUC 的 gate 损失端到端学习（对应 TTA 通过 MCP 协方差损失学习）。
4. **训练/在线一致性**：训练 `forward_sequence` 逐位置因果摘要；在线 `forward` 只用最后位置；复用现有 6 事件窗口与 `test_online_off_equivalence.py`。
5. **接线**：新增架构变体（暂名 `ruc-residual`）；新增 `HUCRUCT` 类（子类化 `IWGRGCMA`）；`load_iwg_rg_cma_checkpoint` 按 checkpoint schema 分派，runtime 不改代码即可加载；`--iwg-rg-cma-output` 对新变体语义变为 `final`=带摘要、`base`=摘要置零（推理期组件开关）。
6. **消融矩阵**：`huc_only`（摘要置零）/ `ruc_only`（摘要直接接 gate head）/ `full`；注入点（输入侧 vs 输出侧）；reliability token 各成分；对应 TTA-only/MCP-only/full。
7. **测试**：模型 shape、残差语义、开关等价、checkpoint roundtrip、在线/离线等价、全量现有 362 个测试零回归；先 smoke train，再按原协议重训三数据集，TrackEval raw 对比官方 baseline（MOT17 77.7277 / MOT20 79.1448 / SportsMOT 83.2326 raw HOTA）。

---

## 4. 已执行的操作

- 已创建分支 **`feature/huc-ruc-tta-style`**，从 `agentguard` 的 `25c2e31` 拉出。
- **尚未修改任何代码**；`main`、`agentguard` 和三个未跟踪文件（`scripts/agentguard/audit_detection_cache_legacy080_20261004.py`、`scripts/agentguard/verify_paper_baseline_20261004.py`、`tools/screen_shared_tracking_gaps.py`）保持原样。
- 尚未提交任何 commit。

## 5. 待审查/待决问题

1. **注入点选择**：输入侧残差（TTA 式）是否优于输出侧修正？是否有第三选择（例如注入 modality token 而非融合事件）？
2. **可靠性 token 的上游证据够不够**？去掉 base gate/policy/cue/risk 后，RUC 的"可靠性引导"还剩多少信息？是否需要增加一个 pre-gate 的轻量可靠性头专门供 RUC 使用？
3. **RUC-only 消融**是否有意义？TTA-only 在 MoTTrack 中能独立预测协方差；SHUTrack 的 summary 直接接 gate head 是否能成立？
4. **预期收益风险**：新结构需要重训三个数据集，但当前证据显示 base→final 只有 ±0.1 HOTA。改成输入侧增强后是否可能仍无显著提升？是否应该优先做路线 B（运动/外观对称双分支，证据更强：−4.44 / −1.25 HOTA）？
5. **论文叙事**：HUC 主体 + RUC 守卫，还是 TTA/MCP 式的互补双组件？如果做新结构但指标不涨，如何避免过度声明？
6. **训练协议**：需要用户提供/确认三个数据集的训练命令与 schedule（MOT17 热启动 50 轮 lr 1e-5；MOT20 4×1×25 sqrt-size；SportsMOT 4×1×50）。
7. **cue/risk 是否保留**：建议保留为辅助监督/诊断，不进 RUC 输入；是否同意？
8. **从零训练 vs 热启动现有 HUC 权重**：热启动收敛快但 RUC 输入分布改变；倾向从零训练保证与官方 baseline 公平。

## 6. 关键代码位置索引

SHUTrack / AgentGuard：

- `agentguard/src/agentguard/models/iwg.py:18` — IWG 基类；`:42-43` — risk/cue 头
- `agentguard/src/agentguard/models/iwg_rg_cma.py:467` — SafeDirectIWG；`:537-538` — detached cue/risk；`:655` — IWGRGCMA；`:825-826` — correction head 零初始化；`:881` — `_refine_flat`；`:1026` — bounded correction；`:1040-1041` — mask + clip；`:1078` — `_build_reliability_input`
- `agentguard/src/agentguard/data/label_schema.py:70-92` — cue/risk 目标定义
- `agentguard/src/agentguard/training/loss_iwg_rg_cma.py:219` — base loss；`:268` — 总损失权重；`:342` — safe target
- `agentguard/src/agentguard/runtime/manager.py:13-19` — 诊断打包；`:263-268` — alpha；`:274-276` — gate 通道开关
- `3. Tracker/trackers/track.py:223` — `update_with_gates`
- `outputs/agentguard/experiments/iwg_rg_cma_best_base_final_raw_20260725/base_final_delta.tsv`
- `outputs/agentguard/ablation/summary.csv`
- `outputs/agentguard/overnight/iwg_rg_cma_1x6_sportsmot_val_alpha_sweep_20260725/metrics_summary.tsv`

MoTTrack：

- `mamba_kalman_filter/mamba_kalmanfilter.py:41` — TTA；`:124` — MambaCovNet（MCP）；`:281-334` — 推理 batch_step；`:336-360` — 训练 forward_sequence；`:362-420` — MambaKalmanFilter（Q/R 两套 MCP+TTA）
- `mamba_kalman_filter/run_component_multiseed.py:38-44` — 组件变体；`:418-420` — 配对差值定义
- `mamba_kalman_filter/diagrams/gpt_image_prompts.md`、`gpt_image2_mcp_3col_prompt.md` — 论文图叙述方式
