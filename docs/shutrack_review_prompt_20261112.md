# SHUTrack 方案审查提示词（2026-11-12）

> 用法：把本提示词与 `docs/shutrack_session_summary_20261112.md` 一起发给高级模型。下附中文版和英文版，任选其一。

---

## 中文版

你是一位资深的多目标跟踪（MOT）研究者，熟悉 Kalman filter、数据关联、在线跟踪器设计，以及 CVPR/ICCV/ECCV 的方法写作与审稿标准。请审查我随消息附上的项目总结（`shutrack_session_summary_20261112.md`），并和我讨论下一步方案。

背景要点（与总结文档一致）：

- 我有两个在研项目：SHUTrack（代码库 TrackTrack/AgentGuard）和 MoTTrack（代码库 mamba_kalman_filter）。
- SHUTrack 的两个组件：HUC（Safe-Direct IWG，6 事件历史 + 融合事件 Transformer，输出运动/外观门控）和 RUC（RG-CMA，时序/跨模态注意力 + 有界残差修正）。
- 关键实验事实：同一 checkpoint 只切 base/final，MOT17 +0.11、MOT20 −0.09、SportsMOT +0.004 HOTA；α sweep 平坦；MOT17 02/09/10 上通道消融：关运动门控 −4.44 HOTA、关外观门控 −1.25 HOTA，而带 CMA 训练 vs without-CMA 架构为 66.62 vs 64.91。
- MoTTrack 的 MCP/TTA：MCP 是协方差预测管线，TTA 是管线输入侧的残差摘要侧枝（`x = x + summary`），训练用因果注意力、推理用每 track 缓冲区和 SSM 状态；消融为 full / tta_only / mcp_only。
- 我倾向的改造方向：把 RUC 从"输出侧修正"改为"输入侧残差摘要器"（TTA 式），HUC 作为主干（MCP 式）。备选方案：运动/外观对称双分支、propose–guard 表述、乘性可靠性门控。

请回答：

1. 诊断是否正确？HUC/RUC 的核心问题是否是"两个组件在同一输出空间竞争"？有没有其他更准确的解释？
2. 比较四条路线——(a) 输入侧残差改造，(b) 运动/外观对称双分支，(c) propose–guard 表述，(d) 乘性可靠性——从性能潜力、实现/重训成本、审稿人接受度、与 MoTTrack 的区分度四个维度给出排序和明确推荐。
3. 如果走 (a)：RUC 的 reliability token 去掉 base gate/policy/cue/risk 后应如何设计？是否需要增加 pre-gate 的轻量可靠性头？RUC-only 消融是否成立？最可能的失效模式是什么？
4. 如果走 (b)：如何与 MoTTrack 的 Q/R 结构区分，避免自我重复或 novelty 重叠？现有 −4.44/−1.25 的通道消融是否足以支撑论文主张？还缺哪些实验？
5. 给定现有证据（base/final 差异在噪声级），论文应该采用什么叙事才不会被审稿人拆穿？需要补哪些最小实验（数据集、种子、统计显著性、消融）？
6. 指出总结文档中的逻辑漏洞、过度推断或遗漏的关键对照实验。
7. 最后给出明确结论：选哪条路线、先做哪三个实验、预期风险与止损条件。

要求：结论明确并给出理由；不要泛泛而谈；如果需要更多信息，列出具体问题。

---

## English version

You are a senior multi-object tracking (MOT) researcher, familiar with Kalman filters, data association, online tracker design, and the method-writing/review standards of CVPR/ICCV/ECCV. Please review the attached project summary (`shutrack_session_summary_20261112.md`) and discuss the next steps with me.

Key context (consistent with the summary):

- I have two ongoing projects: SHUTrack (codebase TrackTrack/AgentGuard) and MoTTrack (codebase mamba_kalman_filter).
- SHUTrack has two components: HUC (Safe-Direct IWG: 6-event history + fused-event Transformer, outputs motion/appearance gates) and RUC (RG-CMA: temporal/cross-modal attention + bounded residual correction).
- Key empirical facts: switching only base/final on the same checkpoint yields MOT17 +0.11, MOT20 −0.09, SportsMOT +0.004 HOTA; the α sweep is flat; on MOT17 02/09/10, disabling the motion gate costs −4.44 HOTA and disabling the appearance gate costs −1.25 HOTA, while with-CMA training vs a without-CMA architecture is 66.62 vs 64.91.
- MoTTrack's MCP/TTA: MCP is the covariance-prediction pipeline; TTA is a residual summarization side-branch on the pipeline input (`x = x + summary`), trained with causal attention and run with per-track buffers and SSM state; ablations are full / tta_only / mcp_only.
- My preferred rework: turn RUC from an output-side corrector into an input-side residual summarizer (TTA-style), with HUC as the backbone (MCP-style). Alternatives: symmetric motion/appearance branches, a propose–guard framing, or multiplicative reliability gating.

Please answer:

1. Is the diagnosis correct? Is the core problem that the two components compete in the same output space? Are there better explanations?
2. Compare four routes—(a) input-side residual rework, (b) symmetric motion/appearance branches, (c) propose–guard framing, (d) multiplicative reliability—on performance potential, implementation/retraining cost, reviewer acceptance, and differentiation from MoTTrack. Give a ranking and a clear recommendation.
3. If (a): how should RUC's reliability token be designed after removing base gate/policy/cue/risk? Is a lightweight pre-gate reliability head needed? Is a RUC-only ablation meaningful? What are the most likely failure modes?
4. If (b): how to differentiate from MoTTrack's Q/R structure and avoid novelty overlap? Are the existing −4.44/−1.25 channel ablations sufficient? What experiments are missing?
5. Given the current evidence (base/final differences are at noise level), what narrative can survive review? What minimal experiments are needed (datasets, seeds, significance, ablations)?
6. Point out any logical gaps, over-claims, or missing control experiments in the summary.
7. End with a clear recommendation: which route, which three experiments to run first, expected risks, and stop conditions.

Be decisive and justify your reasoning; avoid generic advice; if you need more information, list specific questions.
