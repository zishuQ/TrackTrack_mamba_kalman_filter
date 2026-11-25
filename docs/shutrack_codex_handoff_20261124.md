# SHUTrack 对话与代码工作交接给本地 Codex

更新时间：2026-11-25（北京时间）。这是对话交接记录，不是自动执行授权。
项目：TrackTrack / AgentGuard / shutrack；MoTTrack参考代码位于mamba_kalman_filter。
项目通常从 ~/codexpro-projects/TrackTrack 打开；具体绝对目录以本机为准。

## 1. 先看这里：用户最新决定，覆盖历史安排

### 最新讨论：NSA 检测可信度倍率

2026-11-25 追加决定。用户确认历史框使用原检测框，预期 $c=1$ 恢复原 NSA 完整运动更新。首版采用原 NSA 噪声构造规则，按 $R_{\mathrm{eff}}=R_{\mathrm{NSA}}/c$ 调节，保留同一外观 gate 时与旧运动 gate 为 1 的写回行为对齐。监督、训练预算及权重来源继续待讨论。

2026-11-24 当前仅做方案讨论、只读核对和项目记忆记录，代码修改和实验由用户另行安排。运动分支候选为匹配后输出 $c\in[0.2,5]$，中性值为 1，用 $R_{\mathrm{eff}}=R_{\mathrm{NSA}}/c$ 重算 KF 更新。外观保留现有 `[0,1]` gate 与 EMA。代码核对、缓存证据和待决策项见 [NSA 检测可信度方案备忘](shutrack_nsa_reliability_design_20261124.md)。

### 已完成：补齐harm1对照与小样本增强诊断

用户在四组e20报告后决定：只补三帧连续full监督、无增强、motion_harm_weight=1这一组；其余沿用D_e20配置，从头全参数训练。50轮余弦计划、warmup80、在e20/6420步停止，仅最终e20做一次val45及门控分析。

同时以现有train数据做小样本配对增强诊断，区分中心/尺寸扰动和模拟漏检，比较当前检测误差、有害率和下一帧端点收益；不更改增强配方，不新增增强训练。

新组名`G1-3-harm1`，独立输出根`outputs/agentguard/experiments/sportsmot_continuous_harm1_e20_20261124`。入口`tools/run_sportsmot_continuous_harm1.py`；诊断`tools/diagnose_sportsmot_observation_noise.py`。

共同预算下完整2×2结果：原一步harm1 D_e20=83.660，原一步harm2 E_e20=83.294，三帧连续harm2 G1-3=83.517，三帧连续harm1=83.632。新组相对D_e20为−0.028pp，数值接近，未超过强基线；相对三帧harm2高0.115pp。本次不延长到e50。

新组已完成e20/6420步，完整val45一次、门控分析一次，退出码均为0；checkpoint与last.pt权重一致。训练计算777.4秒（约38.87秒/轮），驱动791.9秒；D_e20约21.11秒/轮，新组三帧为1.84倍耗时。新组在线motion gate序列均值0.999990、appearance gate序列均值0.601132；固定val回放有害/有益gate约0.999992/0.999994，仍几乎完整写入。当前状态`awaiting_user_decision`。

小样本train诊断已完成，train45每序列128锚点，共5760锚点，使用epoch1/10/20噪声流。检测叠加框噪后中心误差2.630→3.186像素，同噪声直接加GT的中心误差约1.246像素；固定先验有害率13.447%→24.734%；保留检测的置信度不变。单独漏检与坐标扰动令平均下一步loss分别增加约0.00029与0.01375。只描述本固定样本的分布变化，未证明模型保守化的因果来源。

最新报告：新输出根下`report.md` / `report.json`与`augmentation_diagnostic.md` / `.json`；核验记录`completion_audit.json`。本轮结束，等待用户决定，没有延长训练、正式trainval、test/post或提交。

归因口径：四组报告的G1-3收益只确认整套全参数训练方式的效果；运动与外观gate都发生变化，不能说提升主要来自运动分支。以下四组安排与结果保留为已完成历史。

当前只推进SportsMOT研究。最新协议是四组各训练20个全量epoch，比较连续监督、加噪和3帧/5帧长度：

| 组名 | 窗口 | 运动监督 | 观测增强 | 目的 |
|---|---:|---|---|---|
| G1-3 | 3帧 | 连续full监督 | 无 | 连续监督 |
| G1-5 | 5帧 | 连续full监督 | 无 | 直接对比G1-3，多两帧是否有收益 |
| G2-3 | 3帧历史构建 | anchor单步监督 | MoTTrack | 单独增强 |
| G3-3 | 3帧 | 连续full监督 | MoTTrack | 两项组合 |

G2-3的前两帧按既定全写入策略预热，模型仅在锚点调用一次。它与G1/G3的多步模型训练有明确区别。

共同配置：
- 官方train45训练、val45验证。
- query_film_bilinear，从头初始化、全参数训练，参数量1,208,220。
- dropout=0.05，motion_harm_weight=2，seed=42。
- shards=1，全量CPU特征预载；有效batch=1024、micro_batch=1024。
- AdamW，lr=1e-4，weight_decay=1e-4。
- 外观rank_weight=18、column_weight=50、tau=0.1、alpha=0.95。
- 沿原E_e20安排：50轮余弦计划，实际在e20保存后停止。
- 321步/全量轮；计划16050步，实际6420步；warmup80；e20末LR约6.59e-5，尚未降为0。
- 四组各自独立目录，重任务串行。
- 每组仅最终e20做一次完整val raw和一次门控分析；中间权重只保存。
- 同预算基线复用E_e20，HOTA83.294；E_e100的83.680仅作历史参考。
- 本轮完成训练、val和报告后停止，等待用户决定。正式trainval训练、test/post、打包提交均留待用户决断。
- 用户已取消“选优后自动正式训练和提交”，也已取消这轮各跑100轮。

### 接手后的最终核验：本轮已完成，等待用户决定

本地 Codex 已完整阅读本文件并按第1节接手。宿主机实际进程检查未发现相关训练、跟踪或门控分析重任务；复用现有完整产物，没有重新训练、重复val或门控推理。

| 组 | 最终训练 | 最终val | HOTA | 相对E_e20 / pp |
|---|---|---|---:|---:|
| G1-3 | e20 / 6420步 | val45完整 | 83.517 | +0.223 |
| G1-5 | e20 / 6420步 | val45完整 | 83.318 | +0.024 |
| G2-3 | e20 / 6420步 | val45完整 | 82.969 | −0.324 |
| G3-3 | e20 / 6420步 | val45完整 | 82.780 | −0.513 |

四组门控分析均完成。逐组核验checkpoint/last.pt、全部20轮日志、优化器和调度器、全参数数量、相同初始化哈希、官方训练/验证序列集合以及45份raw输出。E_e20共同配置和运动监督样本对齐通过；新旧事件表存储顺序不同，比较时按样本绑定对齐。当前测试216 passed、3 skipped。

G1-5比G1-3每轮多27.4秒、约慢70%，HOTA低0.199pp。加噪两组总误伤约为总节省的4.76/5.17倍；较高分离度没有转化为本轮整体收益。单seed结果仅作当前预算下比较，未隔离运动门控的独立HOTA贡献。

完整报告和核验记录：
- `outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124/report.md`（原主报告）
- `outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124/analysis_report.md`（接手续验、总节省/误伤、逐序列翻转和成本分析）
- `outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124/completion_audit.json`（checkpoint哈希与逐序列数值）
- `tools/analyze_sportsmot_mechanisms_e20.py`（仅复查产物，不启动训练/推理）

已停止在报告阶段，状态为`awaiting_user_decision`。正式trainval、test/post、打包提交和延长训练均等待用户决定。以下任务快照保留为接手前历史记录。

### 接手前核对的代码与任务快照（历史）

已直接读取最新 tools/run_sportsmot_g123_experiment.py：
- RUNS确为G1-3 / G1-5 / G2-3 / G3-3。
- COMMON确为planned_epochs=50、stop_epoch=20、warmup80。
- EVAL_EPOCH=20。
- 输出根为 outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124。
- --stage all完成分析后结束；formal仅作为显式独立stage存在，不再自动调用。
- 文件顶部docstring当时仍残留“100轮/e100”等旧文字，实际COMMON/EVAL_EPOCH已是20轮。接手后已整理为50轮计划、e20停止和E_e20基线。

刚读取该目录state.json时确认：
| 组 | train状态 | 已核对末轮 | 驱动耗时 |
|---|---|---:|---:|
| G1-3 | complete | 20 | 796.2秒 |
| G1-5 | complete | 20 | 1371.7秒 |
| G2-3 | complete | 20 | 380.5秒 |

该快照尚无G3-3的完成条目，也未显示evals/gates完成条目。不能由此判断G3-3是否正在运行；接手先看当前进程、日志、state及checkpoint，沿现有进度衔接，避免重复启动。
state可能在此文件写出后继续更新。

## 2. 用户的工作习惯与约束

- 说人话，先讲结论与下一步，不要大量堆“不能、不代表、不值得、不是而是”式评论。
- 通常由ChatGPT分析、写提示词，本地agent执行；本次用户明确授权ChatGPT直接review和修代码，已实际完成。
- 用户非常在意：模型结构保持不变、全参数训练、运动门控的真实贡献、论文叙事。
- 每组只完整评最终checkpoint；不要自行增加中期val、额外固定motion=1全量消融、多种子、大批附加实验。
- 本地研究始终train训练/val验证；正式test阶段才讨论trainval。
- 重任务串行，启动前检查系统负载和峰值内存；不随便杀进程，不改.wslconfig。
- 保留已有未提交改动、缓存、旧结果与IWG数据；不要reset/clean或删除历史产物。
- 磁盘优先复用；当前状态数据已经生成且验收，3帧/5帧可直接切窗口，无须重采。
- rank_alpha固定0.95，shards已从调参轴移出；SportsMOT目前shards=1。
- 用户倾向保留dropout0.05。现有结果支持将其作为研究配置，尚未实证泛化提升。
- 正式MOT17/MOT20保持现有最佳版本，与SportsMOT机制研究分开。
- 自动调参/定时任务当前未启用。

## 3. 项目和模型究竟是什么

SHUTrack目标：Suppress Harmful Updates，在TrackTrack里学习NSA Kalman写回与EMA外观写入的两个gate。

现架构名称query_film_bilinear，是模型结构版本名，不是一个新加的数值超参数。近期MOT17/MOT20/SportsMOT新架构实验一直使用它。较早IWG是另一套结构。

主要组件：
- HCE：Historical Context Encoder，历史容量5个事件，d_model128、2层、4头，运动/外观各自查询历史。
- RSC：Reliability-aware Suppression Controller；竞争信息FiLM调制，双线性关系头输出运动/外观gate。
- ReID维数2048、appearance_proj_dim256；参数总量1,208,220。
- 运动输入13维：4框偏差、4KF速度、2预测方差、检测分、观测间隔、有无检测；另有3维竞争信息与ReID。
- 当前/历史证据前12维使用训练统计归一化，最后has_detection保持原值；竞争3维也有归一化。

写回规则：
- 现有运动gate在[0,1]，对NSA更新前后KF全部8维均值插值，协方差凸组合并对称化。当前effective_box还有预测框到检测框插值。
- gate=1恢复原NSA更新，gate=0不吸收本次检测；Q/R保持既定NSA规则。
- 外观gate乘原EMA新特征权重，alpha0.95，新特征权重约0.05×检测分×gate。
- 当前不改模型架构、不学Q/R、不采用直接写入B版。

此前B版直接替换NSA/EMA写回已失败并归档，archive/shutrack-b-final；当前主线是A式双门控与D监督。

D监督：运动用t+1同身份GT框DIoU，外观用模板rank＋column竞争监督。motion_harm_weight针对端点显示完整写入更差的事件加权，不是简单把所有运动loss整体乘该数。

## 4. 研究问题与已知机制

SportsMOT运动gate在多个架构下迅速接近1；MOT17/MOT20仍能输出有区分的gate。

已有离线诊断：
- SportsMOT有效运动样本约300457，完整写入比完全不写入更好的事件约86.4%。
- 原一步损失固定g=0/0.5/1约0.2028/0.1644/0.1352，最佳固定常数gate为1。
- 逐事件GT选gate约可把0.1352降到0.1272，约5.9%的运动损失空间。
- 数据偏向快速运动、较低密度、较高检测质量；先验误差常明显大于检测误差，所以大多数事件需要充分吸收检测。
- 有害事件能用现有输入做一定区分，GBM按序列留出AUC约0.63–0.75，不能直接下结论“输入完全没有信息”。
- sigmoid早期饱和削弱后续学习；运动头接近1后运动loss对共享主干梯度很小。
- 加大harm/BCE/排序后反复出现：有害gate下降，同时有益gate也下降；有益事件数量多，累计误伤压过节省。
- 固定常数损失最优在1不等于条件模型的理论最优必然恒1；逐事件oracle已显示可改善。
- 位置/速度四路反事实：一步收益主要来自位置/尺寸，拆位置/速度gate的oracle额外空间小，未优先改结构。
- t+1监督与长期跟踪目标之间存在差距，这是尝试连续监督的动机；3帧/5帧哪种好尚待本轮验证。

在线GT选门诊断：
- SportsMOT原新基线83.343，现场用下一帧GT在0到1步长0.05选运动gate可达84.213，约+0.870pp，AssA约+1.585pp。
- 这是使用GT的逐步贪心诊断，不能部署，也不应说成严格全局HOTA上界。
- 旧BCE模型82.772；同权重motion强制1可到83.529，说明BCE运动抑制的净误伤明显。
- 旧IWG同权重正常83.233、motion1为83.239，运动门控净HOTA贡献约-0.006pp。该IWG训练含90条trainval，val是in-sample。
- 因此“旧IWG整体效果较好”并不说明其SportsMOT运动gate曾有显著正HOTA收益。

## 5. 已做过的关键实验

以下多为用户/本地agent汇报，本助手没有逐个重新跑；真实数值细节以对应comparison.json为准。HOTA表中统一用百分制，变化用pp。

### 5.1 原D监督与架构

- legacy appearance_proj_dim64→256对MOT20改善大，对SportsMOT几乎持平。
- query_film_bilinear对照旧legacy256：
  - MOT20：73.58→79.61，+6.03pp，增益主要在密集歧义场景MOT20-05。
  - MOT17：66.93→67.15，+0.22pp。
  - SportsMOT：83.48→83.34，-0.14pp。
- 正式旧新架构SportsMOT D e200 test75.727；代价监督正式test75.677，略降。

### 5.2 gate标签、代价BCE、排序

- 直接拟合逐事件soft gate，SportsMOT val从83.343掉到82.772，DetA回退明显。
- 冻结共享/外观，仅运动头代价BCE lambda0/.25/1/4快筛约83.343/83.315/83.081/82.797；.25延长200仍约83.300。
- 全参数从基线继续20轮：代价约83.162；加运动排序约83.142。
- 从头全参数200分片轮：代价监督83.477；代价+排序83.339。
- 同一个代价监督checkpoint把motion固定1后约83.556，运动抑制贡献仍约-0.079pp。早先聊天出现83.563和差值不一致，接手以结果JSON为准。
- 所以排序分离度或gate均值下降，尚不足以说明跟踪获益；须看总节省/总误伤及最终指标。

### 5.3 SportsMOT全量驻留A/B/C/D/E

旧SportsMOT shards4每轮仅处理约1/4样本。新shards1每轮全量，20全量轮的样本预算等于旧80分片轮。epoch数字不能直接横比。

| 组 | dropout | harm | 预算/调度 | val HOTA |
|---|---:|---:|---|---:|
| 历史D e80 | 0 | 1 | 旧4分片，80轮 | 83.589 |
| A_e20 | 0 | 1 | 全量20，计划50余弦 | 83.666 |
| B_e20 | 0.1 | 1 | 同A | 83.409 |
| C_e20 | 0 | 8 | 同A | 82.352 |
| D_e20 | 0.05 | 1 | 同A | 83.660 |
| D_e30 | 0.05 | 1 | 沿50轮计划继续 | 83.658 |
| E_e20 | 0.05 | 2 | 全量20，计划50余弦 | 83.294 |
| E_e30 | 0.05 | 2 | 沿50轮计划继续 | 83.329 |
| E_e100 | 0.05 | 2 | 从头全量100，100轮余弦到0 | 83.680 |

E_e100：AssA75.002、DetA93.388、IDF183.703、MOTA98.247、IDSW458、Frag670。
- 有害gate0.9209、有益gate0.9785、分离度0.0576。
- 有害节省0.003814/事件、有益误伤0.000993/事件，净DIoU变化+0.000465（相对全写入更差）。
- HOTA比A_e20高0.014pp；预算和余弦也变了，不能单独归因于harm或长训练。
- 整体HOTA改善由哪个分支造成尚未隔离，不要仅凭一步代理指标断言“全来自外观”。

C harm8：
- 有害gate约0.555、有益约0.817，总误伤大，HOTA下降约1.31pp相对A。
- E harm2误伤较小，但20/30轮仍比harm1差。
- 用户仍希望探索充分训练与监督/加噪，因此做当前四组；不是继续盲扫harm更大值。

Dropout：
- 0.05与0的短预算HOTA几乎相同；0.1较差。
- 用户希望保留一点Dropout，当前0.05固定研究。
- 更早代码默认0并不等于bug；没有实证说明模型小就绝对不需要Dropout。

### 5.4 DanceTrack支线，已转回SportsMOT

- GT现场选门val62.360→73.304，+10.944pp，25/25改善；使用GT的诊断空间大。
- 原D训练train40，e80/e100/e200正常门控约63.056/62.755/62.618。
- e80 motion固定1约63.308，学习运动gate仍几乎恒1。
- trainval65、保持200轮余弦到e80停止，test+AFLink官方65.700，低于历史最佳66.500。
- 用户决定回SportsMOT主线，当前不扩DanceTrack工作。

## 6. MoTTrack借鉴及后续优先级

MoTTrack参考代码：mamba_kalman_filter/dataset.py、config.py及其训练器。
已核对的观测增强参数：

| 数据集 | 中心std/像素 | 宽高std/像素 | 基础miss | burst概率 |
|---|---:|---:|---:|---:|
| SportsMOT | 1 | 1 | 0.004 | 0 |
| MOT17 | 1 | 1 | 0.02 | 0.02 |
| MOT20 | 3 | 2 | 0.02 | 0.02 |

共同细节：
- cxcywh像素单位独立高斯噪声，宽高最小1像素。
- 面积小于2500，miss乘2；边界20像素内乘1.5；基础概率上限0.5。
- burst长度3–6帧；连续missing的保持概率0.4；片段首帧保留。
- seed/epoch/anchor决定可复现随机流，不随微批次分组改变。
- 现SHUTrack增强来源是现有真实检测加噪，保持GT/GMC和原缓存不变。MoTTrack部分模式是GT合成观测，不能说两者数据来源完全相同。
- MOT17/MOT20共享增强配置组件已经实现；相应正式训练实验和效果验证尚未完成。

研究优先级：
1. 有害权重与充分训练（E100已做）。
2. 连续状态监督（当前四组的一部分）。
3. MoTTrack观测增强（当前四组的一部分，之后讨论其他数据集）。
4. c可靠性调节：候选R有效=R_NSA/c，范围[0.2,5]，c=1中性。最新讨论见第1节，仍待用户决定实现。

用户曾讨论普通KF vs NSA、让有益更新加强吸收。此前的监督和增强实验沿用原NSA及运动gate。

## 7. 本助手直接review并修复的代码问题

用户因原5帧训练约306秒/轮、三组约30小时而停止任务，并授权直接检查修改。

### 7.1 真实错误

1. 重建13维evidence后用raw值覆盖已归一化current_evidence，模型本身不再归一化。
   - 原CPU检查raw与期望归一化输入最大差33.57。
   - 必须对前12维用训练mean/std归一化，最后has原值。
2. KF状态递推了，但窗口内历史证据、track ReID/EMA、检测mask等仍使用旧全写入缓存。
3. 有害标签仍取缓存端点，与动态状态/增强观测不一致。
4. 外观监督按窗口位置平均会放大小量短窗口，现仅锚点一次。
5. G2虽然称单步/anchor，却运行五次模型并保留五步模型图。
6. 原增强阈值1024/10与已核对MoTTrack的2500/20不同，burst参数未真正完整实现。
7. 旧checkpoint缺训练实现版本隔离；错误归一化权重不适合作修复版续训来源。

### 7.2 性能问题及修复

- 逐锚点GPU状态字典、小切片/cat引入大量反向节点，改为批量状态张量。
- bool(cuda_scalar)、逐步sum转int、每batch统计拷CPU导致同步，改为设备统计轮末读取。
- 窗口索引预计算，增强向量化，小字段prefetch/pin_memory。
- 每个序列原始特征暂存GPU并索引组批，避免每帧重复完整CPU gather/H2D；不缓存模型编码特征，不冻结梯度。
- KF固定矩阵缓存，batch求解使用solve_ex避免每步主机错误检查。
- 保存并恢复模型/optimizer/scheduler/Python、NumPy、Torch、CUDA RNG。
- resume检查目标/预算/配置；现有输出以resume或独立目录管理。

### 7.3 新实现的重要边界

- 标记window_implementation=packed_replay_v2。
- 标记association_context=frozen_replay_candidates。
- 窗口内KF均值/协方差、EMA、证据及mask动态变化。
- 离散匹配和竞争候选上下文仍来自固定采集回放，不是整套在线关联的端到端可导训练。
- GT用于监督和动态harm权重，不作为模型输入。
- 当前micro_batch=1024与有效batch相同。若主动缩小micro_batch，按样本比例累加微批损失均值；有效标签比例不同时，不保证与整batch比值损失完全一致。

### 7.4 直接核验结果

修复时的SHUTrack测试：217 passed、2 skipped。
- KF update/GMC/predict与numpy路径最大绝对误差约2.384e-7。
- 单步重建模型输入与原训练器最大差7.868e-6。
- 单步总目标差1.073e-6。
- 批量与逐样本回放、CPU特征和GPUbank数值一致。
- 后续预测误差对前部gate的梯度非零。
- G2无噪固定历史预热后锚点evidence与原缓存最大差2.718e-5。
- dropout0.05下恢复RNG/AdamW/scheduler，下一步loss差0、参数最大差0。

当时按100轮调度、只跑完整一轮45序列的性能验收：
| 组 | 秒/全量轮 | CUDA峰值MiB | RSS峰值GiB |
|---|---:|---:|---:|
| 5帧full无噪 | 71.851 | 2556 | 5.077 |
| 5帧历史/anchor加噪 | 21.023 | 969 | 5.016 |
| 5帧full加噪 | 69.676 | 2556 | 5.076 |

原一步约22秒/轮；连续5帧仍增加计算，约3.2倍，但原306秒的多余开销已大幅削减。
3帧最新20轮运行已经有G1-3约796.2秒的驱动记录，均值约39.8秒/轮；该数包含驱动开销，与单轮测量口径稍异。

## 8. 代码与报告位置

核心：
- tools/train_window_groups.py
- shutrack/training/window_replay.py
- shutrack/training/recurrent_motion.py
- shutrack/training/observation_noise.py
- tools/run_sportsmot_g123_experiment.py（当前四组20轮主入口）
- tools/run_sportsmot_mechanism_formal.py（旧正式阶段wrapper，当前停用）
- tools/run_sportsmot_abc_experiment.py（被主入口复用的执行/评估基础）
- tools/train_supervision_groups.py、tools/paired_training_input.py
- tools/sportsmot_supervision_pilot.py、tools/generate_trainval_paired.py

核验：
- shutrack/tests/test_window_replay.py
- tools/verify_window_training_review.py
- tools/verify_window_resume_pipeline.py
- tools/verify_window_state_data.py
- tools/verify_final_data.py
- tools/check_window_input_normalization.py（旧错误协议的诊断，已明确标记archived）

报告：
- docs/shutrack_window_training_audit_20261124.md
- docs/shutrack_mottrack_training_comparison_20261123.md
- TODO.md
- outputs/agentguard/experiments/window_training_speed_audit_20261124/

注意版本：
- TODO.md及审计文档中“各100轮→自动正式提交”段落是历史安排，已被本文件第1节覆盖。
- 当时217测试通过是在packed_replay_v2修复时；后续本地agent又改了四组20轮驱动。不要把旧测试结果说成之后所有驱动改动均已重新完整验收。
- verify_window_resume_pipeline.py原有100轮/旧组名硬编码可能需要适配当前20轮驱动后再运行。
- formal wrapper原基于100轮和旧组名；当前train_command已变为planned50/stop20。未来用户授权正式阶段时要重新核对wrapper，不能直接套旧formal入口。
- 这些是接手应关注的版本衔接事项，本次只写交接文档，未擅自改当前正在运行的任务。

## 9. 新状态数据的真实位置、占用与完整性

### 数据目录

train45：
outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data

trainval90：
outputs/agentguard/experiments/sportsmot_mechanisms_formal_20261124/data

| 数据 | 样本 | 事件 | 实际文件块占用 |
|---|---:|---:|---:|
| train45 | 305631 | 306245 | 3.510525 GiB |
| trainval90 | 593762 | 594801 | 6.801888 GiB |

合计10.312412GiB。两份各保存一份train数据，按inode检查无文件级链接去重。新增8类状态数组仅102.848MiB/199.759MiB，其他主要是event_track和samples/rank。权重、推理结果不计入这张表。

每序列包含：
samples.npz、summary.json、event_evidence、event_track、event_detection_index、event_has_detection，
以及event_state_mean、event_state_cov、event_det_measurement、event_det_score、
event_warp_next、event_gap、event_frame、event_target_row等npy。

### 采集bug是否已解决

初次开发时lean路径走_lean_register，绕过原register闭包，导致新增状态数组为空。
本地agent随后改为按ordered_keys/event_index，从event_states统一构造event_state_array，再生成目前两份数据。
目前数据已由本助手重新核验，不是初版空数组。

直接检查结果：
- 45/45、90/90序列齐全，complete通过。
- 样本→事件→frame/detection、反向event_target_row绑定正确。
- 样本所需KF状态缺失0。
- 新数组无NaN/Inf；样本绑定先验框尺寸和协方差对角为正。
- 全量有效端点复现：train300457行，trainval582144行；最大框误差小于0.000191像素。
- 原始运动证据重建误差最大3.571e-5。
- 614/1039个未绑定监督样本的初始化历史事件没有完整状态；window_index按target_row<0截断，它们不进入状态递推。其原缓存evidence仍可作HCE上下文。
- trainval正式验证：data_ok=true；593762个键，34680次检测索引抽查、20486次column抽查，全无错位，finite_bad0。
- 3帧/5帧直接取现有窗口，增强在训练中动态产生，无需重采。

核验结果文件：
- outputs/agentguard/experiments/window_training_speed_audit_20261124/state_data_integrity_and_footprint.json
- outputs/agentguard/experiments/window_training_speed_audit_20261124/verify_state_rich_trainval90.json
- outputs/agentguard/experiments/window_training_speed_audit_20261124/review_verification.json
- outputs/agentguard/experiments/window_training_speed_audit_20261124/resume_pipeline_verification.json

## 10. 旧实验权重和输出

当前四组20轮：
outputs/agentguard/experiments/sportsmot_mechanisms_e20_20261124/
  state.json、preflight.json、logs、runs/<组>/groups/appearance_row_column、
  evals/<组>_e020、gate_targets、最终report。

E_e20同预算基线：
outputs/agentguard/experiments/sportsmot_de_20261123/runs/E/groups/appearance_row_column/checkpoints/epoch_020.pt

E_e100历史参考：
outputs/agentguard/experiments/sportsmot_e100_20261123/runs/E/groups/appearance_row_column/checkpoints/epoch_100.pt

A/B/C：
outputs/agentguard/experiments/sportsmot_abc_20261123/

D/E：
outputs/agentguard/experiments/sportsmot_de_20261123/

D/E e30：
outputs/agentguard/experiments/sportsmot_de_e30_20261123/

旧错误归一化G1：
outputs/agentguard/experiments/sportsmot_mechanisms_20261124/runs/G1/
最初审计读到e89。该目录中的data可用，旧训练权重按错误协议归档，勿用作修复版resume。

本助手修复后完整单轮验收：
outputs/agentguard/experiments/window_training_speed_audit_20261124/full_epoch_G1
full_epoch_G2、full_epoch_G3同级。
它们仅短验收，不是100轮正式成果。

曾规划但已被最新20轮协议取代的输出根：
outputs/agentguard/experiments/sportsmot_mechanisms_review_v2_20261124
以及sportsmot_mechanisms_formal_review_v2_20261124。
是否存在额外本地运行产物，以本机实际为准。

## 11. 正式成绩与其他数据集背景

用户报告的当前正式最佳：
| 数据集 | 版本 | HOTA | 备注 |
|---|---|---:|---|
| MOT20 | 原新架构正式e100 | 66.637 | 继续保留 |
| MOT17 | MOT20 e100初始化→正式e50 | 67.827 | 继续保留 |
| SportsMOT | 仅train训练的旧D e80 | 75.840 | 提交1064937，Finished |

SportsMOT其他：
- trainval正式e80：75.774，提交1064222。
- 原新架构D e200：75.727。
- 代价监督trainval正式：75.677，提交1063982。

MOT20 e80正式66.635，基本持平但未超过e100。
MOT17由MOT20 e80初始化重训50轮67.763，未超过67.827。
低LR续训：
- MOT20原e100追加50分片轮，lr1e-5→0，66.587。
- MOT17原e50追加50单分片轮，lr1e-6→0，67.689。
均未刷新最佳。不要默认多训就更好。

MOT17/MOT20调参工作流：
- 用户希望agent基于结果自主分析自己模型的超参数，日常额度使用后需留账本。
- 后来倾向少侵入TrackTrack原关联/检测阈值，多研究自有训练参数。
- 配额UTC自然日，上海时间08:00刷新，平台can_make_submission为最终依据。
- 有jojo0和Edge账号，token只从本机配置文件读取，绝不写入提示词/报告。
- 目前自动调参循环和定时任务关闭。
- 这轮SportsMOT已明确等待用户决定，不包含任何自动提交。

## 12. 接手顺序

1. 阅读本文件第1节，确认最新四组20轮协议和用户“报告后等决断”的边界。
2. 先看sportsmot_mechanisms_e20_20261124的state、当前日志、checkpoint与进程。前三组完成是本交接写出前的快照，后续状态可能已推进。
3. 延续现有串行流水线。仅在阶段真实完成时跳过，避免重复训练/推理。
4. 核对最终e20每组仅一次val及门控分析；复用已存在且完整的评估。
5. 报告用E_e20同预算基线；E_e100单列历史参考。明确3帧/5帧计算成本与收益。
6. 对少数序列大幅翻转、总误伤/节省、HOTA/AssA/DetA同时分析；不把gate均值下降或排序准确率提高直接当作成功。
7. 报告完成后等待用户决定是否延长训练、正式trainval和test/post/提交。
8. 如发现需要修复驱动的边界问题，保留当前任务和结果，先核对实际影响，避免误停/重跑。

用户要的是能接续工作的事实记录。后续以代码、日志和结果JSON为证据更新，不要沿用聊天里曾经被覆盖的100轮/自动提交计划。
