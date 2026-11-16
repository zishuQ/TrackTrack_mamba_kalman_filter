# SHUTrack v2（HCE + RSC）重构实施说明（2026-11-12，2026-11-16 更新）

## 0. 任务目标与本轮交付

项目：/home/shang/workspace/TrackTrack。
当前讨论确认的方向：SHUTrack = Suppress Harmful Updates。接受匹配不意味着运动状态与外观模板都应该完整更新。采用“结构化证据 -> HCE历史上下文 -> RSC当前跨模态融合 -> 双连续Gate”。

### 0.0 最终命名与边界（2026-11-16，覆盖旧 HUC/RUC 名称）

- **HCE — Historical Context Encoder（历史上下文编码器）**：只编码当前之前的 5 个历史事件；共享时序编码器（d_model=128、2 层 Transformer、4 heads）编码为历史事件序列，两个可学习 query 分别读出运动与外观历史上下文 `h_m, h_a`。HCE 不输出 gate，不预测离散策略。
- **RSC — Reliability-aware Suppression Controller（可靠性感知抑制控制器）**：以 `h_m, h_a` 为 query，以当前运动、外观、匹配上下文 token 为 key/value，做跨模态注意力与表示空间残差融合，由两个独立 sigmoid 头输出 motion gate 与 appearance gate。RSC 是控制器整体，跨模态注意力、表示级残差融合与双 gate 输出头都是其内部组成，不是第三个组件。
- 两个 gate 不使用 softmax，不要求和为 1；不再存在 base gate、refined gate、correction、policy、cue、risk 或五策略原型。
- 旧实现中的 `HUC`/`RUC`/`IWGRGCMA` 历史命名保持原样，不进行全仓库替换；新代码、配置、日志、测试和本文档统一使用 HCE/RSC。下文中旧 HUC/RUC 表述仅作为历史映射说明保留。

### 0.1 v3 最终输入与信息流（2026-11-16 修订，覆盖下文 18+2/margin 描述）

- 运动与上下文合并为 13 维 evidence：4 个框偏差（旧 27:31）、4 个 KF 速度（39:43）、2 个中心预测方差（43:45）、检测置信度（9）、观测间隔（59）、是否有检测（0/1 flag）。删除宽高方差、快照历史长度、selected margin 及其有效位；保留独立的 has-detection 与历史 padding/reset mask。
- 新增可选竞争分支 3 维：旧 63 维缓存第 15、21、25 列（0-based）——行前两名代价差、列前两名代价差、当前检测与候选池内其他检测的最大 IoU。在线使用同一候选池、迭代前 `final_cost`、相同归一化重现；最大 IoU 排除自身，单候选沿用旧特征的 `second_min=1.0` 边界。训练配置 `--competition off` 可关闭该分支，checkpoint 记录 `competition_enabled` 并在加载时检查。
- HCE 仍只读最多 5 个历史事件；每个历史事件由 evidence 与外观两路编码融合。RSC 以两个历史读出为 query，读取当前 evidence、外观、竞争三个 token，输出两个 gate。
- 历史重置按事件幂等：同一事件只重置一次，事件段号与缓冲区段号一致；第 50 帧历史为空，第 51 帧模型实际 mask 恰好读到第 50 帧。失配仍记录为真实历史事件，无匹配只预测、重匹配再算 gate。
- 提前启用为独立选项（`--shutrack-early-enable`）：第 1 次观测初始化，第 2 次匹配原样更新并记录更新前证据，第 3 次匹配在 New 路径首次应用 gate；竞争特征取 New 路径自身候选池；候选轨迹移除时清理缓冲。默认两组竞争分支共用 `observation_count >= 6` 的启用条件。
- 本轮不迁移递推训练、不重建缓存；写回仍为原 `Track.update_with_gates`，损失仍为加权 BCE + 0.1 MSE。详细命令见 `shutrack/README.md`。

本轮实施：新增独立架构与只读数据适配路径，完成训练/在线接线、必要测试、少量真实数据smoke，交付可运行训练命令。
不要自动启动三数据集正式长训练，不执行检测/ReID提取、轨迹再采集或rollout标签再生成。
不要改动MoTTrack仓库。

先检查当前AGENTS.md、工作树、分支与已有代码。2026-11-12已从25c2e31建立并切换到独立分支feature/shutrack-v2；agentguard、main和原feature/huc-ruc-tta-style保留。只在feature/shutrack-v2实施，不强行reset/checkout覆盖后来产生的工作，保留现有未跟踪文件及其他工作。新架构id为shutrack-huc-ruc-v2。

### 0.1 用户明确要求：独立分支与独立代码边界

Git分支隔离与代码依赖隔离必须同时做到。使用同一checkout节省空间，不创建worktree或复制仓库/数据。
- 新实现放在独立顶层shutrack/目录，建议shutrack/src/shutrack/下分features、data、models、training、runtime、integration，配置/测试/小型入口脚本也归属shutrack/。
- 新模型、loss、训练循环、checkpoint loader、runtime和统计接口独立，不向旧IWGRGCMA/旧训练器内部不断增加v2条件分支。
- 禁止继承旧IWGRGCMA、调用旧policy/correction训练链或通过伪造旧输出字段适配旧runtime。
- 允许在明确的兼容边界只读复用旧CompactEventCacheReader、检测缓存reader、标签读取函数及稳定底层数据类型。将这些依赖集中在例如shutrack/data/legacy_cache_adapter.py；新模型与loss不import agentguard的models/training/runtime。
- 不调用可能自动build的旧Dataset构造器；独立适配已有只读索引/数组，或只调用已确认无写副作用的读取函数。
- TrackTrack端只增加必要的入口选择/工厂分派到新adapter；默认旧路径行为不变。新adapter交付gate，仍使用既有update_with_gates；不修改该更新算法。
- 旧loader只继续负责旧checkpoint，新checkpoint由新loader负责。不能为了新架构改写旧checkpoint格式或旧feature/label schema。
- 若发现需要改动共享底层接口，应在实施报告中明确修改理由、最小范围及旧路径回归结果；不借重构顺便重写旧系统。
- 提交时按新features/data adapter、model/loss、integration/tests分组，精确暂存本任务文件；不使用git add .把已有未跟踪文档或审计脚本一起提交。不合并到agentguard/main，不推送，除非后续用户要求。

## 1. 硬约束：不生成新的大型数据文件

用户硬盘不足，但可容纳模型文件。
- 现有检测缓存、事件缓存、标签、索引、归一化文件均作为只读来源。
- 不创建整套新的特征数组、timeline memmap、样本窗口副本或数据集副本。
- 不使用复制目录、新worktree、搬数据到/tmp等方式绕开约束；/tmp同样可能占本地磁盘。
- 允许少量代码、配置、训练统计、模型checkpoint及小型测试产物。
- 在batch或样本读取时转换特征；缓存读取使用有限容量LRU，不能把全部数据装进RAM。
- 禁止因为新模型id、输入维度或schema不匹配而自动调用旧的cache/index重建器。
- 若必需输入或现有索引缺失，明确报告缺失字段/路径及影响；不要默默启动数据生成。
- 新feature/model schema与旧source-cache schema分开。保留旧source版本和hash，新增模型输入schema；不要修改全局旧FEATURE_SCHEMA_SHA256来让整个数据链失效。
- 不删除、覆盖旧数据或旧checkpoint。

## 2. 架构契约

### 2.1 时间边界

上下文仍是6个事件（不是必须连续6帧）：
HCE只接收[t-5,...,t-1]的5个过去事件；RSC接收当前事件t。
保留缺测事件、真实frame gap、padding、分段reset。不能跨track、跨sequence或跨reset取历史。
所有当前输入均在当前写入前采集；当前检测和关联结果可以使用。标签中的未来GT/oracle不可进入输入。
冷启动使用显式null-history token/表示，并配正确mask，避免全masked attention产生NaN。无历史但有检测仍允许用当前证据生成gate。

### 2.2 结构化事件编码器

共享训练与在线的特征定义，构成：
- motion_token：12个核心运动标量编码；
- appearance_token：更新前track ReID与当前det ReID的投影及关系编码；可复用投影后concat(track,det,abs_diff,product)的形式；
- context_token：当前检测质量、关联歧义、历史有效性编码。
历史事件也使用同一原始证据编码规则，融合成历史event token。历史token不包含模型policy、cue/risk或旧gate预测。
缺测时保留可用预测状态，用mask标记检测/外观不可用；padding和真实缺测不是同一种情况。无检测的det ReID零占位必须配mask。

### 2.3 HCE

默认d_model=128，2层Transformer，4 heads；其他合理小网络超参数在配置显式记录，不扩展为架构搜索。
对5个过去event编码为H。
使用两个可学习query从同一个H读取：
h_m = Read(q_m,H), h_a = Read(q_a,H)。
共享历史主干，不复制两套完整时序网络。
HCE不输出base gate，不预测五策略。

### 2.4 RSC

当前token为o_m,o_a,r。
以[h_m,h_a]为两个query，当前[o_m,o_a,r]为key/value，进行一层跨注意力及必要FFN/归一化，输出delta_m,delta_a。
允许每个query访问两个模态与context，显式modality embedding区分token。
不再在RSC内建立第二套6事件时序摘要。
在表示空间融合：
z_m = h_m + W_m delta_m
z_a = h_a + W_a delta_a
然后独立解码：
m = sigmoid(D_m(z_m))
a = sigmoid(D_a(z_a))
不使用softmax耦合两个gate。
第一版不额外增加pre-gate风险头、SNN、乘性第二道gate或更多架构变体。
初始化可采用普通/小幅残差投影；不要设置两个串乘因子同时为零阻断训练。不能宣称新架构初始化等于旧HUC，历史/当前划分已经改变。RSC 的跨模态注意力、表示级残差融合和双 gate 输出头共同构成一个组件，不拆分为第三个组件。

### 2.5 更新执行

沿用当前track.py:update_with_gates的均值、协方差、effective_box、EMA及其他字段更新语义。只换gate来源。
gate=1对应原始更新，gate=0对应该通道不写入；不要把它误写为整个track完全不变。
当前无检测时applied gate为[0,0]，照常维护缺测事件和track生命周期。
不要更改关联算法、KF噪声、EMA公式、检测阈值和轨迹管理规则来配合新模型。

## 3. 输入v2建议定义与数据来源

固定第一版为18个核心标量，加2个竞争margin有效性bit；padding/reset与检测存在性仍有显式mask接口。不要为了凑维度新增证据。

### 3.1 motion：12维

旧scalar63的0-based索引：
- 27:31：中心偏移2维、宽高log ratio2维；
- 39:43：归一化KF速度4维；
- 43:47：旧定义的归一化log协方差对角线4维。

第一版保持这些数值的现有定义，避免同时更改尺度约定。注明协方差实际使用旧image-scale定义，不宣称已改成目标尺度归一化。
既有tensor若已经归一化，不可再次归一化。

### 3.2 context：6个核心量+2bit

- det_score：旧索引9；
- selected row margin；
- selected column margin；
- history length：旧索引58；
- observation gap：旧索引59；
- has_detection：旧索引26；
- row_margin_valid；
- col_margin_valid。

新margin固定定义：对该事件记录的关联候选池，使用同一raw cost矩阵，margin = 最佳其他候选cost - 被接受匹配cost。正值表示所选匹配比其他候选更优，负值表示存在更优竞争者。
必须通过recorded track row与accepted_detection_index查到所选元素；排除自身、非有限值、padding、记录明确标识的不可行项。不要重建一套新的关联算法。
若没有其他候选，margin=0且valid=0；不要用任意大常量模拟“确定”。
不要把旧索引15/21的top1-top2差直接当作selected margin，除非逐事件已证明所选就是最优。
记录候选池语义是“缓存记录的候选池”，不能宣称它必然等于某一迭代时刻的动态活跃池。
若缓存矩阵缺失，不能把缺失数据静默当作无竞争；报告范围。必要时提出明确的无margin精简输入方案，待用户选择，不生成新事件。

### 3.3 appearance

track_feature使用该事件更新前的模板；det feature通过accepted_detection_index读取既有检测ReID。
不能用最终轨迹模板或未来模板替换；不得用GT身份原型作为模型输入。
history里的每个事件保持当时自己的template与det，不用当前模板广播覆盖过去。

### 3.4 归一化

定义未变的维度可从既有训练归一化统计选择对应项，需核实来源split正确且与在线一致；bit/mask不做普通z-score。
新margin可以只遍历训练划分流式拟合均值/方差（含valid过滤、稳定方差下限），只保存小型统计json/npz；不保存整套新margin数组。
两侧共用schema、顺序、默认值、normalizer；不得一侧使用旧已归一化scalar，一侧直接拼原始scalar。
新分组LayerNorm/MLP不能依赖被删除的63维字段。不能通过先过旧63维LayerNorm再取子集而让已删除特征继续影响模型。

### 3.5 实施时的真实数据核查（2026-11-16）

- 实际可用的训练源只有 packed `outputs/agentguard/train_data/{MOT17,MOT20,SportsMOT}`：`data.pt` 含 `event_indices [N,6]`、`safe_gate_target`、`valid_channels`、`sample_weight`、已归一化的 `timeline_scalar_feats [T,63]`、`timeline_has_detection`；`reid_features.npy [T,2,2048]` 含更新前 track 模板与 det ReID。三者均 `context_size=6`、`reid_dim=2048`。
- `outputs/agentguard/detection_cache` 在线可用。
- 源 compact event cache（`events_*.pt`/`associations_*.pt`/`states_*.pt`）、compact labels 与关联矩阵在当前 checkout 中不存在；packed 数据也不含 `raw_cost`、`accepted_detection_index`、`association_track_row`。因此真实 packed 训练中 row/col margin 与两个 valid bit 恒为 0，`margin_enabled=false`、`margin_source="unavailable_packed_train_data"`、`margin_valid_rate=0.0` 显式写入数据集元数据、checkpoint 与日志；在线推理严格读取 checkpoint 的 `margin.enabled`，为 false 时即使拿到完整关联矩阵也强制两个 margin 与两个 bit 为 0，保证训练/在线输入契约一致。`--margin-mode required` 会在启动时报告缺失字段。margin 计算、源缓存只读适配器与合成候选池测试均已实现，待只读缓存恢复即可启用。

## 4. 只读数据适配路径

优先复用既有IWG_RG_CMA timeline/index与compact labels，在读取后做in-memory adapter。
相关现有文件：
- agentguard/src/agentguard/data/compact_event_cache.py
- agentguard/src/agentguard/data/cache_reader.py
- agentguard/src/agentguard/data/compact_iwg_labels.py
- agentguard/src/agentguard/datasets/iwg_rg_cma_dataset.py
- agentguard/src/agentguard/datasets/timeline_utils.py
- agentguard/src/agentguard/features/builder.py
- agentguard/src/agentguard/features/scalar.py

事件缓存格式支持pre_update_state、track_feature、accepted_detection_index、association矩阵和定位索引；这是代码能力，实施前只读抽查实际数据是否完整，不能假定所有磁盘缓存都完整。
保留原event_keys/shard/offset对齐，不能因特征精简改变事件身份。
旧五策略/cue/risk标签文件可留着，新loss不消费；兼容旧reader验证这些文件不等于模型使用这些监督。
禁止默认构建新timeline_scalar_feats.npy、复制timeline_track_feats或重打包全部labels。
新数据adapter必须走真正的reuse_only/read_only路径，先检查现有文件，再加载；不实例化有自动build副作用的旧Dataset，缺失时给出明确错误，不回退build。
数据读取需要关联记录时按sequence/shard有限缓存。控制dataloader workers和prefetch，避免多worker复制整个数据集。
任何未来GT/oracle仍仅属于旧标签来源，不得加入新feature adapter。

## 5. 标签和损失

复用同一事件上的safe_gate_target、valid_channels、sample_weight。
第一版loss沿用实际旧gate项：按通道有效性和样本权重归一化的BCE + 0.1*MSE。可以用gate_logits计算数值稳定的BCEWithLogits，sigmoid gate计算MSE，确保含义一致。
删除新架构中的policy KL、cue/risk BCE、base/final双重gate loss、residual/revision/no-harm/correction abstain项。
目标是原有rollout收益的代理监督，不命名为校准有害概率或连续动作的真实最优解。
不更改rollout时域、收益函数、safe target映射、样本权重和数据划分。
不要直接调用旧compute_iwg_without_cma_loss而保留其policy/cue/risk字段依赖；新增独立、简洁的loss接口。
全无有效label的batch处理需稳定且明确。
新架构端到端训练，不能沿用旧RUC输入.detach()的梯度隔离。监督应能回传到HCE、RSC及有效编码器。

## 6. 模型/运行时/CLI兼容

在独立shutrack/目录新增特征模块、只读数据adapter、模型、loss、训练入口、checkpoint loader及runtime。组织方式遵守0.1的隔离要求，不把新实现塞入旧agentguard模型/训练/runtime文件。
最小共享入口只负责选择新旧adapter，不承担新模型的具体实现。

必须审查但不应直接塞入新业务逻辑的旧接线点：
- runtime/manager.py：当前固定打包base/refined/correction/policy/cue、使用policy confidence fallback；
- runtime/buffers.py；
- runtime/statistics.py；
- training/train_iwg_rg_cma.py:load_iwg_rg_cma_checkpoint；
- TrackTrack适配器、CLI配置、训练stats里对旧输出字段的硬依赖。

新模型核心输出gate_logits、gate及可选小型诊断；不要伪造五类均匀policy或空risk来继续旧fallback逻辑。
通过最小入口分派到独立的新runtime adapter、loader和统计路径；保留旧checkpoint加载和旧架构行为。
新架构不使用--iwg-rg-cma-output base或rg_cma_alpha做新语义的隐式映射；显式拒绝不适用的非默认选项，旧模式仍正常。
新架构若要求policy fallback_threshold>0，明确报不支持；不能通过虚构policy_probs让它运行。
可保留运动/外观通道置1的诊断开关，但不需要本轮扩展消融套件。
第一版默认把启用条件对齐旧缓存的采集规则（`observation_count >= 6`），避免模型在训练分布之外的早期轨迹状态上直接决策；`--shutrack-min-observations 1` 保留为显式早冷启动选项，必须有单独的分布验证，不能仅凭 padding/null token 无 NaN 就宣称可用。若原集成对新生轨迹有特殊规则，明确保留/调整的边界并测试。
推理按真实当前轨迹状态实时构建输入，不能从旧训练事件缓存注入状态。
保存checkpoint包含architecture/version、feature schema、特征顺序/normalizer、context=6、hidden dims、源缓存/标签来源及训练配置。
新模型需要重新训练；禁止strict=False静默加载旧模型并声称resume成功。迁移少数兼容层若实施，需明确列出并作为独立选项，默认不做。

## 7. 必须完成的验证

这些测试验证实际契约，不写仅重复实现的无效测试：
1. 原始事件与离线adapter/在线builder对同一事件产生相同特征，包括selected不是row/col最优、无其他候选、无检测、缺失历史、reset。
2. 当前写入前状态与标签严格对应；未来输入变化不影响当前gate；跨轨迹/跨reset无历史污染。
3. 长于6事件的forward_sequence与逐事件在线forward在eval下等价；padding、缺测、冷启动和重激活均无NaN。
4. gate范围和通道含义正确；无检测输出[0,0]；HCE/RSC在有效非退化样本上都有梯度；无policy/cue/risk/correction参数和监督。
5. checkpoint roundtrip输出一致；旧checkpoint加载路径不回归；新架构对不适用旧参数给出明确错误。
6. 只读数据路径的保护测试：把build/reindex/label-generation入口替换为失败桩，确认读取与训练小样本不会调用；小fixture源目录只读仍可使用。真实小批读取前后核对source文件未新增/修改，不能对全数据做昂贵hash扫描。
7. 单个极小真实训练smoke（例如10-20 steps、受限workers）和一小段在线推理，报告有限loss/梯度/输出及实际磁盘新增量。若数据/GPU/依赖缺失，准确报告阻塞，不伪造通过。
8. 运行受影响的既有测试；若现有完整agentguard套件成本合理，再运行其回归。完成后不因无关性能探索重复跑测试。

## 8. 保存与交付

训练存储采用有界保留，例如last.pt（含optimizer以便resume）与best.pt（仅模型和必要元数据）；不每epoch累积，不保存每事件attention/gate大数组。
仅在本次新run输出目录内替换自身last/best文件，不删除其他run文件；考虑原子保存的短暂额外空间。
日志使用标量汇总，attention诊断默认关闭。
本轮交付：
- 文件改动清单和最终架构/输入schema；
- 测试与小smoke的真实结果；
- 实际识别的只读数据/标签/索引路径以及复用方式；
- 一条基于仓库真实CLI和既有数据路径的正式训练命令，以及推理命令（尚未验证的命令应标注）；
- 明确列出未解决问题与是否需要字段缺失的输入降级决定。
不要自动开始正式全量训练、生成新轨迹/标签、修改其他项目或开展四路线对比。
