# 连续窗口训练速度与输入一致性审计（2026-11-24）

前半部分保留最初只读审计记录。用户随后停止旧训练并授权直接修复；已完成修复、真实数据核验及三组完整单轮训练，结果见下文。

## 实际状态与成本

读取时G1 checkpoint已经到epoch89，贴出的epoch4进度为旧快照。最近所读epoch88耗时286.9996秒，早期约303-308秒；321个batch、305631锚点、1515472个transition，约4.96次递推/锚点。模型CUDA峰值约1.86GiB，RSS峰值约5.14GiB。

相对原一步约22秒为约13-14倍。5次模型调用增加计算量，但额外开销的占比还需分段计时；不能仅用“保留计算图”解释全部差距。

## 代码中已定位的额外开销

tools/train_window_groups.py：
- 455-457：每个batch逐锚点重建窗口链；固定窗口索引可以预计算。
- 468-469、488-489、575-577：GPU状态拆为逐样本字典，逐帧重新cat并逐样本切片写回。每轮约151万次transition，对mean/cov产生数百万小切片，反传图含大量SliceBackward/CatBackward节点。应改成批量状态张量与有效索引/mask更新。
- 523-525：anchor列表里对每个样本执行bool(has_det[local])，CUDA标量读取会同步。has_det原本来自CPU数组/增强计划，可在CPU一次形成anchor索引。
- 573-574：逐帧int(cuda_tensor.sum())；339-343：逐batch转float；592：逐batch拷贝门控统计到CPU。统计可以设备端聚合，轮末统一搬回。
- 487、518：窗口每一步都重新调用完整batch_tensors并同步to_device；没有沿用原prepared_batches预取/pin_memory通路。非锚点也准备rank/column数组，尽管外观损失只在锚点计算。可打包窗口数据、预取、异步搬运，外观监督字段只在需要的锚点准备。
- G2使用window_supervision=anchor但window_frames=5，依然执行五步模型与状态反传，所以其耗时不是普通一步增强的耗时。
- build_plans每个epoch每个锚点重新构造RandomState和逐事件增强；后续可批量预生成，但须保留明确可复现语义。
- recurrent_motion.py每次重建固定F/H/单位阵；可以缓存/批量构造。linalg.solve的GPU同步/求解开销需要profile后决定优化方式。

## 已实测确认的输入归一化问题

位置：tools/train_window_groups.py:516-520。

base.batch_tensors已经把current_evidence前12维归一化，但新代码把它直接替换为evidence_from_state返回的原始13维证据。model_inputs只透传，模型forward没有重新应用checkpoint的均值/方差。

执行：
.venv/bin/python tools/check_window_input_normalization.py

实际CPU检查（G1 epoch89 checkpoint，首序列32行）：
- 原始重建证据与原始缓存最大差：2.5928e-6。
- 当前raw覆盖值与模型期望的归一化输入最大差：33.5700。
- 对重建证据补同一归一化后，与期望输入最大差：4.4346e-5。
- 一个中心方差特征原始约-10.98，期望输入约1.225。
- 检测分原始0.97416，期望输入约0.58272。

产物：
outputs/agentguard/experiments/window_training_speed_audit_20261124/normalization_check.json

因此“13维重建与原始缓存一致”尚不足以核验最终模型输入一致。应在保持可导的前提下，用训练/运行时同一统计对重建证据前12维归一化，第13维has_detection保持原值。该修复改变实际训练输入；旧G1结果和修复后实验需要区分，不能直接当相同训练实现接续。

## 后续核验项目

- 动态has_det与current_has_detection、历史证据/mask是否同步；当前代码只覆盖current_evidence，需审计增强一致性。
- 递推后的历史事件与EMA/ReID是否按承诺更新；不能用单个非零梯度检查替代输入与在线路径的一致性。
- harm标记当前仍取原缓存box0/box1，是否符合新动态状态下的加权定义需记录。
- 增强默认small_area=1024、boundary_margin=10，与已核对MoTTrack的2500/20不同；augment_burst有参数但当前函数没有完整burst实现。
- G2的5步递推末端监督与G0固定单步训练存在额外状态构造差异，应明确记录。

## 建议执行顺序

1. 由本地agent在安全的checkpoint边界管理流水线，保留旧产物，先核验并修复输入一致性。
2. 优先张量化批量状态、移除逐样本GPU标量读取、预计算窗口与打包预取数据。
3. 在无额外重任务争用时，对真实batch分别测数据准备、H2D、前向、KF、backward、step，CUDA计时使用events与必要的统一同步。
4. 检查优化前后loss/梯度与更新差异，Dropout RNG和批次顺序单独记录。正确性修复与纯性能优化分开归因。
5. 实测稳态轮时和内存后重估100轮成本；保留5帧监督、全参数与有效batch，最终仍仅评一次val。


## 修复验收（packed_replay_v2）

### 已修复的训练语义

- 重建当前与窗口内历史证据后，用训练归一化统计处理前12维，has_detection保持原值。
- 连续KF均值/协方差、EMA模板、历史事件、当前/历史检测mask同步递推；后续误差能回传前部gate。
- 动态有害事件权重由当前状态和增强观测的两个端点计算；GT仅用于损失，不进入模型输入。
- 外观rank/column仅在锚点计算一次，避免按窗口位置平均时放大小量短窗口。
- G2采用受扰动的全写入历史构建单步样本，训练模型只调用锚点一次。与G1/G3的五次模型调用分开。
- MoTTrack公共增强组件完成：三数据集参数化配置、2500面积阈值、20像素边界、3–6帧burst、Markov缺失、首帧保留与可复现随机流。SportsMOT接真实检测加噪；没有换成GT观测训练。
- 恢复时检查训练目标/预算等配置一致性，恢复优化器、调度器和Python/Numpy/Torch/CUDA RNG。已完成计划的恢复可直接退出。
- 旧G1使用了错误的模型输入归一化，保留原目录作归档；修复后在新目录从头开始。
- 模型结构、参数量1,208,220、全参数优化器、batch1024、训练样本和100轮余弦预算保持既定配置。

### 性能实现

- 每个窗口以批量张量保存状态，移除逐样本GPU状态字典和重复切片拼接。
- 窗口链索引按序列预计算；小型元数据预取，增强批量生成。
- 当前序列的原始特征暂存在GPU并索引组批，避免每个窗口位置重复CPU gather/H2D。跨序列释放，模型编码结果不缓存。
- batch内统计在GPU累计，epoch末统一搬回；KF固定矩阵缓存，批量求解避免逐次主机错误检查。
- 正式命令micro_batch=effective batch=1024，无梯度累积切分。若主动缩小micro_batch，将使用按样本比例累加的微批次损失均值；有效标签比例不同时不保证与整批比值严格相同。

### 完整45序列、单轮实测

每次都采用100轮调度，仅以stop-after-epoch=1完成单轮验收；每轮321步、305,631锚点。所有loss有限，退出码0。

| 组 | 训练含义 | 实测秒/轮 | transitions | CUDA峰值/MiB | RSS峰值/GiB |
|---|---|---:|---:|---:|---:|
| G1 | 五帧连续监督 | 71.851 | 1,515,472 | 2,556 | 5.077 |
| G2 | 单步监督+观测增强 | 21.023 | 305,631 | 969 | 5.016 |
| G3 | 五帧连续监督+观测增强 | 69.676 | 1,515,472 | 2,556 | 5.076 |

旧G1约306秒/轮，修复版约快4.3倍。按本次单轮外推三组100轮训练合计约4.5小时，加三次最终val约二十分钟；这是耗时估计，尚无100轮成绩。正式90序列另算，不包含在4.5小时内。

### 已运行的核验

- shutrack/tests：217 passed、2 skipped。
- verify_window_training_review.py：KF与numpy最大绝对误差2.384e-7；单步模型证据最大差7.868e-6；单步总目标差1.073e-6。
- 批量与逐样本回放、CPU特征与GPU特征bank数值一致；未来误差对前部gate梯度非零。
- G2无噪历史预热后的锚点证据与原缓存最大差2.718e-5。
- 高斯噪声标准差约[1.00018,0.99714,1.00152,1.00110]；改变微批次分组时增强完全相同；缺失/burst/mask通过。
- verify_window_resume_pipeline.py：保存并恢复dropout RNG、AdamW和调度器后，下一步loss差0、全部参数最大差0。
- 驱动仅最终e100评估，val命令指向新目录。正式四种选优分支均验证为90条、593,762样本、624步/轮、62,400步、warmup312。
- 本次验收运行短训练和数值检查，尚未启动修复版三组100轮，也没有执行val或提交。

### 实验范围与启动方式

关联候选及竞争上下文仍采用采集时的固定回放值，标记association_context=frozen_replay_candidates。窗口内KF、EMA、证据和mask动态变化，但该实验不重新执行离散关联决策；训练中也未使用GT竞争特征。

复用现有train45与trainval90完整状态数据，无需重新采集。新输出根：
outputs/agentguard/experiments/sportsmot_mechanisms_review_v2_20261124

单独训练：
.venv/bin/python tools/run_sportsmot_g123_experiment.py --stage train

完整既定流程（训练三组→每组仅e100验证→与G0选优→90条正式训练→test/post/打包/单份提交）：
.venv/bin/python tools/run_sportsmot_g123_experiment.py --stage all

正式阶段已补入独立wrapper；当前仅核验配置/命令生成，真实推理/提交继续复用项目现有流水线。断点沿用同一命令，修复版last.pt支持恢复；原始错误协议checkpoint不会作为续训来源。

产物与可复查脚本：
- outputs/agentguard/experiments/window_training_speed_audit_20261124/review_verification.json
- outputs/agentguard/experiments/window_training_speed_audit_20261124/resume_pipeline_verification.json
- outputs/agentguard/experiments/window_training_speed_audit_20261124/full_epoch_G1（G2/G3同级）
- tools/verify_window_training_review.py
- tools/verify_window_resume_pipeline.py



## 新采集数据完整性与磁盘复核

已执行 tools/verify_window_state_data.py（CPU/只读源数据）与正式trainval验证器：

| 数据根 | 序列 | 样本 | 文件占用/GiB（st_blocks） | 新增8种状态数组/MiB |
|---|---:|---:|---:|---:|
| sportsmot_mechanisms_20261124/data | 45 | 305631 | 3.510525 | 102.848 |
| sportsmot_mechanisms_formal_20261124/data | 90 | 593762 | 6.801888 | 199.759 |

合计10.312412GiB；按inode去重后相同。train45与trainval90各自包含一份train数据，未使用文件级链接去重。此处统计数据目录，权重及推理结果另算。

- 45/45、90/90官方序列齐全，complete均通过。
- 样本→事件→frame/detection及事件→样本反向绑定均通过；样本所需KF状态缺失0。
- 全部新增状态/证据数组无非有限数值，样本绑定的KF先验协方差对角和框尺寸为正。
- 逐个复现全部有效运动端点：train 300457行、trainval 582144行；最大全写入/不写入框误差小于0.000191像素。
- 原始运动证据最大重建误差3.571e-5。
- 另有614/1039个未绑定监督样本的初始化历史事件无回放状态；window_index按event_target_row<0截断，状态递推不使用这些事件。其缓存历史证据仍可作为HCE上下文。
- verify_final_data 对trainval90：data_ok=true，593762个键检查通过，34680次检测索引抽查/20486次column抽查无错，finite_bad=0。检测索引/column为抽查，非全部匹配重放。
- 初版lean路径绕过register造成空状态数组的问题已经由按ordered_keys统一构造event_state_array修正；当前数据不是该空数组版本。
- 当前核验支持直接复用现有数据，无需为改3帧或训练归一化修复而重新采集。

报告：outputs/agentguard/experiments/window_training_speed_audit_20261124/state_data_integrity_and_footprint.json；
verify_state_rich_trainval90.json。
