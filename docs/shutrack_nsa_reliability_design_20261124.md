# SHUTrack｜NSA 检测可信度方案备忘

2026-11-24（北京时间）。状态为方案讨论，等待用户决定实现口径。当天仅核对代码、既有缓存和报告，并记录项目记忆。代码修改和实验由用户另行安排。

## 当前候选

在现有匹配完成后，运动分支为每个匹配输出可信度倍率 $c\in[0.2,5]$，中性值为 $1$。NSA 继续使用原始检测置信度 $s$ 构造 $R_{\mathrm{NSA}}$，然后用 $R_{\mathrm{eff}}=R_{\mathrm{NSA}}/c$ 更新 KF。$c<1$ 减弱当前测量的影响，$c>1$ 增强测量的影响，$c=1$ 恢复原 NSA 完整更新。外观继续使用原 `[0,1]` gate 和基于原始检测分数的 EMA。`[0.2,5]` 覆盖早先的 `[0.01,5]` 候选。

2026-11-25 用户确认历史框使用原检测框，并确认其预期是 $c=1$ 恢复原行为。首版设计据此保留原 NSA 噪声构造规则，$c=1$ 对齐旧运动 gate 为 1 的完整运动更新。在相同先验、检测和外观 gate 下，KF 后验及历史写回均与这一基线对齐。$c$ 是相对 NSA 的检测可信度倍率，例如 $c=0.2,1,5$ 分别令测量噪声为原来的 $5,1,0.2$ 倍。此前提到的噪声下限只是可选设计讨论，首版采用原噪声规则。训练初始化建议输出 $c=1$，之后模型按匹配情况学习调节。当前记录方案决定，代码实施仍待另行安排。

## 代码事实与建议

### 测量噪声的接入位置

- 在线 `KalmanFilter.project` 先根据更新前预测框宽高 $w^-,h^-$ 和检测分数 $s$ 构造 $R_{\mathrm{NSA}}=(1-s)\,\operatorname{diag}[(w^-/20)^2,(h^-/20)^2,(w^-/20)^2,(h^-/20)^2]$，再将它加到 $HP^-H^\top$ 上。`update` 随后用 Cholesky 分解求增益。离线 NumPy 和可微 Torch 有同一套更新公式。 [在线 NSA](../3.%20Tracker/trackers/kalman_filter.py#L65) · [离线 NSA](../agentguard/src/agentguard/motion/nsa_numpy.py#L96) · [Torch KF](../shutrack/training/recurrent_motion.py#L48)
- 建议在构造原 NSA 噪声之后、形成创新协方差之前应用 $1/c$。使用 $S=HP^-H^\top+R_{\mathrm{eff}}$ 和 $K=P^-H^\top S^{-1}$ 重算后验均值和协方差，由线性求解得到增益。运动预测 $Q$、原始检测置信度、检测阈值、置信度关联代价、运动输入里的检测分和外观 EMA 沿用原规则。检测分给出检测器自身的置信度，模型估计每个匹配相对 NSA 基线需要调整多少测量噪声。模型拿到的已是匹配结果，因此这次更新只影响写回和后续关联。

### KF 状态和历史框分开处理

- 旧 `Track.update_with_gates` 的运动 gate 有两条作用路径。KF 路径将 NSA 更新前后的八维均值和协方差插值。框路径将预测框与检测框插值，所得 `effective_box` 用于历史、角点速度 `self.velocity` 和 `self.box`。旧 gate 为 1 时历史框是检测框，历史状态则是已更新的 KF 状态。 [旧门控写回](../3.%20Tracker/trackers/track.py#L223)
- `Track` 有 KF 均值时，导出用的 `x1y1wh` 和 IoU 关联用的 `x1y1x2y2` 都从均值取得。角度关联另外读取历史框及 `self.velocity`。因此历史框即使沿用检测框，$c$ 仍会影响输出框和后续 IoU 关联。 [框属性](../3.%20Tracker/trackers/track.py#L310) · [输出](../3.%20Tracker/run.py#L377) · [IoU](../3.%20Tracker/trackers/utils.py#L114) · [角度关联](../3.%20Tracker/trackers/utils.py#L213)
- 首版由 $c$ 直接更新 KF 均值和协方差。按用户决定，框字段、角点速度和 `self.box` 使用原检测框。$c=1$ 时，同样的先验、检测和外观 gate 会得到旧运动 gate 为 1 时的更新结果。小 $c$ 时，原检测框仍进入角度历史。TrackTrack 角点历史与 SHUTrack 的 HCE 证据缓冲是两个对象，后者仍以各帧事件证据记录。 [证据缓冲写入](../shutrack/integration/tracktrack.py#L277)

### 缓存和一步监督

- 当前 `motion_box_diou_loss` 用 sigmoid gate 在 `pred_box_g0/g1` 间插值，再和同身份的 $t+1$ GT 算 $1-\mathrm{DIoU}$。缩放 $R$ 时，$K(c)$ 随 $c$ 非线性变化，当前两端点插值无法重现新更新。可保留同一 GT 和 DIoU 目标，接入可微 KF 求解，顺序为用 $t$ 帧更新前状态和检测重算后验、应用 $t+1$ GMC、再做一次 KF predict 并转成 xyxy。固定先验的一步监督只直接观察下一帧框。协方差对后续增益、证据和关联的影响需要更长状态递推或在线跟踪才能观察。 [旧一步损失](../shutrack/training/loss.py#L231) · [GMC 与预测](../shutrack/training/recurrent_motion.py#L89)
- train45 和 trainval90 的现有状态缓存都有 `event_state_mean`（更新前 8 维）、`event_state_cov`（64 个数，恢复成 8×8）、`event_det_measurement`（像素 cxcywh）、`event_det_score`、`event_warp_next`（下一帧 2×3）、`event_frame`、`event_gap`、`event_has_detection`、`event_detection_index`、`event_target_row`。样本表有 `current_event_index`、`gt_box_t1` 和 `motion_gt_valid`。按样本索引取事件，并核对反向行号、帧号、检测索引、有效检测和 GT 掩码。只含旧预测端点的缓存缺少重算新增益所需的信息。现有一步训练入口还需装载这些状态字段。以后实现时可用固定 $c=1$ 的重算框与缓存全写入端点 $g=1$ 对齐。 [采集](../tools/sportsmot_supervision_pilot.py#L588) · [缓存核验](../tools/verify_window_state_data.py#L85) · [一步训练入口](../tools/train_supervision_groups.py#L1176)
- 既有审计已逐行核验 train45 的 300,457 条及 trainval90 的 582,144 条有效运动样本，旧端点最大重建误差小于 0.000191 像素。两份缓存分别有 614 和 1,039 条未绑定监督行的初始化历史事件，状态为占位，可保留原证据上下文。今天只读核对了实际数组头和置信度。train45 全部 305,631 条、trainval90 全部 593,762 条监督绑定样本中，均无分数达到 0.99、落出 `[0,1]` 或为非有限值。此计数覆盖缓存样本，线上其他分布需另行核对。 [既有完整性审计](../outputs/agentguard/experiments/window_training_speed_audit_20261124/state_data_integrity_and_footprint.json) · [train45](../outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data) · [trainval90](../outputs/agentguard/experiments/sportsmot_mechanisms_formal_20261124/data)

### 范围、初始化和边界

- `[0.2,5]` 在对数尺度上对称，对应测量噪声倍率 `[0.2,5]`。候选映射 $c=\exp(\log(5)\tanh z)$ 在 $z=0$ 时恰为 1，有非零梯度，有限 logit 时逐渐接近区间两端。现运动头 logit 是双线性项和线性项之和。精确中性初始化须同时让两项初始贡献为零，或经新的零初始化输出层。若复用旧权重，旧 motion logit 保留旧 gate 语义，运动输出重新初始化。外观 sigmoid 沿用。 [双线性输出头](../shutrack/models/query_controller.py#L77)
- 噪声缩放五倍的实际影响取决于 $HP^-H^\top$ 相对 $R_{\mathrm{NSA}}$ 的大小。`c=0.2` 仍会吸收观测，`c=5` 也仍是 KF 更新。若 $s=1$，原 NSA 噪声为零，该事件的更新对 $c$ 无响应。$s$ 接近 1 时作用也可能很小。若给 $R$ 设下限或限制分数上界，受影响事件在 $c=1$ 时也会改变。当前 SportsMOT 缓存的分数核对支持首版维持原 NSA 噪声规则。另需定义非有限、越界分数，退化框，非正定或病态创新协方差，以及求解失败的处理。Torch 路径当前用 `solve_ex` 并忽略返回的状态码，新机制应明确如何检查并对齐训练和推理的异常行为。 [Torch 求解](../shutrack/training/recurrent_motion.py#L54)
- 现有模型返回两个 sigmoid gate，runtime 和 Track 均把值裁剪到 `[0,1]`。需要为运动可信度定义独立输出及范围，明确 checkpoint 语义、运行统计和固定 $c=1$ 的对应开关。保持现有未匹配路径只预测，初始化、未启用门控的匹配走原更新。当前 New 轨迹可选的 early-enable 路径从第 3 次匹配起用模型。首版建议只为 NSA 模式启用，新输出在非 NSA 配置中如何处理需明确。 [runtime 裁剪](../shutrack/runtime/adapter.py#L455) · [早期匹配](../shutrack/integration/tracktrack.py#L195) · [应用入口](../shutrack/integration/tracktrack.py#L263)

## 下次讨论需用户拍板

1. **首版监督和预算**。建议先以一步 DIoU、无增强、`harm_weight=1` 为干净对照。旧 harm 标签定义是 `g=1` 相对 `g=0` 更差，新 $c$ 的有害事件需围绕中性 $c=1$ 重新定义。用户经验是更大的 harm 权重需要更长训练才可能形成分离度，训练预算、加权与是否添加 $\log c$ 中性正则仍待用户决定。
2. **比较口径与权重来源**。决定从头训练或复用主干和外观头并重置运动输出。日后可用固定 $c=0.2,1,5$ 探查调整空间、观察学到的 $c$ 是否整体贴近一端，并用同一 checkpoint 的学习 $c$ 与固定 $c=1$ 隔离运动贡献。现有 SportsMOT 一步目标偏好旧完整写入，开放 $c>1$ 是否带来收益仍待验证。
3. **异常输入处理**。实施前明确异常分数、退化框、协方差求解失败和非 NSA 配置的处理。
