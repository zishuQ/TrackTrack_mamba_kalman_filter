# SHUTrack A SportsMOT 监督实验交接文档（2026-11-19）

写给接手的新对话/同事：这份文档汇总了已经完成的工作、关键结论、现状与下一步建议。
所有数字都是 45 条 SportsMOT val raw 的**开发对照**——这 45 条 val 参与过 trainval
训练，不是独立 test。

## 0. 环境与代码状态

- 仓库：`/home/shang/workspace/TrackTrack`，分支 `feature/shutrack-v2`，HEAD `efed8ad`。
- Python：`/home/shang/workspace/TrackTrack/.venv/bin/python`。
- 起点权重：`outputs/agentguard/experiments/shutrack_a_submit_20261117/SportsMOT/checkpoints/last.pt`
  （e200，step 28,400）。
- 工作区还有约 34 个未提交改动（本轮及并发实验的修复、工具、测试），**不要盲目提交或回退**；
  旧权重、旧数据、检测缓存都保留原样。
- 测试：`pytest shutrack/tests` = **125 passed, 1 skipped**。
- 未启动正式 200 轮从头训练，未做 Codabench/test 提交。

SHUTrack A = HCE + RSC，输出运动、外观两路 gate（两路都保留）。在线写回由
`Track.update_with_gates` 完成：运动 gate 插值 KF 状态，外观 gate 插值 EMA 模板。

## 1. 成对监督数据管线（tools/）

核心思路：用 e200 对 90 条 SportsMOT trainval 序列做**同一次回放**，为每个成熟匹配事件
保存线上输入、标签、mask、以及可验证的检测缓存索引（不按 track_id 拼接）。

- `tools/generate_trainval_paired.py`：回放 + 标签生成，写到
  `outputs/agentguard/experiments/<实验>/data/`。每序列含 `samples.npz`、事件证据/轨迹/
  检测索引数组、`summary.json`（含 replay protocol）。
- 标签版本：`LABEL_SEMANTICS_VERSION = 3`。schema 3 包含三处语义修复：
  外观正负例搜索走完整 5 帧窗口、补检池复用线上 `find_deleted_detections`（+1 像素 IoU）、
  身份锚点跨 GT 断档时清空旧证据。
- schema 闸门：旧 schema/无 protocol 数据会被生成器与正式 `--from-scratch` 训练拒绝；
  新旧混合在加载样本前报 `Incompatible replay protocols`。真实分片已验证：90 条
  schema 3 分片生成续跑 skipped=90、真实 full-write schema 3 分片可从零训练 1 轮并
  resume、真实旧+新混合被拒绝。
- `tools/train_supervision_groups.py`：统一的组训练器。所有组：
  从 e200 初始化、90 条序列、batch 1024、seed 42、lr 1e-4、wd 1e-4、warmup 185、
  4 分片 ×10 循环、40 轮 ×185 batch = **7,400 步**；保存 epoch_020/040/last。
- `shutrack/training/loss.py` 中的损失：
  - `motion_box_diou_loss`：g=0/1 下一帧 GT 框插值 DIoU；
  - `appearance_soft_loss`：外观区分度软标签 BCE+0.1MSE（联合组）；
  - `appearance_rank_loss`：用模型外观 gate 做可微 EMA，身份排序 `softplus((d_pos−d_neg)/τ)`，
    无有效正负例时回退原外观软标签；
  - `motion_speed_l1_loss` / `motion_speed4_l1_loss`：KF 中心速度 / 四维速度 L1；
  - `appearance_column_loss`：下一帧真实匹配池中的外观争抢损失（见 §4）。

## 2. 已完成的监督组与结果

45 条 val raw，HOTA（括号内为 checkpoint）：

| 组 | HOTA | AssA | IDF1 | DetA | 数据/说明 |
|---|---|---|---|---|---|
| e200 | 83.014 | 73.861 | 82.980 | 93.329 | 起点 |
| control | 83.133 | 74.179 | 83.141 | 93.195 | 原配方软标签 |
| motion_diou | 83.414 | 74.507 | 83.325 | 93.411 | 运动 GT 框 DIoU |
| appearance_score | 83.485 | 74.714 | 83.511 | 93.309 | 外观区分度软标签 |
| joint（motion_diou_appearance_score） | 83.606 | 74.800 | 83.562 | 93.472 | 旧数据联合组 |
| **appearance_rank（B 旧数据）** | **83.667** | **74.887** | **83.596** | **93.501** | 运动 DIoU + 外观身份排序；目前最高 |
| motion_diou_speed | 83.573 | 74.781 | 83.527 | 93.423 | 中心速度 L1(50) |
| motion_diou_speed4 | 83.635 | 74.843 | 83.575 | 93.484 | 中心(50)+宽高(14) |

schema 3 数据上的 B/C/D 重跑（`shutrack_a_supervision_bcd_20261119`）：

| 组 | HOTA | AssA | IDF1 | DetA | 相对 B |
|---|---|---|---|---|---|
| B：rank（schema 3 重训） | 83.640 | 74.839 | 83.557 | 93.499 | — |
| C：rank + 四维速度 | 83.623 | 74.805 | 83.530 | 93.505 | HOTA −0.016 pt |
| D：rank + 外观争抢 | 83.653 | 74.854 | 83.554 | 93.511 | HOTA +0.014 pt |

关键结论：

- 运动 DIoU 修复了 c003 GT3/GT4（109 帧起）与 kUtTtmLaJA_c008 GT6/7 等长期互换；
  外观区分度修复了 c011 GT0/GT7（341/280 帧）但在 c003 上反而丢失了运动组的修复；
  联合组保留了两者的大部分收益（c003 除外）。
- 直接外观排序（appearance_rank/B）总体最好；c003、c007、c011 的长期互换仍存在。
- 四维速度叠加在排序监督上没有额外收益（C vs B 略负），不建议进入下一轮正式配方。
- 外观争抢监督（D）有方向性信号（单序列 kUtTtmLaJA_c004 +1.32 pt、外观 gate 均值
  0.6365→0.6450），但总体 +0.014 HOTA 属噪声级，且 dw7LOz17Omg_c067 −0.92 pt、
  长期互换无改善，证据不足以加入正式配方。

## 3. 已完成的关键修复（第二轮验收通过）

- 外观正负例提前退出：走完整 5 帧窗口后 c003/c007/train-c001 分别 +69/+423/+513
  个有效外观对，原有有效样本选择全部保留。
- GT 绑定：真实最大 IoU、仅 `IoU>=0.5` 绑定；补检选择直接复用线上
  `find_deleted_detections`（含 +1 像素），保留原缓存索引；三条真实序列逐帧补检集合无差异。
- 身份补标跨 GT 断档：GT 中断后清除依赖该身份的 `established/streak` 证据，恢复后
  需重新积累；两个反例从误补 2 条变为 0 条，c003 原有 64 条全部保留。
- CLEAR 辅助脚本空预测帧：与仓库 TrackEval 的前帧匹配优先一致；三帧反例从 2 次
  IDSW 变为 0，45 条已有结果统计不变。
- verify_speed4：整批零样本失败、错误归因改为逐序列；历史 90 条 593,673 样本通过。
- 外观模板 A/B 实验（逐帧反事实）：oracle 模板 7 次纠正/0 次改错，线上可选模板
  7/1；c007 长期互换在 583–584 帧可被 B 纠正但未持续恢复，c003 108 帧线上旧模板造成
  1 次改错。结论：B 只做逐帧对照，**持续身份恢复未测量**，不推荐作为训练选项。

报告：`outputs/agentguard/experiments/shutrack_a_review_fixes_20261119/REPORT.md` 与
`REPORT_ROUND2.md`；证据在 `.../evidence/`。

## 4. D 的损失定义（便于接手）

`appearance_row_column` = B + 下一帧外观争抢损失：

- 同一次回放的 ColumnPilot 在实际下一帧关联调用中，为每个被争抢事件记录：
  `column_det`（本身份唯一检测的缓存索引）、`column_offset`（非外观代价：
  raw cost − 0.5·cos + 来源惩罚）、`column_rival`（竞争轨迹到同一检测的最优最终代价）。
- loss = `softplus((clamp(column_offset + 0.5·(1−cos(T_g, positive)), 0, 1) − column_rival)/τ)`,
  `T_g` 是用模型外观 gate 对写前模板做线上 EMA 的结果。
- 与 `appearance_rank` 的差别：排序用未来 5 帧的同 GT 正例与同帧竞争检测做纯余弦排序；
  column 用下一帧真实匹配池、竞争轨迹完整代价，只在被争抢事件生效、无回退项。
- 数据覆盖：115,205 个有效事件，覆盖 90 条序列；真实 batch 中约 38% 行有效；
  默认权重 1.0 时外观 gate 梯度约 9.6e-8（rank 的 0.5%），本轮按 25% 对齐取 50。

## 5. 产物与路径

- 旧四组：`outputs/agentguard/experiments/shutrack_a_supervision_trainval_20261118/groups/`
- 排序/中心速度：`outputs/agentguard/experiments/shutrack_a_supervision_rank_speed_20261119/groups/`
- 四维速度：`outputs/agentguard/experiments/shutrack_a_supervision_speed4_20261119/groups/`
- 最新 B/C/D（schema 3）：`outputs/agentguard/experiments/shutrack_a_supervision_bcd_20261119/`
  - 权重：`groups/{appearance_rank,appearance_rank_speed4,appearance_row_column}/checkpoints/epoch_040.pt`
  - 报告：`REPORT.md`、`summary.json`、`bcd_validation.json`、`evals/`
  - 数据：`data/`（90 条 schema 3 + column probe；6.7G）
- 外观模板实验：`outputs/agentguard/experiments/shutrack_a_appearance_template_20261119/`
- 复现说明：`docs/shutrack_a_supervision_repro_20261119.md`、
  `docs/shutrack_a_speed4_repro_20261119.md`
- 主要工具：`tools/generate_trainval_paired.py`、`train_supervision_groups.py`、
  `verify_bcd.py`、`verify_speed4.py`、`bcd_analysis.py`、`run_train_bcd.sh`、
  `run_eval_bcd.sh`、`clear_id_analysis.py`、`paired_gap_diagnostic.py`、
  `appearance_template_experiment.py`，以及审计工具 `appearance_search_audit.py`、
  `recovered_pool_audit.py`、`identity_anchor_audit.py`。

## 6. 下一步建议（供新对话执行）

1. **确定正式训练配方**：建议以 B（运动 GT 框 DIoU + 外观模板排序 + 旧外观回退）为
   基础。四维速度（C）无收益；外观争抢（D）证据不足，均不应直接加入正式配方。
2. **准备正式数据**：若正式训练走 from-scratch 配方，需要按 schema 3 生成 full-write
   数据（`--full-write` + 所需的 `--column-probe`/`--rank-arrays` 组合），并先用
   `tools/verify_bcd.py`/`verify_speed4.py` 类似校验确认 90 条、593,673 样本、事件键、
   mask、端点一致；生成器/训练器会拒绝旧 schema 与新旧混合。
3. **正式从零训练与评估**：按既定预算（如 200 轮）训练后，跑相同 45 条 val raw；
   再决定是否做 Codabench test 提交（本轮未授权）。
4. **未解决/待研究**：
   - c003 GT3/GT4、c007 GT2/7、c011 的长期互换在 B/C/D 中均未修复；外观模板 A/B
     仅逐帧反代，未做闭环回放或整段恢复测量。
   - 排序权重 18、四维速度权重 50/14、column 权重 50 都是基于真实 batch 梯度定的
     起始值，未做权重扫描；正式训练前可复核。
   - 45 条 val 参与训练，所有排名都是开发对照；最终选型需要在独立 test 上确认。

## 7. 给新对话的提示

- 先读本文件、`REPORT.md`（B/C/D）与 `REPORT_ROUND2.md`（修复验收）。
- 不要修改/覆盖已有实验目录、旧 checkpoint、检测缓存；新实验一律新建目录。
- 任何训练前先跑 `pytest shutrack/tests -q` 和对应 verify 脚本；正式从头训练只接受
  schema 3 且配置一致的完整分片。
- 未经明确授权不要做 Codabench/test 提交。
