# SportsMOT：逐事件 GT gate 标签实验

基线快照：652f27b。模型固定为 query_film_bilinear，appearance_proj_dim=256。

## 变更范围

新增独立标签模块、生成器、训练入口、验证器和串行实验驱动。
当前 MOT20/MOT17 使用的 train_supervision_groups.py、原 loss.py 和推理入口
保持此次实验开始前的内容；独立训练进程内适配既有循环。

## 标签

从已有 samples.npz 的 pred_box_g0、pred_box_g1、gt_box_t1 和 motion_gt_valid
计算 21 个候选 gate（0–1，间隔 0.05）。每事件取 DIoU 损失最小的候选；
与最小损失相差不超过 1e-6 的候选中选择最大的 gate。

每序列独立保存 float32 motion_gate_target.npy；训练启动一次预载。
metadata.json 绑定原始行身份、端点、GT 和有效掩码的摘要，以及标签本身的摘要。
检测、ReID、历史事件和原采集文件继续复用。

## 损失与对照

运动分支采用 BCEWithLogitsLoss 对连续 gate 标签拟合，仍使用 motion_gt_valid。
外观 rank/column、KF/EMA 更新、初始化、采样、随机种子和学习率计划沿用新架构基线。
第一轮采用单位事件权重。

训练：官方 train 45 条；验证：官方 val 45 条。
预算：200 个分片轮次、4 分片、seed=42、batch=1024、lr=1e-4。
先完成 8 个分片轮次（2 个完整周期）测门控与峰值，再从相同断点继续到 200；
总学习率计划从一开始就按 200 轮建立，续训保持该计划。

## 已完成检查

- 9 项 CPU 测试通过：端点/部分写入目标、持平规则、缺失 GT 掩码、
  DIoU 公式一致性、饱和时 BCE 梯度、行绑定与校验、独立入口恢复。
- 真实篮球 c003 的 3,608 条标签生成与 128 行核验通过。
- 真实新架构检查中，目标 [0.5,0,1,1] 对应原模型输出均接近 1，
  新 BCE 的运动头梯度范数 4.151（有限且非零）；检查在 CPU 上完成。

## 资源评估与执行

当前硬件：RTX 4070 Laptop，显存总计约 8 GiB，主机内存约 30.6 GiB。
先以保守预算等待现有正式重任务结束，再做 128/256/1024 三档实际训练步测量。
batch=1024 的新 BCE + rank + column 前向、反向及 AdamW 更新实测：
CUDA 峰值分配 465.05 MiB，峰值保留 608 MiB。
据此启动预算设为空闲显存至少 2,048 MiB、MemAvailable 至少 18,000 MiB。
连续两次检查通过后才启动下一阶段。

8 轮实测后，用实际 CUDA 峰值加 1,024 MiB、主机峰值加 4 GiB 更新启动预算。
资源不足时等待，现有任务保持运行。生成、训练、离线验证、跟踪评测串行执行。

入口：
.venv/bin/python -u tools/run_motion_gate_experiment.py

实验目录：
outputs/agentguard/experiments/shutrack_motion_gate_targets_20261121/SportsMOT/

state.json：执行阶段与资源判断。
resource_checks.jsonl：启动前评估记录。
groups/appearance_row_column/motion_target_metrics.jsonl：分组 gate、框误差和峰值。
comparison.json / comparison.md：最终新架构同预算 HOTA/AssA 对照。
