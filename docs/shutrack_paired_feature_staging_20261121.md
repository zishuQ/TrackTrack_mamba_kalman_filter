# 配对训练数据供给提速与断点续训验收

日期：2026-11-21（Asia/Shanghai）

## 已完成结果

MOT20 正式训练保持 query_film_bilinear 新架构、float32 原始 ReID、
D 方案损失、batch=1024、sqrt-size 采样及原学习率计划。
从 epoch 25 / step 6775 完整断点续训，已核验到 epoch 29 / step 7859。
原断点另存为 before_io_fix_epoch_025.pt。

| 指标 | 修复前最后 10 轮 | 修复后 epoch 26–29 |
|---|---:|---:|
| 平均每轮耗时 | 203.48 s | 28.76 s |
| 每轮耗时范围 | 154.43–240.71 s | 26.48–30.54 s |
| 每轮 batch | 271 | 271 |
| 每轮样本 | 约 275,700 | 约 275,700 |
| 平均每步 | 0.751 s | 0.106 s |

同一正式配置提速约 7.08 倍。IWG 对照 28.5 秒/257 步，约 0.111 秒/步。
修复后进程已记录峰值 VmHWM=5,835,144 KiB，约 5.56 GiB。

## 原因与处理

旧的小字段预载、prefetch、mmap 和 compact ReID 均仍生效。
大特征池约 29.85 GiB；逐 batch 随机读取历史 track、检测特征和 rank
造成重复搬运，以及大文件缓存的驻留/回收开销。采样期间曾观察到 CPU
主要处于系统态、磁盘读入接近零，故不能把全部慢速归因于物理磁盘吞吐。

新增 shutrack/training/feature_staging.py：
1. 根据原采样计划收集本序列需要的唯一 event、检测和 rank 行；
2. 按源行号排序，以有界连续文件块预载，保留原 dtype；
3. batch 仍按原采样索引和顺序，从连续内存取值；
4. 本序列处理完后释放临时缓存，恢复原 mmap 句柄；
5. 使用 MADV_DONTNEED 和 POSIX_FADV_DONTNEED 释放已用完的干净文件页，
   文件内容保持原样，当前训练使用独立内存副本。

当前配对训练统一在每条序列开始时暂存采样计划需要的特征行。MOT20 的
4 分片 sqrt-size 配方每个分片内每条序列暂存一次。旧 checkpoint 可以通过
`--resume` 接着训练，恢复原优化器、学习率调度和随机数状态。

训练配置文件 SHA-256 与修复前相同：
d3aa9a86cf2bbf1467561ed217b7b612cf2e32f09159355194019d641bc6e9d8

## 验证

- test_feature_staging.py 与 test_paired_pipeline.py：21 passed。
- 覆盖 float32/float16 源、重复采样、历史 padding、column 额外检测、
  rank 稀疏有效行、NPZ 成员偏移及 mmap 切片。
- 正式四序列原采样计划的 batch 数值与 dtype 逐项相同。
- 使用新架构与三项 D 损失进行相同的短训练，loss、模型参数、
  AdamW 状态最大差异均为 0。
- 正式续训连续完成四种分片，step 从 6775 连续增加到 7859。

复现验证：
.venv/bin/python tools/verify_feature_staging.py --checkpoint <checkpoint> --out <report.json>

原始验收数据位于：
outputs/agentguard/experiments/shutrack_architecture_night_20261120/
- feature_staging_verification_v3.json
- feature_staging_production.json
- io_fix_restart.json
- logs/mainline_io_fix_resume.log

## soft gate 标签保存方案评估

该方案可行。每序列以原 samples 行号保存 float32 的 motion_gate_target.npy，
训练开始一次性预载，batch 按原索引取标签。30 万条约 1.2 MB，58 万条约 2.3 MB。
候选 gate 的扫描成本放在标签生成阶段，避免在 batch 中重复扫描。

本轮完成的是原 D 正式训练的数据供给修复。
soft gate 标签生成和对应运动损失适用于后续独立监督实验，另行比较效果。
