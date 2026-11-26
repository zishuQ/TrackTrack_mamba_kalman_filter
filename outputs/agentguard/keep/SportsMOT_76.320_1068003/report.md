# SportsMOT harm=4 · 官方 train45+val45（90 条）100 轮

- 权重：`/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/sportsmot_harm4_trainval90_e100_20261125/groups/appearance_row_column/checkpoints/epoch_100.pt`（epoch 100 / 62,400 步，sha256 407e975e554dfcf1…）
- 预算：624 步/轮 × 100 轮 = 62,400 步；warmup 312（0.5%）；lr 1e-4 余弦至 0；训练 5171.6 s
- 配方：harm=4、dropout 0.05、seed 42、batch 1024、shards 1、原运动 gate、从头训练
- test：150 序列，seed 10000，NSA KF，线性 post + AFLink（thrT=(0,20)/thrS=100/thrP=0.05）
- 包：`/home/shang/workspace/TrackTrack/outputs/agentguard/experiments/sportsmot_harm4_trainval90_e100_20261125/submissions/SportsMOT.zip`（150 文件 / 1,021,611 行 / 13077620 B / sha256 f005250150fce480…，本地校验 ok）

提交留给用户执行。


## 提交结果（1068003，jojo0，Finished）

- HOTA **76.320** / AssA 65.990 / DetA 88.346 / IDF1 77.467 / MOTA 96.609 / IDSW 1848 / Frag 2546
- 对比 train45 harm=4（1067021，HOTA 75.988）：HOTA +0.332pp、AssA +0.666、DetA −0.129、IDF1 +0.508、IDSW −22。
- 对比旧官方最佳 75.840：+0.480pp。
