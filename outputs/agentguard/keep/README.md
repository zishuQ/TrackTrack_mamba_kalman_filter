# 保留的最高 test 权重与数据（2026-11-27）

规则：每个数据集只保留官方 test HOTA 最高的一个权重及其最小数据（权重、训练配置与曲线、提交记录、提交包、报告）。

## MOT20

- HOTA **66.637**，提交 1063086（Finished）
- 提交包：`MOT20_66.637_1063086/submission/MOT20.zip`
- 权重：`MOT20_66.637_1063086/weights/epoch_100.pt`（epoch 100，sha256 4726b1703f77cba2…）
- 配方：harm=1.0、dropout=None、27100 步、warmup 271、shards 4

## MOT17

- HOTA **68.229**，提交 1070802（Finished，2026-11-27）
- 提交包：`MOT17_68.229_1070802/submission/MOT17.zip`
- 权重：`MOT17_68.229_1070802/weights/epoch_050.pt`（epoch 50，sha256 f2ef5142df673542…）
- 配方：harm=1.5、dropout=0.0、LR 2e-5、5200 步、warmup 104、shards 1，底座 MOT20 harm=2 e100（#1068664）
- 同日次佳：init MOT20 harm=1 e100（#1063086），HOTA 68.167（#1070817）

## SportsMOT

- HOTA **76.32**，提交 1068003（Finished）
- 权重：`SportsMOT_76.320_1068003/weights/epoch_100.pt`（epoch 100，sha256 407e975e554dfcf1…）
- 配方：harm=4.0、dropout=0.05、62400 步、warmup 312、shards 1

