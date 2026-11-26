# 第一轮：harm=2、dropout=0

| 数据集 | 初始化 | 提交 | HOTA | 基线 | Δ(pp) | 其他指标 |
|---|---|---|---:|---:|---:|---|
| MOT20 | scratch | 1068664 | 65.213 | 66.637 | -1.424 | HOTA 65.213 / AssA 67.418 / DetA 63.258 / IDF1 81.133 / MOTA 76.566 / IDSW 775 / Frag 1008 |
| MOT17 | mot20 e100 (round 1) | 1068665 | 68.023 | 67.827 | +0.196 | HOTA 68.023 / AssA 70.250 / DetA 66.172 / IDF1 84.945 / MOTA 81.794 / IDSW 810 / Frag 1458 |

判定：平均 ΔHOTA = -0.614pp → **harm=1 (original recipe) + dropout 0.05**

