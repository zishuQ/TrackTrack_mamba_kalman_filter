# Codabench 提交台账（官方 test，2026-11-21 至 11-25）

每次提交出分后在此追加；`submissions.json` 为机器可读版本。
范围：本项目 SHUTrack 近期（2026-11-21 至 11-25）jojo0 / shangqingnie 两个账号在 MOT20 / MOT17 / SportsMOT / DanceTrack 的提交。
Failed/Cancelled 的尝试也保留在册（无分数）。

## SportsMOT（phase 21402，competition 13077）

台账内最佳：**76.320** — SportsMOT harm=4 trainval90 e100 (post+AFLink)（#1068003，2026-11-25 20:20）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1062928 | 2026-11-21 18:56 | - | SportsMOT formal e100 (night final) | Finished | 75.727 | 64.648 | 88.790 | 76.085 | 96.661 | 1986 |
| 1063982 | 2026-11-22 16:01 | - | SportsMOT cost trainval formal e80 | Finished | 75.677 | 64.546 | 88.811 | 76.080 | 96.678 | 1937 |
| 1064222 | 2026-11-22 22:49 | - | SportsMOT midpoint selection e80 | Finished | 75.774 | 64.720 | 88.802 | 76.130 | 96.667 | 1975 |
| 1064937 | 2026-11-23 17:34 | - | SportsMOT ablation e80 (train45, D) | Finished | 75.840 | 64.798 | 88.850 | 76.105 | 96.677 | 1975 |
| 1066994 | 2026-11-25 03:12 | jojo0 | SportsMOT harm=2 E_e100 (post+AFLink) | Finished | 75.979 | 65.095 | 88.766 | 76.755 | 96.659 | 1885 |
| 1067021 | 2026-11-25 04:10 | shangqingnie | SportsMOT harm=4 train45 e100 (post+AFLink) | Finished | 75.988 | 65.324 | 88.475 | 76.959 | 96.642 | 1870 |
| 1068003 | 2026-11-25 20:20 | jojo0 | SportsMOT harm=4 trainval90 e100 (post+AFLink) | Finished | 76.320 | 65.990 | 88.346 | 77.467 | 96.609 | 1848 |

## MOT20（phase 16383，competition 10050）

台账内最佳：**66.637** — MOT20 formal_e100 (sqrt-size)（#1063086，2026-11-21 21:39）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1063086 | 2026-11-21 21:39 | - | MOT20 formal_e100 (sqrt-size) | Finished | 66.637 | 69.283 | 64.270 | 82.661 | 77.752 | 635 |
| 1064223 | 2026-11-22 22:53 | - | MOT20 midpoint_e20 | Finished | 66.512 | 68.848 | 64.430 | 82.220 | 77.879 | 616 |
| 1064320 | 2026-11-23 00:48 | - | MOT20 e80 round | Finished | 66.635 | 69.200 | 64.342 | 82.518 | 77.812 | 619 |
| 1064623 | 2026-11-23 11:59 | - | MOT20 low-LR continuation | Finished | 66.587 | 69.170 | 64.276 | 82.534 | 77.748 | 633 |
| 1066968 | 2026-11-25 02:21 | - | MOT20 sample-proportional e100 | Finished | 66.531 | 68.913 | 64.410 | 82.325 | 77.834 | 632 |

## MOT17（phase 16382，competition 10049）

台账内最佳：**67.827** — MOT17 formal_e50 (init MOT20 e100)（#1063098，2026-11-21 21:54）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1063098 | 2026-11-21 21:54 | - | MOT17 formal_e50 (init MOT20 e100) | Finished | 67.827 | 69.442 | 66.551 | 83.931 | 81.888 | 837 |
| 1064234 | 2026-11-22 23:11 | - | MOT17 midpoint_e20 | Finished | 67.672 | 68.983 | 66.696 | 83.709 | 82.147 | 813 |
| 1064329 | 2026-11-23 01:05 | - | MOT17 e80 (from MOT20 e80, 50 epochs) | Finished | 67.763 | 69.298 | 66.566 | 84.026 | 81.980 | 819 |
| 1064624 | 2026-11-23 12:00 | - | MOT17 low-LR continuation | Finished | 67.689 | 69.206 | 66.500 | 83.851 | 81.805 | 846 |
| 1066979 | 2026-11-25 02:45 | - | MOT17 sample-proportional e50 (from round-1 MOT20) | Finished | 67.814 | 69.434 | 66.527 | 84.282 | 81.998 | 795 |

## DanceTrack（phase 24635，competition None）

台账内最佳：**65.700** — DanceTrack trainval formal e80（#1064153，2026-11-22 20:36）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1064153 | 2026-11-22 20:36 | - | DanceTrack trainval formal e80 | Finished | 65.700 | 51.500 | 83.900 | 66.800 | 93.500 | - |

