# Codabench 提交台账（官方 test，全历史）

每次提交出分后在此追加；`submissions.json` 为机器可读版本。
范围：jojo0 / shangqingnie 两个账号在 MOT20 / MOT17 / SportsMOT / DanceTrack 四个测试 phase 的全部提交。
Failed/Cancelled 的尝试也保留在册（无分数）。

## SportsMOT（phase 21402，competition 13077）

历史最佳（全架构）：**76.533** — pre-ledger: sportsmot_test_0.80_mamba_exp10_1_7.zip（#616517，2026-03-12 19:05）
SHUTrack 时期最佳：**76.320** — SportsMOT harm=4 trainval90 e100 (post+AFLink)（#1068003）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 518264 | 2026-02-02 16:47 | jojo0 | pre-ledger: sportsmot_test_0.80.zip | Finished | 75.521 | 64.307 | 88.777 | 75.908 | 96.540 | 2091 |
| 522902 | 2026-02-04 19:59 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp1_2.zip | Failed | - | - | - | - | - | - |
| 522919 | 2026-02-04 20:08 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp1_2.zip | Finished | 75.393 | 64.373 | 88.383 | 76.213 | 96.522 | 2060 |
| 530587 | 2026-02-09 00:57 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_2_58.zip | Finished | 75.462 | 64.291 | 88.657 | 76.020 | 96.509 | 2114 |
| 531268 | 2026-02-09 13:33 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_3_ep.zip | Finished | - | - | - | - | - | - |
| 533810 | 2026-02-10 11:24 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_3_ep.zip | Failed | - | - | - | - | - | - |
| 533855 | 2026-02-10 11:51 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_3_ep.zip | Finished | 75.274 | 64.094 | 88.485 | 75.989 | 96.523 | 2070 |
| 537216 | 2026-02-12 00:21 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_7.zip | Finished | 75.438 | 64.507 | 88.301 | 76.523 | 96.519 | 2088 |
| 538015 | 2026-02-12 13:20 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_8.zip | Finished | 75.725 | 64.782 | 88.600 | 76.465 | 96.518 | 2069 |
| 539879 | 2026-02-13 12:37 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba.zip | Finished | 75.186 | 63.850 | 88.623 | 75.440 | 96.487 | 2136 |
| 541928 | 2026-02-14 20:13 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp2_9.zip | Finished | 75.413 | 64.420 | 88.361 | 76.383 | 96.489 | 2070 |
| 542933 | 2026-02-15 10:47 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp4_1.zip | Finished | 74.775 | 63.192 | 88.564 | 74.879 | 96.479 | 2197 |
| 545141 | 2026-02-16 19:01 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp4_4.zip | Failed | - | - | - | - | - | - |
| 545183 | 2026-02-16 19:14 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp4_4.zip | Finished | 74.824 | 63.369 | 88.430 | 75.062 | 96.483 | 2178 |
| 548196 | 2026-02-18 23:40 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_3.zip | Finished | 76.043 | 65.224 | 88.740 | 76.740 | 96.529 | 2037 |
| 549102 | 2026-02-19 15:08 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_4.zip | Finished | 75.234 | 63.878 | 88.692 | 75.392 | 96.491 | 2171 |
| 552997 | 2026-02-21 00:55 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba._exp5_6_e.zip | Finished | 76.034 | 65.199 | 88.753 | 76.762 | 96.540 | 2009 |
| 553542 | 2026-02-21 08:30 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_6.zip | Finished | 76.096 | 65.309 | 88.750 | 76.855 | 96.535 | 2017 |
| 555592 | 2026-02-22 09:20 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_8.zip | Finished | 74.864 | 63.637 | 88.145 | 75.712 | 96.507 | 2101 |
| 558755 | 2026-02-23 21:11 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_1_76.zip | Finished | 64.123 | 55.672 | 74.018 | 72.957 | 87.300 | 2177 |
| 560028 | 2026-02-24 10:56 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_8_ep.zip | Finished | 76.030 | 65.248 | 88.679 | 76.840 | 96.542 | 2011 |
| 563830 | 2026-02-25 19:43 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp8_1_ep.zip | Failed | - | - | - | - | - | - |
| 563964 | 2026-02-25 19:51 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp8_1_ep.zip | Finished | 74.518 | 62.676 | 88.686 | 74.047 | 96.409 | 2346 |
| 566678 | 2026-02-26 20:04 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp5_11.zip | Finished | 75.988 | 65.182 | 88.671 | 76.842 | 96.540 | 1996 |
| 568234 | 2026-02-27 11:18 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp9_1.zip | Finished | 76.348 | 65.702 | 88.799 | 77.262 | 96.543 | 2013 |
| 582958 | 2026-03-02 03:53 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba.zip | Finished | 76.048 | 65.249 | 88.718 | 76.850 | 96.535 | 1999 |
| 585373 | 2026-03-03 01:08 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp10_4.zip | Finished | 76.265 | 65.543 | 88.822 | 77.129 | 96.556 | 1992 |
| 588283 | 2026-03-04 05:05 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp11_1.zip | Finished | 76.347 | 65.747 | 88.739 | 77.284 | 96.539 | 2004 |
| 589154 | 2026-03-04 13:09 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp13_1.zip | Finished | 76.335 | 65.692 | 88.782 | 77.284 | 96.548 | 1989 |
| 616136 | 2026-03-12 16:54 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp10_1_7.zip | Failed | - | - | - | - | - | - |
| 616517 | 2026-03-12 19:05 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp10_1_7.zip | Finished | 76.533 | 66.041 | 88.774 | 77.575 | 96.541 | 1979 |
| 638373 | 2026-03-20 01:12 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3.zip | Finished | 76.490 | 65.916 | 88.839 | 77.412 | 96.559 | 1990 |
| 650425 | 2026-03-26 01:43 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_5.zip | Finished | 76.370 | 65.702 | 88.850 | 77.266 | 96.562 | 1986 |
| 658200 | 2026-03-31 07:51 | jojo0 | pre-ledger: sportsmot_bytetrack_post.zip | Finished | 64.531 | 53.135 | 78.438 | 72.329 | 95.878 | 2885 |
| 658250 | 2026-03-31 09:01 | jojo0 | pre-ledger: sportsmot_bytetrack_mamba_post.zip | Finished | 61.379 | 49.988 | 75.640 | 68.697 | 92.531 | 4694 |
| 660118 | 2026-04-01 14:51 | jojo0 | pre-ledger: sportsmot_botsort_post.zip | Finished | 73.684 | 62.030 | 87.597 | 75.307 | 97.208 | 2145 |
| 661717 | 2026-04-02 22:14 | jojo0 | pre-ledger: sportsmot_ocsort_post.zip | Finished | 72.620 | 60.050 | 87.926 | 72.893 | 95.722 | 2898 |
| 663326 | 2026-04-04 07:43 | jojo0 | pre-ledger: sportsmot_hybrid.zip | Finished | 75.188 | 63.735 | 88.794 | 75.475 | 96.712 | 2135 |
| 666637 | 2026-04-06 22:39 | jojo0 | pre-ledger: sportsmot_bytetrack_post.zip | Finished | 64.531 | 53.135 | 78.438 | 72.329 | 95.878 | 2885 |
| 670593 | 2026-04-09 09:45 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_7.zip | Finished | 76.396 | 65.765 | 88.826 | 77.320 | 96.537 | 2007 |
| 672407 | 2026-04-10 09:24 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp15_3.zip | Finished | 76.400 | 65.777 | 88.820 | 77.334 | 96.554 | 2017 |
| 675422 | 2026-04-12 13:31 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3_w.zip | Finished | 76.354 | 65.686 | 88.838 | 77.251 | 96.559 | 2009 |
| 678521 | 2026-04-14 11:26 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3_w.zip | Finished | 76.264 | 65.536 | 88.831 | 77.159 | 96.553 | 2004 |
| 682703 | 2026-04-16 19:24 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3_o.zip | Finished | 76.053 | 65.162 | 88.847 | 76.798 | 96.534 | 2037 |
| 712166 | 2026-05-06 02:15 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3_r.zip | Finished | 74.944 | 63.327 | 88.783 | 74.743 | 96.455 | 2255 |
| 713780 | 2026-05-06 23:17 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp18_3.zip | Finished | 76.254 | 65.544 | 88.793 | 77.181 | 96.552 | 2003 |
| 721479 | 2026-05-11 01:58 | jojo0 | pre-ledger: SportsMOT_HybridSORT_post.zip | Finished | 73.178 | 60.789 | 88.196 | 73.200 | 95.877 | 2665 |
| 742364 | 2026-05-20 13:57 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3.zip | Failed | - | - | - | - | - | - |
| 742375 | 2026-05-20 14:05 | jojo0 | pre-ledger: sportsmot_test_0.80_mamba_exp14_3.zip | Finished | 76.503 | 65.924 | 88.858 | 77.398 | 96.556 | 1984 |
| 818318 | 2026-06-29 14:58 | jojo0 | pre-ledger: sportsmot_test_0.80_orig_post_sport.zip | Finished | 75.519 | 64.282 | 88.807 | 75.852 | 96.536 | 2092 |
| 850450 | 2026-07-15 13:08 | jojo0 | pre-ledger: sportsmot_test_0.80_old_e100_iwg_at.zip | Finished | 75.824 | 64.817 | 88.781 | 76.211 | 96.672 | 1950 |
| 852039 | 2026-07-16 22:57 | jojo0 | pre-ledger: sportsmot_test_0.80_trainval_e150_i.zip | Finished | 75.908 | 65.001 | 88.723 | 76.430 | 96.689 | 1931 |
| 852511 | 2026-07-17 08:03 | jojo0 | pre-ledger: sportsmot_test_0.80_trainval_e200_i.zip | Finished | 76.162 | 65.422 | 88.743 | 76.899 | 96.694 | 1878 |
| 853588 | 2026-07-18 10:23 | jojo0 | pre-ledger: sportsmot_test_0.80_h8_bound010_e20.zip | Finished | 76.306 | 65.730 | 88.662 | 77.101 | 96.650 | 1884 |
| 854468 | 2026-07-19 15:49 | jojo0 | pre-ledger: sportsmot_test_0.80_iwg_rg_cma_sqrt.zip | Finished | 76.270 | 65.637 | 88.705 | 77.044 | 96.662 | 1879 |
| 861697 | 2026-07-25 01:41 | jojo0 | pre-ledger: sportsmot_test_0.80_1x6_oldbest_e20.zip | Finished | 76.377 | 65.813 | 88.713 | 77.189 | 96.668 | 1867 |
| 862164 | 2026-07-25 10:39 | jojo0 | pre-ledger: sportsmot_test_0.80_1x6_residual_ta.zip | Finished | 76.287 | 65.672 | 88.694 | 77.107 | 96.673 | 1857 |
| 958646 | 2026-10-03 22:34 | jojo0 | pre-ledger: ByteTrack_post_AFlink.zip | Failed | - | - | - | - | - | - |
| 958671 | 2026-10-03 22:48 | jojo0 | pre-ledger: ByteTrack_post_AFlink.zip | Failed | - | - | - | - | - | - |
| 958864 | 2026-10-04 00:43 | jojo0 | pre-ledger: ByteTrack_post_AFLink_GMC.zip | Failed | - | - | - | - | - | - |
| 958882 | 2026-10-04 00:57 | jojo0 | pre-ledger: ByteTrack_post_AFLink_GMC.zip | Finished | 66.403 | 55.678 | 79.252 | 74.459 | 96.218 | 2252 |
| 959346 | 2026-10-04 09:42 | jojo0 | pre-ledger: ByteTrack_XYWH_GMC_post.zip | Cancelled | - | - | - | - | - | - |
| 1057848 | 2026-11-18 10:45 | jojo0 | pre-ledger: shutrackA_sportsmot_trainval_e200_p.zip | Finished | 75.872 | 64.934 | 88.733 | 76.428 | 96.664 | 1916 |
| 1062928 | 2026-11-21 18:56 | jojo0 | SportsMOT formal e100 (night final) | Finished | 75.727 | 64.648 | 88.790 | 76.085 | 96.661 | 1986 |
| 1063982 | 2026-11-22 16:01 | jojo0 | SportsMOT cost trainval formal e80 | Finished | 75.677 | 64.546 | 88.811 | 76.080 | 96.678 | 1937 |
| 1064222 | 2026-11-22 22:49 | - | SportsMOT midpoint selection e80 | Finished | 75.774 | 64.720 | 88.802 | 76.130 | 96.667 | 1975 |
| 1064937 | 2026-11-23 17:34 | - | SportsMOT ablation e80 (train45, D) | Finished | 75.840 | 64.798 | 88.850 | 76.105 | 96.677 | 1975 |
| 1066994 | 2026-11-25 03:12 | jojo0 | SportsMOT harm=2 E_e100 (post+AFLink) | Finished | 75.979 | 65.095 | 88.766 | 76.755 | 96.659 | 1885 |
| 1067021 | 2026-11-25 04:10 | shangqingnie | SportsMOT harm=4 train45 e100 (post+AFLink) | Finished | 75.988 | 65.324 | 88.475 | 76.959 | 96.642 | 1870 |
| 1068003 | 2026-11-25 20:20 | jojo0 | SportsMOT harm=4 trainval90 e100 (post+AFLink) | Finished | 76.320 | 65.990 | 88.346 | 77.467 | 96.609 | 1848 |

## MOT20（phase 16383，competition 10050）

历史最佳（全架构）：**66.637** — MOT20 formal_e100 (sqrt-size)（#1063086，2026-11-21 21:39）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 456767 | 2025-12-15 20:46 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp4_2.zip | Failed | - | - | - | - | - | - |
| 456812 | 2025-12-15 21:13 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp4_2.zip | Failed | - | - | - | - | - | - |
| 457817 | 2025-12-16 12:45 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp5_1.zip | Finished | 65.047 | 67.507 | 62.832 | 81.296 | 76.303 | 947 |
| 459403 | 2025-12-17 21:22 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp6_2.zip | Finished | 65.096 | 67.566 | 62.874 | 81.323 | 76.335 | 941 |
| 460072 | 2025-12-18 12:04 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp6_3.zip | Failed | - | - | - | - | - | - |
| 461293 | 2025-12-19 13:05 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp6_3.zip | Finished | 65.192 | 67.730 | 62.905 | 81.573 | 76.383 | 912 |
| 468092 | 2025-12-25 14:03 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp7_1.zip | Finished | 65.085 | 67.555 | 62.866 | 81.369 | 76.298 | 927 |
| 469054 | 2025-12-26 10:37 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp7_1.zip | Cancelled | - | - | - | - | - | - |
| 470212 | 2025-12-28 01:21 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp9_2.zip | Finished | 64.938 | 67.379 | 62.740 | 81.249 | 76.207 | 953 |
| 470421 | 2025-12-28 14:08 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp9_3.zip | Finished | 65.225 | 67.851 | 62.854 | 81.579 | 76.323 | 922 |
| 471171 | 2025-12-29 15:00 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp9_1.zip | Finished | 65.131 | 67.719 | 62.794 | 81.549 | 76.243 | 902 |
| 471841 | 2025-12-30 11:47 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp9_5.zip | Finished | 65.054 | 67.621 | 62.730 | 81.557 | 76.246 | 874 |
| 472854 | 2025-12-31 13:58 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp8_2.zip | Finished | 65.078 | 67.574 | 62.826 | 81.436 | 76.330 | 939 |
| 474393 | 2026-01-02 22:15 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp8_6.zip | Finished | 64.961 | 67.383 | 62.780 | 81.295 | 76.302 | 912 |
| 476709 | 2026-01-06 00:11 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp10_3.zip | Failed | - | - | - | - | - | - |
| 477352 | 2026-01-06 16:04 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp10_3.zip | Finished | 64.937 | 67.309 | 62.809 | 81.111 | 76.277 | 935 |
| 480101 | 2026-01-09 00:16 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp12_2.zip | Finished | 65.099 | 67.583 | 62.865 | 81.381 | 76.330 | 916 |
| 483442 | 2026-01-11 23:55 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp10_1_.zip | Finished | 65.037 | 67.511 | 62.811 | 81.355 | 76.327 | 919 |
| 483944 | 2026-01-12 11:49 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp12_4.zip | Finished | 65.030 | 67.496 | 62.811 | 81.343 | 76.327 | 920 |
| 485986 | 2026-01-13 22:19 | jojo0 | pre-ledger: mot20_test_0.80_mamba.zip | Finished | 65.146 | 67.749 | 62.798 | 81.577 | 76.296 | 941 |
| 487361 | 2026-01-14 22:23 | jojo0 | pre-ledger: mot20_test_0.80_mamba.zip | Failed | - | - | - | - | - | - |
| 489228 | 2026-01-15 22:48 | jojo0 | pre-ledger: mot20_test_0.80_mamba_MOT20_exp14_1.zip | Finished | 65.124 | 67.674 | 62.827 | 81.524 | 76.327 | 929 |
| 491810 | 2026-01-17 20:38 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp11_1_.zip | Finished | 65.019 | 67.531 | 62.758 | 81.379 | 76.252 | 954 |
| 493640 | 2026-01-18 22:35 | jojo0 | pre-ledger: mot20_test_0.80_mamba_post.zip | Failed | - | - | - | - | - | - |
| 495067 | 2026-01-19 16:58 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_2.zip | Failed | - | - | - | - | - | - |
| 495115 | 2026-01-19 17:35 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_2.zip | Finished | 65.266 | 68.291 | 62.515 | 82.574 | 76.311 | 930 |
| 498368 | 2026-01-21 23:36 | jojo0 | pre-ledger: mot20_test_0.80_mamba_post.zip | Finished | 66.334 | 69.148 | 63.793 | 83.294 | 77.780 | 593 |
| 499338 | 2026-01-22 17:55 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_2_65.26.zip | Finished | 66.301 | 69.073 | 63.796 | 83.262 | 77.877 | 582 |
| 503944 | 2026-01-26 00:09 | jojo0 | pre-ledger: mot20_test_0.80_mamba.zip | Finished | 65.249 | 68.223 | 62.542 | 82.530 | 76.316 | 894 |
| 504538 | 2026-01-26 15:29 | jojo0 | pre-ledger: mot20_test_0.80_mamba.zip | Finished | 65.218 | 68.208 | 62.497 | 82.533 | 76.282 | 916 |
| 505771 | 2026-01-27 14:40 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_7_epoch.zip | Failed | - | - | - | - | - | - |
| 505910 | 2026-01-27 15:41 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_7_epoch.zip | Failed | - | - | - | - | - | - |
| 506111 | 2026-01-27 17:35 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_7_epoch.zip | Finished | 65.120 | 68.125 | 62.389 | 82.571 | 76.264 | 900 |
| 507391 | 2026-01-28 09:47 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp14_7_post.zip | Finished | 66.376 | 69.301 | 63.727 | 83.546 | 77.903 | 562 |
| 510170 | 2026-01-29 18:22 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp15_1_epoch.zip | Finished | 65.452 | 68.710 | 62.494 | 83.006 | 76.298 | 941 |
| 512007 | 2026-01-30 13:13 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp15_1_epoch.zip | Finished | 66.521 | 69.600 | 63.739 | 83.748 | 77.765 | 577 |
| 513853 | 2026-01-31 12:49 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp15_1_epoch.zip | Failed | - | - | - | - | - | - |
| 515771 | 2026-02-01 09:51 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp15_1_epoch.zip | Finished | 65.281 | 68.369 | 62.474 | 82.898 | 76.331 | 851 |
| 517770 | 2026-02-02 12:05 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp15_1_epoch.zip | Finished | 65.352 | 68.527 | 62.468 | 82.820 | 76.276 | 923 |
| 565817 | 2026-02-26 13:09 | jojo0 | pre-ledger: mot20_test_0.80_botsort_ablation.zip | Failed | - | - | - | - | - | - |
| 565827 | 2026-02-26 13:14 | jojo0 | pre-ledger: mot20_test_0.80_botsort_ablation.zip | Finished | 64.657 | 66.508 | 63.041 | 80.326 | 76.331 | 1310 |
| 582904 | 2026-03-02 03:21 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp16_4_epoch.zip | Finished | 65.259 | 68.344 | 62.457 | 82.746 | 76.321 | 927 |
| 600296 | 2026-03-07 21:24 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp17_1.zip | Finished | 65.439 | 68.723 | 62.452 | 82.961 | 76.267 | 930 |
| 616612 | 2026-03-12 19:43 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp18_2_epoch.zip | Finished | 65.520 | 68.298 | 63.007 | 82.134 | 76.473 | 939 |
| 619076 | 2026-03-13 16:49 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp19_2.zip | Finished | 65.586 | 68.424 | 63.015 | 82.241 | 76.480 | 932 |
| 622563 | 2026-03-14 17:52 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp19_3.zip | Failed | - | - | - | - | - | - |
| 622566 | 2026-03-14 17:53 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp19_3.zip | Finished | 65.550 | 68.376 | 62.992 | 82.282 | 76.451 | 924 |
| 626058 | 2026-03-15 21:24 | jojo0 | pre-ledger: mot20_test_0.80_mamba_MOT17_exp19_1.zip | Finished | 65.001 | 67.249 | 62.990 | 81.236 | 76.368 | 1049 |
| 630364 | 2026-03-17 00:21 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp21_10.zip | Finished | 65.558 | 68.391 | 62.993 | 82.209 | 76.425 | 930 |
| 632810 | 2026-03-17 16:13 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp22_1.zip | Finished | 65.307 | 67.887 | 62.981 | 81.544 | 76.390 | 955 |
| 636395 | 2026-03-18 21:05 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp21_12.zip | Finished | 65.570 | 68.419 | 62.992 | 82.211 | 76.443 | 944 |
| 638080 | 2026-03-19 21:40 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp26_2.zip | Finished | 65.597 | 68.456 | 63.008 | 82.279 | 76.481 | 925 |
| 638783 | 2026-03-20 12:02 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp27_1.zip | Finished | 65.263 | 67.737 | 63.037 | 81.380 | 76.382 | 943 |
| 642271 | 2026-03-22 11:03 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp29_1.zip | Finished | 65.561 | 68.375 | 63.013 | 82.274 | 76.484 | 905 |
| 645026 | 2026-03-23 12:20 | jojo0 | pre-ledger: mot20_test_0.80_mamba_BOTH_exp12_2.zip | Finished | 65.422 | 68.174 | 62.936 | 82.042 | 76.392 | 926 |
| 650551 | 2026-03-26 07:55 | jojo0 | pre-ledger: mot20_test_0.80_botsort_mamba_post.zip | Finished | 62.585 | 63.200 | 62.160 | 76.659 | 75.509 | 781 |
| 654675 | 2026-03-28 21:53 | jojo0 | pre-ledger: hybrid_mot20_test.zip | Finished | 62.079 | 61.905 | 62.481 | 75.939 | 75.422 | 1564 |
| 655407 | 2026-03-29 11:39 | jojo0 | pre-ledger: mot17_bytetrack_test.zip | Finished | 59.929 | 57.840 | 62.304 | 73.308 | 76.212 | 1348 |
| 659978 | 2026-04-01 12:04 | jojo0 | pre-ledger: MOT20_BoTSORT_post.zip | Finished | 62.213 | 61.346 | 63.341 | 75.548 | 76.867 | 1221 |
| 661401 | 2026-04-02 16:37 | jojo0 | pre-ledger: mot20_ocsort_post.zip | Finished | 61.398 | 60.737 | 62.281 | 75.091 | 75.714 | 1058 |
| 663321 | 2026-04-04 07:40 | jojo0 | pre-ledger: MOT20_HybridSORT.zip | Finished | 61.902 | 60.696 | 63.383 | 75.297 | 76.885 | 1292 |
| 664939 | 2026-04-05 20:50 | jojo0 | pre-ledger: mot20_ocsort_post.zip | Finished | 61.716 | 61.013 | 62.646 | 75.257 | 76.027 | 1140 |
| 665842 | 2026-04-06 09:24 | jojo0 | pre-ledger: mot20_hybridsort_mamba_post.zip | Finished | 62.385 | 61.781 | 63.241 | 76.011 | 76.362 | 1310 |
| 667467 | 2026-04-07 11:15 | jojo0 | pre-ledger: mot20_bytetrack_post.zip | Finished | 61.148 | 59.922 | 62.590 | 75.031 | 76.522 | 1146 |
| 673911 | 2026-04-11 10:28 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp_split_con.zip | Finished | 65.618 | 68.527 | 62.982 | 82.434 | 76.507 | 922 |
| 704972 | 2026-05-01 04:46 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp21_4_real_.zip | Finished | 65.741 | 68.678 | 63.083 | 82.520 | 76.507 | 975 |
| 705323 | 2026-05-01 12:55 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp21_4_real_.zip | Finished | 65.884 | 68.844 | 63.206 | 82.578 | 76.509 | 909 |
| 706964 | 2026-05-02 17:56 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp21_4_real_.zip | Finished | 65.985 | 69.031 | 63.228 | 82.764 | 76.538 | 900 |
| 708812 | 2026-05-04 00:04 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp33_1_epoch.zip | Finished | 65.805 | 68.783 | 63.108 | 82.666 | 76.517 | 908 |
| 709510 | 2026-05-04 10:58 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp34_1.zip | Finished | 65.697 | 68.432 | 63.225 | 82.289 | 76.561 | 974 |
| 711910 | 2026-05-05 22:52 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp34_2_epoch.zip | Finished | 65.783 | 68.580 | 63.253 | 82.421 | 76.596 | 958 |
| 713965 | 2026-05-07 01:03 | jojo0 | pre-ledger: MOT20-MoT-ByteTrack.zip | Finished | 62.847 | 62.524 | 63.401 | 76.838 | 77.089 | 939 |
| 714273 | 2026-05-07 09:26 | jojo0 | pre-ledger: MOT20_ByteTrack_mamba_post.zip | Finished | 63.217 | 63.040 | 63.630 | 77.219 | 77.134 | 900 |
| 717119 | 2026-05-08 16:59 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 717232 | 2026-05-08 17:49 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 717638 | 2026-05-08 21:35 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 717739 | 2026-05-08 22:29 | jojo0 | pre-ledger: MOT20_HybridSORT_mamba_exp31_1_epoc.zip | Finished | 61.843 | 60.863 | 63.103 | 75.165 | 76.373 | 1392 |
| 718320 | 2026-05-09 09:13 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 718392 | 2026-05-09 10:21 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 718586 | 2026-05-09 11:51 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_post.zip | Failed | - | - | - | - | - | - |
| 719228 | 2026-05-09 18:19 | jojo0 | pre-ledger: MOT20_OCSORT_mamba_exp31_1_post.zip | Finished | 62.880 | 63.170 | 62.791 | 77.220 | 76.168 | 1022 |
| 721537 | 2026-05-11 03:28 | jojo0 | pre-ledger: MOT20_HybridSORT_post.zip | Finished | 62.430 | 61.883 | 63.227 | 76.087 | 76.365 | 1308 |
| 726398 | 2026-05-13 00:57 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_exp31_1_epoch35.zip | Failed | - | - | - | - | - | - |
| 742437 | 2026-05-20 14:35 | jojo0 | pre-ledger: mot20_test_0.80_mamba.zip | Finished | 66.108 | 69.277 | 63.234 | 82.922 | 76.526 | 898 |
| 765005 | 2026-05-29 17:50 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_exp31_1_epoch35.zip | Failed | - | - | - | - | - | - |
| 765278 | 2026-05-29 19:30 | jojo0 | pre-ledger: MOT20_BoTSORT_mamba_exp31_1_epoch55.zip | Failed | - | - | - | - | - | - |
| 765428 | 2026-05-29 20:50 | jojo0 | pre-ledger: MOT20_ByteTrack_to_BoTSORT_mamba_ex.zip | Finished | 63.360 | 63.425 | 63.533 | 77.154 | 76.965 | 949 |
| 767375 | 2026-05-30 14:21 | jojo0 | pre-ledger: trackssm_mot20_submission_dti_drop_.zip | Failed | - | - | - | - | - | - |
| 767622 | 2026-05-30 15:57 | jojo0 | pre-ledger: trackssm_mot20_submission_dti_drop_.zip | Failed | - | - | - | - | - | - |
| 767782 | 2026-05-30 16:48 | jojo0 | pre-ledger: trackssm_mot20_submission_dti_drop_.zip | Failed | - | - | - | - | - | - |
| 768221 | 2026-05-30 19:06 | jojo0 | pre-ledger: trackssm_mot20_submission_dti_drop_.zip | Failed | - | - | - | - | - | - |
| 818231 | 2026-06-29 13:47 | jojo0 | pre-ledger: mot20_test_0.80_orig_post_mot20_com.zip | Finished | 65.601 | 67.318 | 64.122 | 80.814 | 77.555 | 724 |
| 832762 | 2026-07-08 01:16 | jojo0 | pre-ledger: mot20_test_0.80.zip | Finished | 64.589 | 66.527 | 62.881 | 80.192 | 76.222 | 1054 |
| 839210 | 2026-07-11 00:52 | jojo0 | pre-ledger: mot20_test_0.80_iwg500_agentguard_i.zip | Finished | 65.738 | 67.724 | 64.002 | 81.078 | 77.535 | 690 |
| 849951 | 2026-07-14 22:37 | jojo0 | pre-ledger: mot20_test_0.80_iwg_attn_final_post.zip | Finished | 66.006 | 67.990 | 64.266 | 81.343 | 77.793 | 660 |
| 850407 | 2026-07-15 12:18 | jojo0 | pre-ledger: mot20_test_0.80_best_interleaved_e1.zip | Finished | 66.125 | 68.197 | 64.296 | 81.799 | 77.826 | 654 |
| 851859 | 2026-07-16 18:07 | jojo0 | pre-ledger: MOT20_test_mamba_native_e050_final_.zip | Failed | 65.873 | 67.867 | 64.126 | 81.305 | 77.637 | 706 |
| 852556 | 2026-07-17 09:17 | jojo0 | pre-ledger: mot20_test_0.80_mot20_s4_e100_iwg_a.zip | Finished | 66.273 | 68.610 | 64.196 | 81.958 | 77.750 | 667 |
| 853595 | 2026-07-18 10:59 | jojo0 | pre-ledger: mot20_test_0.80_bound010_e100_iwg_p.zip | Finished | 66.269 | 68.513 | 64.280 | 81.866 | 77.845 | 669 |
| 854465 | 2026-07-19 15:48 | jojo0 | pre-ledger: mot20_test_0.80_iwg_rg_cma_sqrt_siz.zip | Finished | 66.407 | 68.804 | 64.269 | 82.270 | 77.749 | 674 |
| 859208 | 2026-07-23 14:19 | jojo0 | pre-ledger: mot20_test_0.80_iwg_rg_cma_final_po.zip | Finished | 66.345 | 68.643 | 64.306 | 81.955 | 77.932 | 664 |
| 860516 | 2026-07-24 10:52 | jojo0 | pre-ledger: mot20_test_0.80_6x6_iwg_rg_cma_fina.zip | Submitting | - | - | - | - | - | - |
| 862176 | 2026-07-25 10:46 | jojo0 | pre-ledger: mot20_test_0.80_1x6_final_iwg_rg_cm.zip | Finished | 66.359 | 68.728 | 64.253 | 82.121 | 77.710 | 666 |
| 901528 | 2026-08-25 22:42 | jojo0 | pre-ledger: mot20_test_0.80_reidproj256_iwg_rg_.zip | Finished | 65.899 | 68.082 | 63.970 | 81.587 | 77.567 | 711 |
| 919429 | 2026-09-09 08:23 | jojo0 | pre-ledger: mot20_test_0.80_mot20_gt_loo_v2_e10.zip | Finished | 66.339 | 68.739 | 64.200 | 82.224 | 77.703 | 691 |
| 925195 | 2026-09-14 07:04 | jojo0 | pre-ledger: mot20_test_0.80_gtonly_v2_sqrt_e100.zip | Finished | 66.275 | 68.561 | 64.243 | 82.154 | 77.695 | 676 |
| 925784 | 2026-09-14 16:45 | jojo0 | pre-ledger: MOT20_mot20_looonly_v2_sqrt_e100_fi.zip | Finished | 66.205 | 68.477 | 64.189 | 82.030 | 77.666 | 675 |
| 1057845 | 2026-11-18 10:44 | jojo0 | pre-ledger: shutrackA_mot20_sqrtsize_e100_post_.zip | Finished | 66.073 | 68.330 | 64.074 | 81.834 | 77.633 | 650 |
| 1063086 | 2026-11-21 21:39 | jojo0 | MOT20 formal_e100 (sqrt-size) | Finished | 66.637 | 69.283 | 64.270 | 82.661 | 77.752 | 635 |
| 1064223 | 2026-11-22 22:53 | jojo0 | MOT20 midpoint_e20 | Finished | 66.512 | 68.848 | 64.430 | 82.220 | 77.879 | 616 |
| 1064320 | 2026-11-23 00:48 | - | MOT20 e80 round | Finished | 66.635 | 69.200 | 64.342 | 82.518 | 77.812 | 619 |
| 1064623 | 2026-11-23 11:59 | jojo0 | MOT20 low-LR continuation | Finished | 66.587 | 69.170 | 64.276 | 82.534 | 77.748 | 633 |
| 1066968 | 2026-11-25 02:21 | jojo0 | MOT20 sample-proportional e100 | Finished | 66.531 | 68.913 | 64.410 | 82.325 | 77.834 | 632 |

## MOT17（phase 16382，competition 10049）

历史最佳（全架构）：**68.004** — pre-ledger: mot17_test_0.80_mamba_MOT20_exp31_1.zip（#818047，2026-06-29 11:27）
SHUTrack 时期最佳：**67.827** — MOT17 formal_e50 (init MOT20 e100)（#1063098）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 456820 | 2025-12-15 21:15 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp4_2.zip | Finished | 65.806 | 67.223 | 64.670 | 82.634 | 79.938 | 873 |
| 457824 | 2025-12-16 13:03 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp6_1.zip | Finished | 65.328 | 66.073 | 64.849 | 81.362 | 80.060 | 999 |
| 459400 | 2025-12-17 21:19 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp6_2.zip | Failed | - | - | - | - | - | - |
| 459401 | 2025-12-17 21:20 | jojo0 | pre-ledger: mot20_test_0.80_mamba_exp6_2.zip | Failed | - | - | - | - | - | - |
| 459624 | 2025-12-18 00:45 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp6_3.zip | Finished | 66.014 | 67.537 | 64.771 | 83.169 | 80.119 | 888 |
| 460511 | 2025-12-18 21:54 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp1_2.zip | Finished | 65.562 | 66.537 | 64.860 | 82.012 | 80.150 | 903 |
| 461294 | 2025-12-19 13:06 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp6_4.zip | Finished | 65.960 | 67.403 | 64.789 | 83.068 | 80.122 | 873 |
| 462756 | 2025-12-20 22:32 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp2_1.zip | Finished | 65.547 | 66.507 | 64.858 | 82.039 | 80.150 | 933 |
| 468029 | 2025-12-25 12:15 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp7_1_9.zip | Finished | 65.749 | 66.887 | 64.885 | 82.279 | 80.117 | 933 |
| 469056 | 2025-12-26 10:38 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp8_1.zip | Finished | 65.721 | 66.820 | 64.893 | 82.269 | 80.139 | 946 |
| 469749 | 2025-12-27 09:48 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp8_1.zip | Finished | 65.663 | 66.685 | 64.910 | 82.251 | 80.183 | 972 |
| 470713 | 2025-12-28 22:51 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp9_2.zip | Finished | 65.670 | 66.763 | 64.846 | 82.336 | 80.162 | 885 |
| 471495 | 2025-12-29 23:31 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp9_4.zip | Finished | 65.693 | 66.944 | 64.723 | 82.363 | 80.010 | 975 |
| 472269 | 2025-12-31 00:31 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp8_1.zip | Failed | - | - | - | - | - | - |
| 472611 | 2025-12-31 08:23 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp8_1.zip | Finished | 65.916 | 67.497 | 64.624 | 82.739 | 79.959 | 942 |
| 473785 | 2026-01-01 21:36 | jojo0 | pre-ledger: mot17_test_0.80_mamba.zip | Failed | - | - | - | - | - | - |
| 473843 | 2026-01-02 00:09 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp8_2_new.zip | Finished | 65.244 | 65.999 | 64.755 | 81.332 | 79.916 | 972 |
| 474442 | 2026-01-02 23:29 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp8_4.zip | Finished | 65.630 | 66.743 | 64.792 | 82.051 | 79.959 | 1020 |
| 474976 | 2026-01-03 21:54 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp10_1.zip | Finished | 65.076 | 65.711 | 64.706 | 80.991 | 79.920 | 993 |
| 475711 | 2026-01-04 23:07 | jojo0 | pre-ledger: mot17_test_0.80_mamba.zip | Finished | 64.372 | 64.393 | 64.627 | 79.155 | 79.791 | 1257 |
| 476728 | 2026-01-06 00:29 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp10_3.zip | Finished | 65.437 | 66.448 | 64.696 | 81.732 | 80.048 | 969 |
| 480107 | 2026-01-09 00:29 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp12_1.zip | Finished | 67.228 | 68.640 | 66.153 | 83.348 | 81.536 | 798 |
| 483444 | 2026-01-11 23:56 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp10_1_.zip | Finished | 66.004 | 67.477 | 64.815 | 82.925 | 80.121 | 849 |
| 483943 | 2026-01-12 11:49 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp12_4.zip | Finished | 66.004 | 67.477 | 64.816 | 82.925 | 80.121 | 849 |
| 485416 | 2026-01-13 13:38 | jojo0 | pre-ledger: mot17_test_0.80_mamba.zip | Finished | 65.983 | 67.495 | 64.761 | 82.989 | 80.108 | 858 |
| 487360 | 2026-01-14 22:22 | jojo0 | pre-ledger: mot17_test_0.80_mamba.zip | Finished | 65.940 | 67.364 | 64.797 | 82.949 | 80.144 | 882 |
| 489232 | 2026-01-15 22:50 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp11_1.zip | Finished | 65.022 | 65.623 | 64.681 | 80.837 | 79.926 | 1011 |
| 491809 | 2026-01-17 20:37 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp11_1_.zip | Finished | 65.532 | 66.695 | 64.646 | 82.310 | 79.987 | 924 |
| 493599 | 2026-01-18 22:06 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp14_2.zip | Finished | 65.577 | 67.444 | 64.070 | 83.031 | 79.693 | 1038 |
| 508982 | 2026-01-29 01:52 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp11_3_epoch.zip | Failed | - | - | - | - | - | - |
| 509482 | 2026-01-29 11:09 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp11_3_epoch.zip | Finished | 65.068 | 65.588 | 64.816 | 81.066 | 80.032 | 1095 |
| 513066 | 2026-01-31 00:13 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp12_1.zip | Finished | 66.222 | 67.828 | 64.899 | 83.370 | 80.195 | 990 |
| 514264 | 2026-01-31 18:27 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp12_1_epoch.zip | Failed | - | - | - | - | - | - |
| 516992 | 2026-02-02 00:57 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp15_2.zip | Finished | 65.655 | 66.632 | 64.949 | 82.057 | 80.204 | 972 |
| 518151 | 2026-02-02 16:14 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp15_2.zip | Failed | - | - | - | - | - | - |
| 520100 | 2026-02-03 12:43 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp15_2.zip | Finished | 65.565 | 66.497 | 64.906 | 81.933 | 80.117 | 993 |
| 564861 | 2026-02-26 04:00 | jojo0 | pre-ledger: mot17_test_0.80_botsort_ablation_po.zip | Failed | - | - | - | - | - | - |
| 564864 | 2026-02-26 04:01 | jojo0 | pre-ledger: mot17_test_0.80_botsort_ablation_po.zip | Failed | - | - | - | - | - | - |
| 564885 | 2026-02-26 04:14 | jojo0 | pre-ledger: mot17_test_0.80_botsort_ablation_po.zip | Failed | - | - | - | - | - | - |
| 565537 | 2026-02-26 10:43 | jojo0 | pre-ledger: mot17_test_0.80_botsort_ablation_po.zip | Finished | 66.158 | 66.714 | 65.934 | 81.596 | 80.820 | 948 |
| 578566 | 2026-02-28 06:27 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp13_1.zip | Finished | 65.840 | 67.000 | 64.952 | 82.557 | 80.255 | 981 |
| 581926 | 2026-03-01 18:26 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp14_1.zip | Finished | 66.222 | 67.749 | 64.971 | 83.372 | 80.292 | 936 |
| 583366 | 2026-03-02 10:10 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp14_2_epoch.zip | Finished | 65.859 | 67.032 | 64.957 | 82.665 | 80.258 | 960 |
| 595762 | 2026-03-06 12:01 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp16_1_epoch.zip | Finished | 66.110 | 67.644 | 64.850 | 83.164 | 80.113 | 975 |
| 599677 | 2026-03-07 16:08 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp17_1.zip | Finished | 66.020 | 67.289 | 65.028 | 82.872 | 80.346 | 942 |
| 601935 | 2026-03-08 12:05 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp18_1.zip | Finished | 66.059 | 67.484 | 64.908 | 83.085 | 80.250 | 993 |
| 611232 | 2026-03-10 23:30 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp14_4_epoch.zip | Finished | 64.981 | 65.397 | 64.830 | 80.660 | 80.006 | 1056 |
| 612245 | 2026-03-11 10:12 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp14_5.zip | Finished | 66.034 | 67.392 | 64.948 | 83.115 | 80.274 | 936 |
| 614901 | 2026-03-12 08:55 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp14_7.zip | Finished | 66.281 | 67.827 | 65.014 | 83.515 | 80.299 | 945 |
| 623951 | 2026-03-15 06:56 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp19_1_epoch.zip | Finished | 64.347 | 64.356 | 64.605 | 79.203 | 79.791 | 1257 |
| 626404 | 2026-03-15 23:03 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp19_2.zip | Finished | 66.316 | 68.011 | 64.918 | 83.209 | 80.245 | 948 |
| 630300 | 2026-03-16 23:54 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp21_10.zip | Finished | 66.147 | 67.763 | 64.825 | 82.913 | 80.077 | 975 |
| 636753 | 2026-03-19 01:29 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp21_1.zip | Finished | 66.368 | 68.101 | 64.934 | 83.472 | 80.269 | 936 |
| 639936 | 2026-03-20 23:56 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp20_1_epoch.zip | Finished | 66.179 | 67.682 | 64.956 | 83.207 | 80.218 | 990 |
| 640438 | 2026-03-21 11:02 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp21_1.zip | Finished | 66.177 | 67.605 | 65.025 | 83.241 | 80.372 | 978 |
| 642352 | 2026-03-22 12:15 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp29_1.zip | Finished | 66.348 | 68.091 | 64.898 | 83.374 | 80.122 | 939 |
| 645021 | 2026-03-23 12:17 | jojo0 | pre-ledger: mot17_test_0.80_mamba_BOTH_exp12_2.zip | Finished | 66.413 | 68.129 | 64.985 | 83.506 | 80.320 | 954 |
| 650452 | 2026-03-26 02:28 | jojo0 | pre-ledger: mot17_test_0.80_bytetrack_mamba_pos.zip | Finished | 62.974 | 62.841 | 63.618 | 75.166 | 77.324 | 6579 |
| 651146 | 2026-03-26 16:30 | jojo0 | pre-ledger: mot17_test_0.80_bytetrack_mamba_pos.zip | Finished | 63.143 | 61.595 | 65.121 | 75.230 | 79.121 | 4530 |
| 656351 | 2026-03-30 01:08 | jojo0 | pre-ledger: track_results_dti.zip | Finished | 63.031 | 62.169 | 64.216 | 77.262 | 79.583 | 2565 |
| 659982 | 2026-04-01 12:08 | jojo0 | pre-ledger: MOT17_BoTSORT_post.zip | Finished | 64.233 | 63.783 | 64.962 | 79.068 | 80.559 | 1284 |
| 661325 | 2026-04-02 15:31 | jojo0 | pre-ledger: mot17_ocsort_post.zip | Finished | 62.804 | 62.718 | 63.163 | 77.202 | 77.957 | 2001 |
| 663335 | 2026-04-04 07:53 | jojo0 | pre-ledger: MOT17_HybridSORT_mamba.zip | Finished | 61.299 | 58.925 | 64.127 | 73.874 | 79.004 | 2040 |
| 665248 | 2026-04-05 23:22 | jojo0 | pre-ledger: mot17_hybridsort_mamba_post.zip | Finished | 63.898 | 63.421 | 64.690 | 79.176 | 79.857 | 1212 |
| 666634 | 2026-04-06 22:38 | jojo0 | pre-ledger: mot17_bytetrack_post.zip | Finished | 63.031 | 62.169 | 64.216 | 77.262 | 79.583 | 2565 |
| 706630 | 2026-05-02 10:14 | jojo0 | pre-ledger: mot17_test_0.80_mamba_tracktrack_de.zip | Finished | 65.599 | 66.473 | 64.984 | 81.918 | 80.122 | 1047 |
| 707820 | 2026-05-03 10:31 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp32_1.zip | Finished | 66.510 | 68.278 | 65.034 | 83.504 | 80.226 | 897 |
| 710341 | 2026-05-04 22:49 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp31_2.zip | Finished | 66.454 | 68.050 | 65.155 | 83.609 | 80.386 | 888 |
| 711746 | 2026-05-05 21:13 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp34_2.zip | Finished | 66.429 | 67.976 | 65.174 | 83.560 | 80.391 | 933 |
| 713976 | 2026-05-07 01:19 | jojo0 | pre-ledger: MOT17-MoT-ByteTrack.zip | Finished | 62.742 | 62.664 | 63.194 | 77.137 | 78.658 | 2847 |
| 714262 | 2026-05-07 09:19 | jojo0 | pre-ledger: MOT17_ByteTrack_mamba_post.zip | Finished | 61.970 | 61.935 | 62.455 | 75.619 | 76.823 | 4395 |
| 716913 | 2026-05-08 15:42 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp31_2_epoch.zip | Finished | 66.434 | 68.101 | 65.063 | 83.637 | 80.221 | 939 |
| 719738 | 2026-05-10 00:29 | jojo0 | pre-ledger: MOT17_BotSORT_mamba_exp31_2_epoch10.zip | Finished | 64.720 | 64.769 | 64.981 | 79.657 | 80.594 | 1242 |
| 720183 | 2026-05-10 11:44 | jojo0 | pre-ledger: MOT17_OCSORT_mamba_exp31_2_epoch10_.zip | Finished | 62.661 | 62.614 | 62.984 | 76.758 | 77.693 | 2379 |
| 721786 | 2026-05-11 10:26 | jojo0 | pre-ledger: MOT17_HybridSORT_post.zip | Finished | 63.418 | 62.852 | 64.294 | 78.112 | 79.285 | 1986 |
| 728489 | 2026-05-14 00:28 | jojo0 | pre-ledger: MOT17_ByteTracK_mamba_exp31_1_epoch.zip | Finished | 64.400 | 64.125 | 64.967 | 79.327 | 80.572 | 1041 |
| 742913 | 2026-05-20 18:52 | jojo0 | pre-ledger: mot17_test_0.80_mamba.zip | Finished | 66.746 | 68.627 | 65.172 | 83.764 | 80.398 | 879 |
| 765325 | 2026-05-29 19:58 | jojo0 | pre-ledger: MOT17_ByteTrack_mamba_MOT20_exp31_1.zip | Finished | 62.919 | 61.635 | 64.599 | 76.511 | 80.045 | 1353 |
| 767252 | 2026-05-30 13:30 | jojo0 | pre-ledger: MOT17_BotSORT_mamba_exp31_1_epoch10.zip | Finished | 64.698 | 64.948 | 64.755 | 79.849 | 80.348 | 1125 |
| 787566 | 2026-06-09 17:07 | jojo0 | pre-ledger: mot17_test_0.80_mamba_exp31_1_decay.zip | Finished | 66.664 | 68.478 | 65.156 | 83.592 | 80.385 | 867 |
| 818047 | 2026-06-29 11:27 | jojo0 | pre-ledger: mot17_test_0.80_mamba_MOT20_exp31_1.zip | Finished | 68.004 | 69.746 | 66.610 | 84.584 | 82.049 | 762 |
| 832720 | 2026-07-08 00:48 | jojo0 | pre-ledger: mot17_test_0.80.zip | Finished | 65.695 | 66.739 | 64.922 | 82.029 | 80.126 | 1026 |
| 833346 | 2026-07-08 13:10 | jojo0 | pre-ledger: mot17_test_0.80_a_only_bs1024_100e_.zip | Finished | 64.780 | 65.834 | 63.992 | 81.062 | 78.767 | 1098 |
| 839198 | 2026-07-11 00:43 | jojo0 | pre-ledger: mot17_test_0.80_iwg500_agentguard_i.zip | Finished | 65.541 | 66.611 | 64.745 | 82.105 | 79.894 | 993 |
| 841988 | 2026-07-12 13:16 | jojo0 | pre-ledger: mot17_test_0.80_iwg100_tgr30_post.zip | Finished | 67.317 | 68.781 | 66.186 | 83.344 | 81.598 | 846 |
| 847039 | 2026-07-14 02:33 | jojo0 | pre-ledger: mot17_all_0.80_iwg_v3_oldconfig_tra.zip | Failed | - | - | - | - | - | - |
| 847044 | 2026-07-14 02:36 | jojo0 | pre-ledger: mot17_all_0.80_iwg_v3_oldconfig_tra.zip | Failed | - | - | - | - | - | - |
| 847058 | 2026-07-14 02:46 | jojo0 | pre-ledger: mot17_test_0.80_iwg_v3_train42_post.zip | Finished | 66.803 | 68.065 | 65.878 | 83.003 | 81.084 | 891 |
| 848324 | 2026-07-14 13:39 | jojo0 | pre-ledger: MOT17_test_RG_CMA_bs1024_epoch100_f.zip | Finished | 66.737 | 67.857 | 65.941 | 82.270 | 81.165 | 903 |
| 850536 | 2026-07-15 14:36 | jojo0 | pre-ledger: MOT17_test_mot20_old_e050_xfer_fina.zip | Finished | 67.214 | 68.555 | 66.203 | 82.788 | 81.371 | 837 |
| 852301 | 2026-07-17 02:08 | jojo0 | pre-ledger: mot17_test_0.80_mot20_mamba_e100_iw.zip | Finished | 67.270 | 68.592 | 66.270 | 83.383 | 81.543 | 831 |
| 852779 | 2026-07-17 12:47 | jojo0 | pre-ledger: mot17_test_0.80_mot20s4e100_e050_iw.zip | Finished | 67.791 | 69.429 | 66.491 | 83.581 | 81.746 | 771 |
| 853596 | 2026-07-18 11:07 | jojo0 | pre-ledger: mot17_test_0.80_mot20e100_to_mot17_.zip | Finished | 67.784 | 69.455 | 66.445 | 83.651 | 81.753 | 720 |
| 854464 | 2026-07-19 15:47 | jojo0 | pre-ledger: mot17_test_0.80_iwg_rg_cma_sqrt_siz.zip | Finished | 67.438 | 68.764 | 66.436 | 83.373 | 81.682 | 807 |
| 857540 | 2026-07-22 11:04 | jojo0 | pre-ledger: MOT17_test_group_lr05_final_post.zip | Finished | 67.772 | 69.399 | 66.481 | 83.572 | 81.732 | 780 |
| 861195 | 2026-07-24 18:35 | jojo0 | pre-ledger: mot17_test_0.80_6x6_oldbest_e050_iw.zip | Finished | 67.269 | 68.448 | 66.415 | 82.908 | 81.685 | 825 |
| 863719 | 2026-07-26 17:12 | jojo0 | pre-ledger: mot17_test_0.80_best_e050_base_post.zip | Finished | 67.781 | 69.437 | 66.461 | 83.601 | 81.727 | 741 |
| 874639 | 2026-08-02 20:59 | jojo0 | pre-ledger: mot17_test_0.80_compact_iwg_rg_cma_.zip | Finished | 67.368 | 68.605 | 66.455 | 83.260 | 81.728 | 810 |
| 918827 | 2026-09-08 18:52 | jojo0 | pre-ledger: mot17_test_0.80_mot17_gt_loo_v2_e10.zip | Finished | 66.715 | 67.871 | 65.896 | 82.485 | 81.242 | 849 |
| 919602 | 2026-09-09 11:00 | jojo0 | pre-ledger: mot17_test_0.80_gtloo_v2_e050_post.zip | Finished | 67.696 | 69.240 | 66.476 | 83.639 | 81.674 | 768 |
| 1057846 | 2026-11-18 10:44 | jojo0 | pre-ledger: shutrackA_mot17_fromM20interleaved_.zip | Finished | 67.537 | 69.101 | 66.304 | 83.366 | 81.484 | 798 |
| 1063098 | 2026-11-21 21:54 | jojo0 | MOT17 formal_e50 (init MOT20 e100) | Finished | 67.827 | 69.442 | 66.551 | 83.931 | 81.888 | 837 |
| 1064234 | 2026-11-22 23:11 | jojo0 | MOT17 midpoint_e20 | Finished | 67.672 | 68.983 | 66.696 | 83.709 | 82.147 | 813 |
| 1064329 | 2026-11-23 01:05 | - | MOT17 e80 (from MOT20 e80, 50 epochs) | Finished | 67.763 | 69.298 | 66.566 | 84.026 | 81.980 | 819 |
| 1064624 | 2026-11-23 12:00 | jojo0 | MOT17 low-LR continuation | Finished | 67.689 | 69.206 | 66.500 | 83.851 | 81.805 | 846 |
| 1066979 | 2026-11-25 02:45 | jojo0 | MOT17 sample-proportional e50 (from round-1 MOT20) | Finished | 67.814 | 69.434 | 66.527 | 84.282 | 81.998 | 795 |

## DanceTrack（phase 24635，competition None）

历史最佳（全架构）：**66.500** — pre-ledger: dance_test_0.80_post.zip（#639236，2026-03-20 17:32）
SHUTrack 时期最佳：**65.700** — DanceTrack trainval formal e80（#1064153）

| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 639094 | 2026-03-20 16:22 | jojo0 | pre-ledger: SportsMOT_exp14_3.zip | Finished | 65.000 | 50.700 | 83.600 | 66.500 | 93.500 | - |
| 639236 | 2026-03-20 17:32 | jojo0 | pre-ledger: dance_test_0.80_post.zip | Finished | 66.500 | 52.800 | 83.800 | 67.800 | 93.600 | - |
| 639316 | 2026-03-20 18:22 | jojo0 | pre-ledger: dance_test_0.80.zip | Finished | 66.200 | 52.300 | 83.800 | 67.300 | 93.600 | - |
| 640666 | 2026-03-21 15:09 | jojo0 | pre-ledger: dance_exp2_2.zip | Finished | 64.700 | 50.100 | 83.700 | 65.800 | 93.500 | - |
| 640874 | 2026-03-21 18:09 | jojo0 | pre-ledger: dance_exp2_4.zip | Finished | 64.800 | 50.300 | 83.700 | 65.900 | 93.600 | - |
| 641380 | 2026-03-21 23:38 | jojo0 | pre-ledger: exp2_4.zip | Finished | 65.000 | 50.600 | 83.600 | 66.200 | 93.500 | - |
| 850044 | 2026-07-15 01:21 | jojo0 | pre-ledger: dancetrack_iwg_rg_cma_epoch100_fina.zip | Finished | 64.200 | 50.000 | 82.600 | 66.400 | 93.200 | - |
| 850466 | 2026-07-15 13:29 | jojo0 | pre-ledger: tracker.zip | Finished | 65.000 | 50.800 | 83.400 | 66.600 | 93.400 | - |
| 1064153 | 2026-11-22 20:36 | jojo0 | DanceTrack trainval formal e80 | Finished | 65.700 | 51.500 | 83.900 | 66.800 | 93.500 | - |

