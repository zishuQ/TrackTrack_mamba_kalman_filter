# MOT17 best — HOTA 68.229 (#1070802)

- Official test: HOTA 68.229 / AssA 70.349 / DetA 66.467 / IDF1 84.727 / MOTA 82.066 / IDSW 825 / Frag 1368 / MT 1326 / ML 339
- Control #1069953 (same recipe, LR 1e-5): HOTA 68.028 → +0.201 pp
- Second candidate #1070817 (init MOT20 harm=1 e100 #1063086): HOTA 68.167 (+0.139 vs control)
- Recipe: init MOT20 harm=2 e100 #1068664 (sha fe7896f6…), 7 official train sequences,
  sample-proportional, 1 shard, 50 epochs / 5,200 steps, warmup 104, LR 2e-5, dropout 0,
  harm=1.5, column 50, seed 42; architecture query_film_bilinear / proj 256.
- Verified copy from `mot17_harm15_lr2e5_20261127` (weight sha256 in MANIFEST.json).
