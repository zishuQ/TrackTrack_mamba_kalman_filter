# SportsMOT dataset card

**Reminder: SportsMOT test+post has one extra step vs MOT — after the tracker's
linear post (`--use_post` → `<...>_post`), run AFLink on that directory and
package the `<...>_post_aflink_post` output.**

## Data and sequences

Candidate training roots (all protocol schema 3 / full_write / rank + column):

| Root | Content | Use |
|---|---|---|
| `outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/data` | 90 dirs = official train45 + val45 (593,762 samples) | original E/harm family; select with `--split train` (45) or `--split trainval` (90) |
| `outputs/agentguard/experiments/sportsmot_mechanisms_20261124/data` | train45, 305,631 samples, state-rich | mechanisms / NSA-c train45 runs |
| `outputs/agentguard/experiments/sportsmot_mechanisms_formal_20261124/data` | trainval90, 593,762 samples, state-rich | formal trainval / c trainval90 |

- Official sequence lists: `3. Tracker/trackeval/seqmap/sportsmot/{train,val,test}.txt`
  (first line is the header `name`; 45 / 45 / 150 sequences).
- Detection caches: `outputs/agentguard/detection_cache/SportsMOT/{train,val,test}/`
  (val eval uses the `val` split, test uses `test`).
- Codabench: competition `13077`, phase `21402`, task `26486`;
  seqinfo root `/home/shang/datasets/SportsMOT/dataset/test` (150 dirs).
- Quota: 1 submission per account per UTC day (phase 21402), same model as MOT.

## Training recipes (verified)

Family: `query_film_bilinear` / proj 256, full parameters, batch 1024,
shards 1, seed 42, lr 1e-4 cosine, weight_decay 1e-4, rank 18 / alpha 0.95 /
tau 0.1 / column 50. `motion_harm_weight=4` + `dropout=0.05` is the best
official configuration.

| Run | Data | Epochs / steps | warmup | Official test |
|---|---|---:|---:|---|
| harm=4 trainval90 e100 (#1068003) | 90 | 100 / 62,400 | 312 (0.5%) | **76.320** (current best) |
| harm=4 trainval90 e150 (#1068301) | 90 | 150 / 93,600 | 468 (0.5%) | 76.164 |
| harm=4 train45 e100 (#1067021) | 45 | 100 / 32,100 | 160 (0.5%) | 75.988 |
| harm=2 (E_e100, #1066994) | 45 | 100 / 32,100 | 160 | 75.979 |
| D e80 train45 (#1064937) | 45 | 80 | — | 75.840 (former best) |

Warmup rule: `round(0.005 × total_steps)` — 160/32,100, 312/62,400,
468/93,600 (150 epochs). `snapshot-every 5`. The 90- and 150-epoch runs log
`motion_harm_fraction` / `motion_harm_weight_share` in `metrics.jsonl`; they do
not log per-event harmful/beneficial gate separation (use the separate
gate-target analysis if needed).

Example (trainval90, harm=4):

```bash
.venv/bin/python -u tools/train_supervision_groups.py \
  --group appearance_row_column --dataset SportsMOT --split trainval --from-scratch \
  --data-root outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/data \
  --groups-root <exp>/groups --shards 1 --max-epochs 100 --snapshot-every 5 \
  --batch-size 1024 --lr 1e-4 --weight-decay 1e-4 --seed 42 \
  --rank-weight 18 --rank-alpha 0.95 --rank-tau 0.1 --column-weight 50 \
  --motion-harm-weight 4 --dropout 0.05 --warmup-steps 312 \
  --grad-norm-every 5 --device cuda
```

## Evaluation (raw val)

```bash
cd "3. Tracker"
.venv/bin/python -u run.py --dataset SportsMOT --mode val \
  --sequences <45 val names> --seed 10000 --kf-type nsa \
  --agentguard-mode shutrack-v2 --shutrack-checkpoint <epoch_XXX.pt> \
  --agentguard-device cpu \
  --detection-cache-root <repo>/outputs/agentguard/detection_cache \
  --output_dir <exp>/val/tracking --tracker-suffix <suffix> \
  --agentguard-stats-output <exp>/val/gates.jsonl \
  --eval-json <exp>/val/summary.json --print-per-sequence-metrics
```

- Raw only (no post). Reference old-gate baselines:
  `outputs/agentguard/experiments/sportsmot_de_20261123/evals/D_e020/summary.json`
  (HOTA 83.660). NSA-c comparisons: `nsa_reliability_c_20261125/*/eval_c1/`.
- **Careful**: for trainval90 models, val45 is inside the training data — val
  numbers are optimistic; compare test-side results for decisions.

## Test → post → AFLink → package (the extra AFLink step)

1. Test 150 sequences with `--use_post --skip-eval` (post = MixSort linear
   interpolation):

```bash
cd "3. Tracker"
.venv/bin/python -u run.py --dataset SportsMOT --mode test \
  --sequences <150 test names> --seed 10000 --kf-type nsa \
  --agentguard-mode shutrack-v2 --shutrack-checkpoint <epoch_XXX.pt> \
  --agentguard-device cpu \
  --detection-cache-root <repo>/outputs/agentguard/detection_cache \
  --output_dir <exp>/SportsMOT/test/tracking --tracker-suffix <suffix> \
  --agentguard-stats-output <exp>/SportsMOT/test/gates.jsonl \
  --use_post --skip-eval --torch-num-threads 4 --torch-num-interop-threads 1
```

2. **Extra AFLink step** (reads the `_post` dir, writes `_aflink_post`):

```bash
cd "3. Tracker"
.venv/bin/python -u aflink_use.py \
  --input <out_dir>/<suffix>_post \
  --output <out_dir>/<suffix>_post_aflink_post
```

   Fixed parameters (same as submissions 1067021 / 1068003):
   `thrT=(0,20)`, `thrS=100`, `thrP=0.05`, weights `AFLink/AFLink_epoch20.pth`.
   Expect 150 txt files in the output dir.

3. Package + validate the AFLink output:

```bash
.venv/bin/python outputs/agentguard/experiments/shutrack_a_submit_20261117/scripts/make_package.py \
  --source <..._post_aflink_post> --zip <exp>/submissions/SportsMOT.zip \
  --expected-count 150 --expected-names-file <exp>/logs/sportsmot_test_sequences.txt
.venv/bin/python tools/codabench_submit.py validate --zip <exp>/submissions/SportsMOT.zip \
  --dataset sportsmot --seqinfo-root /home/shang/datasets/SportsMOT/dataset/test \
  --sequences-file <exp>/logs/sportsmot_test_sequences.txt
```

## Gotchas

- `--use_post` on SportsMOT is linear interpolation, **not** AFLink; the AFLink
  pass is always the additional step above.
- On reruns, when discovering the post dir, exclude names ending in
  `_aflink_post` (otherwise two dirs match `*_post`).
- Validation expects exactly 150 files and the official 150-name list.
- SportsMOT quota is per account per UTC day; jojo0 and shangqingnie have
  independent quotas — check both (see the codabench skill).
- The NSA-c (reliability c) work lives in the worktree
  `/home/shang/.codex/worktrees/shutrack-nsa-reliability/TrackTrack` with its
  own trainer and replay caches; summary in
  `outputs/agentguard/experiments/nsa_reliability_c_20261125/handoff_summary.md`.
