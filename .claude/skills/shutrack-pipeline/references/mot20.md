# MOT20 dataset card

> Last verified: 2026-11-26. Scores/bests are authoritative in
> `submissions.json`; this page is the recipe snapshot.

## Data and sequences

- Training replay: `outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT20/data`
  (4 sequences: MOT20-01/02/03/05, 1,102,800 samples; protocol schema 3,
  full_write, rank + column).
- Splits: train = 01/02/03/05; val = 01/05; test = 04/06/07/08.
- Detection cache: `outputs/agentguard/detection_cache/MOT20/` — `--mode all`
  reads the `all/` split.
- Codabench: competition `10050`, phase `16383`, task `20804`.

## Training recipes (verified)

Common: `--group appearance_row_column`, `query_film_bilinear` / proj 256,
full parameters, seed 42, batch 1024, weight_decay 1e-4, rank 18 / alpha 0.95 /
tau 0.1 / column 50, harm 1.

| Recipe | Command highlights | Steps | Official |
|---|---|---:|---|
| formal e100 (sqrt-size) | harm=1, dropout 0, 4 shards × 100, warmup 271 | 27,100 | **66.637**, #1063086 |
| harm=2 e100 (sqrt-size) | same schedule, dropout 0 | 27,100 | 65.213, #1068664 |
| harm=1 e100 (sqrt-size) | same schedule, dropout 0.05 | 27,100 | 66.481, #1069017 |
| sample-proportional e100 | default sampling, `--warmup-steps 270`, `--shards 4 --max-epochs 100` | 27,000 | 66.531, #1066968 |
| low-LR continuation | `tools/continue_low_lr.py` from e100, 50 stage-epochs / 4 shards (`--allow-partial-cycles`), lr 1e-5 | — | 66.587 |

Training data loader: staged features by default; `--preload-features` is the
old mode. Resume compares the full config minus loader keys.

## Evaluation (raw val)

```bash
cd "3. Tracker"
.venv/bin/python -u run.py --dataset MOT20 --mode all \
  --sequences MOT20-01 MOT20-05 --seed 10000 --kf-type nsa \
  --agentguard-mode shutrack-v2 --shutrack-checkpoint <epoch_XXX.pt> \
  --agentguard-device cpu \
  --detection-cache-root <repo>/outputs/agentguard/detection_cache \
  --output_dir <exp>/eval/tracking --tracker-suffix <suffix> \
  --agentguard-stats-output <exp>/eval/gates.jsonl \
  --eval-json <exp>/eval/summary.json --print-per-sequence-metrics \
  --torch-num-threads 4 --torch-num-interop-threads 1
```

Reference val summaries from the old-gate midpoints:
`outputs/agentguard/experiments/mot20_d_midpoints_20261122/evals/e020/summary.json`
(HOTA 79.704 at e20, per-sequence 77.939 / 79.754).

## Test and packaging

- Test 4 sequences → post; expected 4 txt files.
- Package: `make_package.py --expected-count 4`; validate with
  `codabench_submit.py validate --dataset mot20 --seqinfo-root /home/shang/datasets/MOT20/test`.
- The c (NSA reliability) experiment line was abandoned on 2026-11-26; its
  checkpoints and state-replay caches were deleted. The tracker still supports
  `--agentguard-disable-kf-gate` (force the KF motion value to 1, keep the EMA
  gate learned) for future ablation work.

## Gotchas

- sqrt-size uses with-replacement sampling and a different warmup (271 vs 270).
- A 50 stage-epoch continuation with 4 shards needs partial cycles.
- `--resume` requires the identical config (loader keys are ignored).
