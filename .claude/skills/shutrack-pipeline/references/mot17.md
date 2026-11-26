# MOT17 dataset card

> Last verified: 2026-11-26. Scores/bests are authoritative in
> `submissions.json`; this page is the recipe snapshot.

## Data and sequences

- Training replay: `outputs/agentguard/experiments/shutrack_bd_mot_20261120/MOT17/data`
  (7 train sequences, FRCNN variants). The NSA-c work also has a state-rich
  replay for the 5-sequence subset at
  `/home/shang/.codex/worktrees/shutrack-nsa-reliability/TrackTrack/outputs/agentguard/experiments/nsa_reliability_replay/MOT17/data`
  (02/04/05/11/13-FRCNN, 86,878 samples).
- Splits: train = 02/04/05/09/10/11/13-FRCNN (official MOT17 train, 102,569
  samples); test = 01/03/06/07/08/12/14-FRCNN (duplicated to DPM/SDP when
  packaging). The project's val protocol evaluates 09/10 — both are inside the
  training set, so this is a tracking sanity check, not a held-out set.
- Detection cache: `outputs/agentguard/detection_cache/MOT17/all/` for
  `--mode all`.
- Codabench: competition `10049`, phase `16382`, task `20803`.

## Training recipes (verified)

Fine-tuning from a MOT20 checkpoint (typical):

```bash
.venv/bin/python -u tools/train_supervision_groups.py \
  --group appearance_row_column --dataset MOT17 --split train \
  --init <MOT20 epoch_100.pt> --shards 1 --max-epochs 50 \
  --sequence-sampling sample-proportional --batch-size 1024 --lr 1e-5 \
  --weight-decay 1e-4 --seed 42 --warmup-steps 104 \
  --rank-weight 18 --rank-alpha 0.95 --rank-tau 0.1 --column-weight 50
```

5,200 steps, 104 per epoch. The verified top recipe uses harm=2, dropout=0,
and the round-1 MOT20 harm=2 e100 checkpoint (68.023, #1068665). See
`recipes/mot17_finetune.json` for the initializer path and SHA-256.
The same initializer and harm with dropout 0.05 scored 68.007 (#1069047).
The historical MOT20 formal e100 → MOT17 harm=1 recipe scored 67.827 (#1063098).

The NSA-c MOT17 run (worktree) used the old-gate ablation MOT20 e100 as init,
`--shards 4 --max-epochs 200` (4,800 steps, stop e100/2,400, lr 1e-5,
warmup 24) — the c head is zeroed to c=1 and normalization is refit
(`init_mode=weights_only_refit`).

`--init` inherits the checkpoint's dropout unless `--dropout` is passed.

## Evaluation (raw val)

```bash
cd "3. Tracker"
.venv/bin/python -u run.py --dataset MOT17 --mode all \
  --sequences MOT17-09-FRCNN MOT17-10-FRCNN --seed 10000 --kf-type nsa \
  --agentguard-mode shutrack-v2 --shutrack-checkpoint <epoch_XXX.pt> \
  --agentguard-device cpu \
  --detection-cache-root <repo>/outputs/agentguard/detection_cache \
  --output_dir <exp>/eval/tracking --tracker-suffix <suffix> \
  --agentguard-stats-output <exp>/eval/gates.jsonl \
  --eval-json <exp>/eval/summary.json --print-per-sequence-metrics
```

Reference val baseline: `outputs/agentguard/experiments/mot17_d_midpoints_20261122/evals/e100/summary.json`
(HOTA 67.287 at e100, per-sequence 71.348 / 65.374).

## Test and packaging

- Test output is one txt per sequence (7 FRCNN files); duplicate each
  `*-FRCNN.txt` to `*-DPM.txt` and `*-SDP.txt` → 21 files total.
- Expected count 21; validate with `--dataset mot17 --seqinfo-root /home/shang/datasets/MOT17/test`.
- The established drivers do the FRCNN duplication after `--use_post`.

## Gotchas

- Do not forget the FRCNN→DPM/SDP duplication; validation expects 21 files.
- Evaluation uses `--mode all` with the `all` detection cache split.
- MOT17 val is only 2 sequences, so HOTA deltas are noisy (±0.5pp swings are common).
