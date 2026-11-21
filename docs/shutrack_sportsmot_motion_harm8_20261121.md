# SportsMOT-only motion harm-weight ablation

Date: 2026-11-21

## Scope

Only an opt-in SportsMOT B/D training experiment. No changes to collection
protocol, KF/EMA write-back, architecture, inference tensors or MOT17/MOT20
weights/configuration. Existing optimized batch loading/prefetch/transfer paths
remain unchanged.

New argument: --motion-harm-weight, default 1.0.
This experiment passes 8 only in the SportsMOT command. MOT17/MOT20 commands
keep the default 1; there is no dataset/group rejection rule.
The setting is recorded in config/checkpoint training_config and compared on
resume. Old configs missing the field are interpreted as 1.0.

## What changes

For each valid training event, using the two stored next-frame box endpoints:

L0 = 1 - DIoU(box_g0, GT_t1)
L1 = 1 - DIoU(box_g1, GT_t1)

harmful = (L1 > L0 + 1e-6)
w = 8 if harmful else 1

Train the original GT box loss using sum(w*L(g))/sum(w), with invalid rows
masked. The fixed 1e-6 tolerance only excludes numerical ties.
Weights are detached and depend only on stored counterfactual endpoints + GT.

The factor does NOT become a target gate. A harmful event is not forced to g=0;
its optimum can be intermediate. This is not old soft-gate label supervision.

Current D appearance rank, fallback and softplus column supervision stay as-is.
No speed loss, multi-frame loss, gate-mean regularization or temperature change.

## Why try factor 8

Only the training split was used for the audit:
300457 valid rows, 40923 (13.62%) have a worse g=1 endpoint than g=0.

Fixed-training-state weighted target audit:
- factor1/2/4 still prefer g=1 among the scanned constant gates.
- factor8 has best scanned constant gate about0.8.
- At g=0.99, average dL/dg changes from -0.04105 to +0.00820;
  at g=0.5 it remains negative (-0.01501).
- Thus the audited objective still pushes open from the middle but pushes back
  near1, rather than rewarding saturation throughout the range.
- factor8 gives harmful rows55.78% of the total valid-row weight in this audit.

This is a training-target diagnostic, NOT a model improvement.
A uniform0.8 strategy would still be a shortcut; lowered mean gate alone is not
success. The weighting changes the objective's preferred risk tradeoff and may
hurt ordinary detection updates or raw HOTA.

## Verification performed

Seven dedicated loss tests:
- factor1 loss and model-parameter gradients bitwise equal to old formula;
- a harmful example is pushed toward less writing, a beneficial one toward more;
- intended gradient ratio, valid-mask behavior, finite gradients;
- no direct gradient into appearance logits;
- all-invalid batch yields zero loss/gradients;
- invalid numeric factors rejected.

Latest verification: 7 dedicated tests passed; trainer compilation passed.
The default factor remains 1.0, preserving existing loss behavior.
Only tiny synthetic tensors were used; no formal model training or tracker replay
was started during active collection.

No direct appearance-logit gradient does not mean independent feature learning:
the two losses still share model encoders.

Files:
- shutrack/training/loss.py
- tools/train_supervision_groups.py
- shutrack/tests/test_motion_harm_weight.py
- tools/analyze_motion_reweight.py

Audit:
outputs/agentguard/experiments/shutrack_collection_fix_20261121/motion_harm_weight_audit.json

## One proposed training run after collection

Baseline: the completed NEW-256 D model, SportsMOT official train45 -> val45,
HOTA0.8334. Keep from-scratch, seed42, batch1024, lr1e-4, wd1e-4, shards4,
sample-proportional,200 shard-epochs/18800 steps, rank_weight18,
rank_tau0.1, rank_alpha0.95, column_weight50, softplus, proj256,
query_film_bilinear. Use the same immutable45-sequence training view.

Use a new experiment output directory. Change only motion_harm_weight from1 to8.

Do not start alongside the current collection/validation chain. The following
command is prepared for later execution and has NOT been run:

```bash
.venv/bin/python -u tools/train_supervision_groups.py \
  --dataset SportsMOT --group appearance_row_column \
  --architecture-variant query_film_bilinear --appearance-proj-dim 256 \
  --data-root outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/data \
  --groups-root outputs/agentguard/experiments/shutrack_sportsmot_motion_harm8_20261121/groups \
  --split train --from-scratch --device cuda --batch-size 1024 \
  --lr 1e-4 --weight-decay 1e-4 --seed 42 --max-epochs 200 \
  --snapshot-every 20 --shards 4 --sequence-sampling sample-proportional \
  --rank-weight 18 --rank-tau 0.1 --rank-alpha 0.95 --column-weight 50 \
  --motion-harm-weight 8
```

Then reproduce the baseline raw SportsMOT45-val evaluation protocol with the new
epoch200 checkpoint and a new output path: seed10000, NSA, same detector cache,
CPU4 threads, no interpolation/postprocessing or gate-disable flags.
Do not select a higher-scoring snapshot from the fixed-endpoint comparison.

## What would count as useful

Report HOTA/AssA/IDF1/DetA, IDSW/Frag, and full per-sequence deltas vs NEW D.
Inspect whether harmful and beneficial event groups are better separated,
rather than only reporting the overall gate mean. Train GT harm labels must not
be exposed as inference inputs.

Potential benefit: better recognition of a minority of updates that worsen
next-frame prediction, with possible reduction in later association failures.
Risk: suppressing legitimate observations because of short-horizon/noisy GT
preference, reducing DetA or losing identity benefits on other sequences.

The earlier5.9% GT-oracle one-step loss gap is not an expected HOTA gain and is
not an attainable promise for this weighting. No claimed metric improvement
until the training and val evaluation have actually completed.
