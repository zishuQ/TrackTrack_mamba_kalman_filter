---
name: mot-pipeline
description: Run, resume, or audit the MOT17/MOT20 SHUTrack pipeline — autonomously analyze history, pick the next tuning candidate, train or reuse weights, evaluate, run test+post, package, submit to Codabench (one per dataset per day) with duplicate protection, and write daily decision reports. Reads and updates the experiment ledger with configs, official scores and submission IDs.
---

# MOT17/MOT20 pipeline

This skill freezes the verified MOT17/MOT20 workflow: autonomous candidate
selection from history, training (scratch / fine-tune / low-LR continuation),
mid-checkpoint evaluation, test+post, package validation, Codabench submission,
result recording and reporting.

The tuning loop is: analyze the last official score → propose a hypothesis →
select the next candidate from the dataset space → execute → analyze the new
official score → update the search direction.

Paths are relative to the repo root (`TrackTrack/`).

## Files

- `mot_pipeline.py` — the stage driver (dry-run by default).
- `mot_tuner.py` — the autonomous decision loop (dry-run by default).
- `experiments.json` — the ledger: protocols, configs, official scores,
  submission IDs, search state and history. Updated in place with a
  `.json.bak` copy.
- `recipes/` — verified base recipes and per-dataset search spaces
  (`mot20_space.json`, `mot17_space.json`).
- `templates/report.md` — report layout.

## Status (start here)

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py status
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py verify
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py quota --dataset MOT17
```

`status` prints each dataset's official best and every experiment with its
HOTA, submission IDs, status and checkpoint presence. `verify` checks every
ledger path. `quota` queries `can_make_submission` for both accounts and stores
the result under `submission_quota` in the ledger.

Official bests:

| Dataset | Best experiment | HOTA | Submission |
|---|---|---:|---:|
| MOT20 | formal_e100 (epoch_100) | 66.637 | 1063086 (jojo0) |
| MOT17 | formal_e50 (epoch_050) | 67.827 | 1063098 (jojo0) |

## Autonomous tuning loop (mot_tuner.py)

```bash
# analyze history, select the next candidate and write the daily report (no run)
.venv/bin/python .claude/skills/mot-pipeline/mot_tuner.py --dataset MOT20

# propose -> train/reuse -> test -> package -> submit -> poll -> analyze -> report
.venv/bin/python .claude/skills/mot-pipeline/mot_tuner.py --dataset MOT20 --execute
```

One invocation advances one daily decision step. The loop:

1. Polls any non-terminal submission and records its official score.
2. Analyzes the last candidate against its baseline: `improve` (> +0.02 HOTA),
   `degrade` (< -0.02) or `neutral`, plus AssA/DetA/IDF1/MOTA/IDSW/Frag deltas.
3. Chooses the next untested candidate from `recipes/mot20_space.json` /
   `mot17_space.json` (config-hash dedupe against the ledger).
4. On `--execute`: checks the daily budget and both accounts' quota, proposes,
   runs the stages serially, polls the score, updates the search state.

Search policy (generic, dataset-agnostic):

- Numeric axis improves → keep the direction and move to the next step size
  from the new best; degrades → probe the opposite direction at the same
  step; fails both ways → next axis. Neutral → next step.
- Choice axis (`kf_type`, `post_tau`, `post_interval`, `min_len`,
  `max_time_lost`) → next value; `continue_lr` → next LR; `reproduce_seed` →
  one seed rerun of the best training recipe.
- The best official HOTA always stays the base of the next candidate, and the
  best config is never overwritten.

Budget rules:

- At most one official submission per dataset per calendar day
  (`submitted_at` in the ledger). When the budget is used, `--execute` exits
  after writing the report; dry runs still show the plan.
- `can_make_submission` is checked for both accounts; `submit` falls back
  jojo0 → shangqingnie and refuses duplicates.
- Pending submissions are polled instead of resubmitted; a proposed-but-not-run
  candidate is resumed with the same name.

Daily report:
`outputs/agentguard/experiments/mot_pipeline_runs/daily/<date>_<dataset>.md`
with the official best, cumulative evidence table, candidate rationale and
expected effect, config, official result, delta vs baseline, rule-based result
analysis and the next direction.

## Propose a candidate

Parameters are restricted to a whitelist; everything else is locked.

- Tunable training: `mode`, `epochs`, `shards`, `lr`, `batch_size`, `seed`,
  `init`, `source`, `allow_partial_cycles`.
- Tunable inference: `det_thr`, `init_thr`, `match_thr`, `kf_type`,
  `post_interval`, `post_tau`, `penalty_p`, `penalty_q`, `reduce_step`,
  `tai_thr`, `min_len`, `min_box_area`, `max_time_lost`.
- Locked: data splits, train/val/test sequences, data roots, architecture,
  appearance projection, weight decay, snapshot cadence, rank/column weights,
  optimizer and losses.

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py propose \
  --dataset MOT20 --experiment mot20_continue_lr5e6 \
  --base lowlr_continue --set lr=5e-6 --set epochs=50 \
  --reason "halve the continuation LR"
```

`propose` prints the parameter diff vs the base, the `config_hash`, and the
`reuse_from` experiment when an identical configuration already has a
checkpoint. The candidate is stored with `status=proposed`; nothing trains
until `run`.

## Run a candidate

```bash
# plan only (default)
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py run \
  --dataset MOT20 --experiment mot20_continue_lr5e6

# execute all stages serially
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py run \
  --dataset MOT20 --experiment mot20_continue_lr5e6 \
  --stages train,test,package,submit --execute
```

- `run` calls the same verified stage commands documented below, serially.
- If `reuse_from` is set, `train` reuses that checkpoint and `test`/`package`
  reuse its post dir/zip; `submit` is skipped when the source experiment
  already has a submission (no duplicate).
- Completed stages are skipped on rerun (state + artifacts are checked).

## Stages (manual equivalents)

### Train

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py train \
  --dataset MOT20 --experiment mot20_scratch --mode scratch
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py train \
  --dataset MOT17 --experiment mot17_ft --mode finetune \
  --init outputs/agentguard/experiments/shutrack_architecture_night_20261120/final/MOT20/groups/appearance_row_column/checkpoints/epoch_100.pt
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py train \
  --dataset MOT20 --experiment mot20_lowlr --mode continue --epochs 50 --lr 1e-5
```

Defaults mirror the verified recipes: MOT20 scratch 100 epochs/4 shards/lr
1e-4; MOT17 fine-tune 50/1/lr 1e-5 with init MOT20 e100; MOT20 continue
source e100, 50/4/lr 1e-5 with `--allow-partial-cycles`; MOT17 continue
source e50, 50/1/lr 1e-6. Continuation uses `tools/continue_low_lr.py`
(weights-only init, preserved normalization, fresh optimizer/scheduler).

### Eval mid checkpoints

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py eval \
  --dataset MOT20 --experiment mot20_midpoints \
  --checkpoint-dir <ablation/MOT20/.../checkpoints> --epochs 20,40,60,80
```

Writes `evals/eNNN/summary.json` + `gates.jsonl`; completed epochs are skipped
and the ledger's `val_hota`/`best_epoch` are updated on execution.

### Test, package, submit

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py test \
  --dataset MOT17 --experiment mot17_e50 --checkpoint <epoch_050.pt>
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py package \
  --dataset MOT17 --experiment mot17_e50
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py submit \
  --dataset MOT17 --experiment mot17_e50 --execute
```

- `test`: `run.py --mode test --use_post --skip-eval`; expects 4 files (MOT20)
  or 21 (MOT17, 7 FRCNN duplicated to DPM/SDP).
- `package`: `make_package.py` + `codabench_submit.py validate`.
- `submit`: refuses if the ledger already records a submission for the
  experiment; otherwise checks `can_make_submission`, uses jojo0 then
  shangqingnie, records per-account baseline IDs, checks for a new ID before
  retrying after network errors, polls `Submitted/Scoring/...` to
  `Finished/Failed/Cancelled`, and saves ID + scores to state and ledger.

## Report

```bash
.venv/bin/python .claude/skills/mot-pipeline/mot_pipeline.py report \
  --dataset MOT20 --experiment mot20_continue_lr5e6
```

Writes `outputs/agentguard/experiments/mot_pipeline_runs/<experiment>/report.md`
and `report.json`: config diff/reason/hash, reuse source, checkpoint and ZIP
sha256, official scores vs best and base, gate statistics, stage commands and
rule-based suggestions.

## Resume, reuse, duplicate protection

- State: `outputs/agentguard/experiments/mot_pipeline_runs/<experiment>/state.json`
  (stage status, commands, logs, seconds, artifacts). Reruns skip complete
  stages; `--force` where available reruns test/package.
- Weight reuse: a candidate whose `config_hash` matches a completed
  experiment is linked via `reuse_from`; `run` reuses its checkpoint and
  artifacts instead of retraining.
- Submission: the ledger's `submissions` is authoritative. One submission per
  experiment; a saved record is polled rather than recreated; new IDs above
  the recorded per-account baseline are adopted after network failures.
- Heavy jobs: every stage calls `wait_jobs()` and refuses to overlap
  `run.py`/`train_*.py`/`continue_low_lr.py`.

## Ledger rules

- Append experiments; never delete or rewrite history.
- Keep `official_best` as the highest recorded HOTA per dataset.
- `search` per dataset records the loop state (`axis_index`, `step_index`,
  `direction`, `tried_both`, baseline, pending) and the scored history used by
  the daily report. The tuner owns it; hand edits are allowed but must keep
  `config_hash` values intact.
- Phases: MOT20 10050/16383/20804; MOT17 10049/16382/20803.
- Tokens stay outside the repo: `~/.config/codabench/token` (jojo0),
  `~/.config/codabench/token_edge` (shangqingnie); never print them.

## Historical validation example

`hist_replay_lowlr` replays MOT20 `lowlr_continue` without retraining or
resubmitting: `propose` matches the historical `config_hash`, `run` reuses the
checkpoint/post/zip, `submit` is skipped as a duplicate, and `report` compares
66.587 against the best 66.637. See the ledger for that entry.

## Gotchas and troubleshooting

- Continuation must not `--resume` the old scheduler; use `--mode continue`.
- MOT20 50 stage-epochs with 4 shards needs partial cycles (driver sets it).
- MOT17 evaluation uses `--mode all` with `MOT17-09-FRCNN MOT17-10-FRCNN`
  (cache under `detection_cache/MOT17/all`).
- `missing zip`: run package first. `already submitted`: inspect with
  `status`; do not force a duplicate. `can_make=false`: wait or use the other
  token. `manifest.json` missing: wrong split cache (MOT uses `all`).
- `max_time_lost` defaults to `2*frameRate` from `seqinfo.ini`; the tunable
  override (`--max-time-lost-override`, fixed frames) wins over that value.
- Inference-threshold axes apply a uniform value to all sequences; the
  per-sequence MOT17/MOT20 recipe is otherwise kept.
- Session policy: this skill does not create cron/scheduled tasks and does not
  choose session startup/resume; invoke it manually.

## Scope

MOT17/MOT20 only. SportsMOT and DanceTrack experiments live in their own
directories and are not managed here.
