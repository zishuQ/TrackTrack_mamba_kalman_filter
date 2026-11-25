---
name: shutrack-pipeline
description: Run, resume, or audit the SHUTrack tracking pipeline for MOT17, MOT20 and SportsMOT — training (scratch / fine-tune / low-LR continue), raw val evaluation, test+post, SportsMOT's extra AFLink step, package validation, and Codabench submission via the codabench skill (duplicate protection, one submission per account per UTC day). Also hosts the MOT17/MOT20 autonomous tuning loop and experiment ledger (dry-run by default; the loop is paused until the user re-enables it).
---

# SHUTrack pipeline (MOT17 / MOT20 / SportsMOT)

This skill freezes the verified SHUTrack workflows. Paths are relative to the
repo root (`TrackTrack/`).

Start here, then open the dataset card:

| Dataset | Reference | One-line reminder |
|---|---|---|
| MOT20 | [references/mot20.md](references/mot20.md) | 4 test seqs; phase 16383; sqrt-size vs sample-proportional sharding |
| MOT17 | [references/mot17.md](references/mot17.md) | 7 FRCNN test seqs duplicated to DPM/SDP (21 files); phase 16382; init from MOT20 |
| SportsMOT | [references/sportsmot.md](references/sportsmot.md) | 150 test seqs; phase 21402; **extra AFLink step after the linear post** |

Submissions, quota, tokens and polling are **not** documented here: use the
project skill `.claude/skills/codabench`.

## Shared workflow

1. **Prepare** — confirm the dataset card's data root, sequence list and
   detection cache. Check no heavy job is already running (`run.py`,
   `train_supervision_groups.py`, `train_stop_after.py`, `continue_low_lr.py`).
2. **Train** — `tools/train_supervision_groups.py` (`--group appearance_row_column`,
   `query_film_bilinear`, proj 256, staged features by default). Verify the
   effective `config.json` after launch: `architecture_variant`,
   `appearance_proj_dim`, `loading_mode`, `dropout`, `warmup_steps_override`,
   `total_steps`, `sequences`. `--init` inherits the init checkpoint's dropout
   unless `--dropout` is passed explicitly.
3. **Evaluate (raw val)** — `3. Tracker/run.py` with `--eval-json` and
   `--agentguard-stats-output`; seed 10000, NSA KF, CPU. Reproduce the old
   baseline's detection cache / GMC / thresholds / sequences exactly. Default
   policy: evaluate only the final checkpoint unless the user asks for midpoints.
4. **Test + post** — `run.py --mode test ... --use_post --skip-eval`, then
   package + local validate. Expected file counts: MOT20 4, MOT17 21 (7 FRCNN
   triplicated), SportsMOT 150.
5. **SportsMOT only — extra AFLink**: `--use_post` on SportsMOT runs linear
   interpolation; run `3. Tracker/aflink_use.py --input <..._post> --output
   <..._post_aflink_post>` afterwards with `thrT=(0,20)`, `thrS=100`,
   `thrP=0.05` and `AFLink/AFLink_epoch20.pth`. Package the `_aflink_post` dir.
   Full details in the SportsMOT card.
6. **Submit** — read `.claude/skills/codabench/SKILL.md` first; never submit
   without an explicit user instruction.
7. **Record the score** — every official test submission, once scored, must be
   appended to `submissions.json` + `submissions.md` in this skill (ID,
   account, dataset, experiment, all metrics). This is the project's most
   important data; never leave a scored submission unrecorded.

## MOT17/MOT20 ledger and autonomous tuner (MOT only)

- `mot_pipeline.py` — stage driver (dry-run by default): `status`, `verify`,
  `quota`, `propose`, `run`, `train`, `eval`, `test`, `package`, `submit`,
  `record`, `report`.
- `mot_tuner.py` — autonomous decision loop (dry-run by default; **paused**
  until the user explicitly re-enables it — do not start it on your own).
- `experiments.json` — ledger (protocols, configs, official scores, submission
  IDs, search state). Append-only; keep `experiments.json.bak`.
- `submissions.json` / `submissions.md` — the full Codabench submission ledger
  (every official test submission for MOT20/MOT17/SportsMOT/DanceTrack, with
  scores and submission IDs). Append after every scored submission.
- `recipes/` — MOT20/MOT17 base recipes and search spaces; `templates/report.md`.

```bash
.venv/bin/python .claude/skills/shutrack-pipeline/mot_pipeline.py status
.venv/bin/python .claude/skills/shutrack-pipeline/mot_pipeline.py verify
.venv/bin/python .claude/skills/shutrack-pipeline/mot_pipeline.py quota --dataset MOT17
.venv/bin/python .claude/skills/shutrack-pipeline/mot_pipeline.py propose \
  --dataset MOT20 --experiment mot20_continue_lr5e6 \
  --base lowlr_continue --set lr=5e-6 --set epochs=50 --reason "halve the continuation LR"
.venv/bin/python .claude/skills/shutrack-pipeline/mot_pipeline.py run \
  --dataset MOT20 --experiment mot20_continue_lr5e6 --stages train,test,package,submit --execute
```

MOT official bests (ledger): MOT20 66.637 (formal_e100, #1063086),
MOT17 67.827 (formal_e50, #1063098). Later runs are recorded as ledger entries
too (`sample_proportional_e100` 66.531 / #1066968, `sample_proportional_e50_from_round1_mot20`
67.814 / #1066979).

The SportsMOT results are not in the ledger; they live in their experiment
directories (see the SportsMOT card).

Tuner policy highlights (full text in `mot_tuner.py`):

- One submission per dataset per quota day; the quota day is Codabench's UTC
  day (00:00–24:00 UTC = 08:00–08:00 CST) and is recomputed from
  `submitted_at` at every invocation.
- Numeric axis improves → keep direction; degrades → probe opposite; neutral →
  next step; the best official HOTA is always the base of the next candidate.
- Pending submissions are polled, never resubmitted.

## Gotchas

- **Never name a driver `run.py`/`train_supervision_groups.py`** if it must
  coexist with the pipeline: `active_jobs()` matches scripts by basename and a
  background driver named `run.py` deadlocks the tracker/trainer against it.
- The quota is per **participant** (account) per phase per UTC day
  (max 1). Two tokens = two independent quotas; SportsMOT phase 21402 has the
  same limit.
- `--init` uses the init checkpoint's `model_config` (including dropout) unless
  `--dropout` is passed explicitly; verify the launch `verification:` line.
- Staged feature loading is the default (`stage_features=true`); `--preload-features`
  disables it. Loader choice must not affect the optimization protocol; staging
  vs mmap equivalence is regression-tested.
- Repo scripts (`run.py --use_post`) create `<...>_post`; AFLink writes
  `<...>_post_aflink_post`. When Rediscovering post dirs on a rerun, exclude
  names ending in `_aflink_post`.
- MOT17 test needs the FRCNN→DPM/SDP file duplication; MOT20/MOT17 evaluation
  uses `--mode all` and the `detection_cache/<dataset>/all` caches; SportsMOT
  uses `--mode val`/`test` with the `train`/`val`/`test` cache splits.
- Final-only evaluation is the default policy; do not add midpoint evals unless
  the user asks.
- This skill does not create cron/scheduled tasks and does not start the
  autonomous tuner on its own.
