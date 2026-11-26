---
name: codabench
description: Query Codabench submission quota, read submission JSON, validate packages, submit once with duplicate protection and poll official scores for this project's tracking competitions (MOT20, MOT17, SportsMOT). Use whenever a task touches Codabench quota, submission records/IDs/status, or score polling.
---

# Codabench submissions (project)

Covers every Codabench interaction in this repo: quota checks, reading
submission JSON, validating packages, submitting once, and polling scores.
Dataset workflows live in `.claude/skills/shutrack-pipeline`; this skill only
owns the Codabench side.

## Hard rules

- **Never submit without an explicit user instruction** ("提交/提交一次/submit").
  A handed-over token, a validated package or a finished training run is not
  approval. This is recorded in project memory (`codabench-submit-confirmation`).
- Never print, log or paste token contents. Tokens stay outside the repo.
- One submission per experiment/result. Prefer polling an existing record or
  submission ID over creating a second one.

## Accounts and tokens

| Account | Token file |
|---|---|
| jojo0 (default) | `~/.config/codabench/token` |
| shangqingnie (Edge) | `~/.config/codabench/token_edge` |

Quota is **per participant** (account), per phase, per day: the two accounts
have independent quotas, and either can be used once per day.

## Quota model (verified on the live API)

- `/api/phases/<phase>/` → `max_submissions_per_day`, `has_max_submissions`,
  `max_submissions_scope` (currently 1/day, scope `participant`).
- `/api/can_make_submission/<phase>/` → `{can, reason}` for the calling token.
- The quota day is Codabench's **UTC day** (00:00–24:00 UTC = 08:00–08:00 CST).
  Recompute "today's usage" from submission `created_when` at every call; do not
  cache a used flag.
- SportsMOT and MOT phases all use the 1/account/UTC-day model.

## Competitions in this repo

| Dataset | Competition | Phase | Task | Seqinfo root | Expected files |
|---|---:|---:|---:|---|---:|
| MOT20 | 10050 | 16383 | 20804 | `/home/shang/datasets/MOT20/test` | 4 |
| MOT17 | 10049 | 16382 | 20803 | `/home/shang/datasets/MOT17/test` | 21 (7 FRCNN ×3 detectors) |
| SportsMOT | 13077 | 21402 | 26486 | `/home/shang/datasets/SportsMOT/dataset/test` | 150 |

## Reading submission JSON

- List: `GET /api/submissions/?phase=<phase>` (list or `{results:[...]}`).
- Detail: `GET /api/submissions/<id>/` → `id`, `status`, `created_when`,
  `scores` (a list of `{column_key, score}` with string scores).
- Statuses: non-terminal = `Submitting / Running / Preparing / Queued / Pending /
  Submitted / Scoring / Waiting / Waiting for scoring`; terminal = `Finished`
  (plus failure states). `Preparing`/`Scoring` can take tens of minutes.

## Submit / validate / poll

Before every submission, check the live state (this is mandatory, including on
a new UTC day — quota resets at 00:00 UTC = 08:00 CST):

```bash
.venv/bin/python .claude/skills/codabench/scripts/codabench_cli.py today --phase <phase>
.venv/bin/python .claude/skills/codabench/scripts/codabench_cli.py quota --phase <phase>
```

`today` lists this UTC day's submissions per account; `quota` gives
`can_make_submission` for both accounts. Pick an account that still `can`, and
never create a second submission for a result that already has an ID.

Validation (local, before any submit):

```bash
.venv/bin/python tools/codabench_submit.py validate \
  --zip <zip> --dataset <mot20|mot17|sportsmot> \
  --seqinfo-root <root> [--sequences-file <file>]
```

Submit once (polls and saves the record):

```bash
.venv/bin/python tools/codabench_submit.py submit \
  --zip <zip> --competition <id> --phase <phase> --task <task> \
  --name <experiment-name> --dataset <flag> \
  --seqinfo-root <root> [--sequences-file <file>] \
  --token-file <token file> \
  --poll --poll-interval 30 --poll-timeout 2400 \
  --save-record <exp>/submissions/<DATASET>_submission.json
```

- The `--save-record` file is the dedupe unit: if it has an `id`, poll it — never
  resubmit.
- After a network failure mid-submit, list submissions above a recorded baseline
  ID before retrying; adopt the recovered ID.
- Existing drivers (`run_mot_e80_round.Driver.submit`, `mot_pipeline.py submit`)
  implement fallback jojo0 → shangqingnie, baseline recovery and duplicate
  refusal. Reuse them when a dataset driver already exists.

Helper (quota / today / poll):

```bash
.venv/bin/python .claude/skills/codabench/scripts/codabench_cli.py quota --phase 21402
.venv/bin/python .claude/skills/codabench/scripts/codabench_cli.py today --phase 16383 --phase 16382
.venv/bin/python .claude/skills/codabench/scripts/codabench_cli.py poll \
  --record <exp>/submissions/SportsMOT_submission.json \
  --token-file ~/.config/codabench/token
```

## Gotchas

- TLS/network to codabench.org is occasionally flaky (`SSL_ERROR_SYSCALL`);
  retry with backoff (the helper and `codabench_submit.py` do).
- `can_make_submission` is per token: always check the account you will submit
  with, and check the other account before declaring "no quota".
- Scores in the API are strings (`"75.9880000000"`); cast before math.
- A submission can sit in `Preparing`/`Scoring` long after the submit tool's
  poll timeout — keep polling the record with the helper.
- Do not upload the same package twice; a duplicate is noise on the leaderboard
  and burns the daily quota.
- **Every scored submission must be recorded** in
  `.claude/skills/shutrack-pipeline/submissions.json` (+ `submissions.md`):
  ID, account, dataset, experiment, all official metrics. This is the most
  important data of the project — record it as soon as the score resolves.
