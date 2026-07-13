# IWG + TSRM Joint Training Delivery Report

## Phase checkpoints

- Phase A: `c600495316b1057ac3a8f66cf64107e58683f389`
- Phase B: `c0eee2f` (`feat(agentguard): build causal joint window pipeline`)
- Phase C: `7de8555` (`feat(agentguard): train iwg and tsrm jointly`)
- Phase D: `25d85a8` (`feat(agentguard): integrate causal tsrm runtime`)
- P0 offline/online history correction: `7f03fb8`
- Phase E closeout: the commit containing this report

## Files and responsibilities

- `agentguard/src/agentguard/data/label_schema.py`: rollout label v3 descriptor/hash and strict validation.
- `agentguard/src/agentguard/rollout/{motion,appearance}.py`, `rollout_labels.py`, `v0_pipeline.py`: benefit sign documentation and v3 soft/safe/cue/risk labels.
- `agentguard/src/agentguard/datasets/iwg_dataset.py`: strict v3 legacy IWG target loading.
- `agentguard/src/agentguard/datasets/joint_window_dataset.py`: complete matched/unmatched timelines, fixed FRCNN split, 5-event IWG warm-up plus 16-event TSRM windows.
- `agentguard/src/agentguard/data/cache_reader.py`: random access to compact cached events used by joint windows.
- `agentguard/src/agentguard/models/iwg.py`: cue/risk heads and batched causal `forward_sequence()`.
- `agentguard/src/agentguard/models/tsrm.py`: causal TCN, reset-aware GRU, fusion, bounded gate correction, and dynamics head.
- `agentguard/src/agentguard/models/iwg_tsrm.py`: one-pass 21-event IWG encoding, 16-event crop, unmatched sentinel, and TSRM composition.
- `agentguard/src/agentguard/training/loss_iwg_tsrm.py`: production masked base/final/residual/dynamics joint loss.
- `agentguard/src/agentguard/training/train_iwg_tsrm.py`: one-optimizer base/joint trainer, strict checkpoint/resume, metrics, and validation.
- `agentguard/src/agentguard/cli/__init__.py`: build/train/checkpoint-validation commands.
- `agentguard/src/agentguard/runtime/{buffers,manager,statistics}.py`: temporal buffers, pre-IWG gap reset, joint inference, and no-replay statistics.
- `3. Tracker/integrations/agentguard/{adapter,config_bridge}.py`: matched/unmatched online event flow and checkpoint-derived window configuration.
- `3. Tracker/{run.py,trackers/tracker.py}`: joint CLI, strict combined checkpoint loading, and runtime construction.
- `scripts/agentguard/label_diagnostics.py`: v3 label distribution diagnostics.
- `scripts/agentguard/run_iwg_tsrm_{smoke,holdout}.sh`: recoverable, provenance-recorded smoke/formal training.
- `scripts/agentguard/eval_iwg_tsrm_holdout.sh`: all-mode seven-FRCNN-sequence, eight-case raw/post evaluation.
- `agentguard/tests/test_label_schema_v3.py`, `test_rollout_labels.py`, `test_iwg_v2.py`: Phase A schema/gradient contracts.
- `agentguard/tests/test_iwg_tsrm_{dataset,model,loss,runtime,checkpoint}.py`: warm-up windows, causality, production gradients, streaming parity, runtime reset, and strict schema tests.
- Remaining modified AgentGuard tests cover compact cache access, old runtime behavior, initialization, and off-equivalence regressions.

## Schema migration and split

- Rollout label schema remains strict v3: SHA256 `716df3064d31baa842202df71fc0302337a341c3fcfa2753fe6dd337c4d4c03e`.
- Joint dataset schema is v2: SHA256 `b5eb562a1c95388f9355cb8a98e65ed5f8b8563f66ab6388b0875305b16bc866`.
- Joint/base checkpoint schemas are `agentguard_iwg_tsrm_v2` and `agentguard_iwg_base_v2`.
- Each sample has 21 IWG positions (5 segment-local warm-up plus 16 formal events); only the final 16 IWG outputs enter TSRM and supervision.
- Dataset v1 and model v1 checkpoints are explicitly rejected. Compact cache v3 and scalar63 feature hashes were not changed.
- Train: `MOT17-04-FRCNN`, `MOT17-05-FRCNN`, `MOT17-09-FRCNN`, `MOT17-10-FRCNN`, `MOT17-13-FRCNN`.
- Validation: `MOT17-02-FRCNN`, `MOT17-11-FRCNN`.
- The builder ignores DPM/SDP cache directories and asserts the exact fixed FRCNN split.

## Verification

Commands passed:

```bash
git diff --check
bash -n scripts/agentguard/run_iwg_tsrm_smoke.sh
bash -n scripts/agentguard/run_iwg_tsrm_holdout.sh
bash -n scripts/agentguard/eval_iwg_tsrm_holdout.sh
python -m compileall -q agentguard/src agentguard/tests scripts/agentguard \
  '3. Tracker/integrations/agentguard' '3. Tracker/trackers/tracker.py'
PYTHONPATH="$PWD:$PWD/3. Tracker:$PWD/agentguard/src" \
  /home/shang/miniconda3/bin/python -m pytest agentguard/tests -q
```

Result: `459 passed`. The offline 21-event sequence versus online six-event streaming parity test checks formal IWG tokens/base outputs and the endpoint final gate at absolute error `<=1e-6`. Production-loss tests give finite nonzero gradients to encoder, Transformer, policy, IWG residual, cue, risk, TCN, GRU, fusion, delta, strength, and dynamics branches.

## Data and smoke

- Dataset metadata: `outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/dataset/metadata.json`
- Dataset metadata SHA256: `5d0383ee3f355c863a1e6e468a01409944710fe4a10547c6922815c7ec123e9c`
- Label file-set SHA256 recorded by metadata: `90f8565049f7f73844969d6ff0a60c98878be95380ee10a58906050f45f4939d`
- Label summary SHA256: `75925adf1090df7547da10a5764079fadccc0250b709bf63e668be504fc54ab8`
- Smoke checkpoint: `outputs/agentguard/experiments/iwg_tsrm_v2_smoke_seed42/checkpoints/iwg_tsrm_last.pt`
- Smoke checkpoint SHA256: `bb506c061883e12ef913fef34785e3b49376e8148bc40112fbee9b2784759ca0`
- Smoke train final loss: `0.556493 -> 0.469803`; val final loss: `0.715273 -> 0.719864`.
- Smoke validation was finite and schema-valid; correction saturation was zero.

## Formal checkpoints

- Base: `outputs/agentguard/experiments/iwg_base_v4_holdout_seed42/checkpoints/iwg_base_best.pt`
- Base SHA256: `e1c7cbe42f750966e68bda21a75eb2a0a5b19ff9c09144d8bce4a35932e1ec5b`
- Base strict val loss/MAE: `0.679426 / 0.239471`.
- Joint: `outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/checkpoints/iwg_tsrm_best.pt`
- Joint SHA256: `ab8b96657bf84ffb76c90d1c215b6d40bed459c471a9f12041b3873473a82ae6`
- Joint strict val base/final loss: `0.670852 / 0.542307`.
- Joint strict val final MAE: `0.247064`; base-to-final MAE improvement: `0.098609`.
- Both formal runs selected the first completed epoch (372 optimizer steps) and early-stopped after 16 epochs.

## Runtime smoke

- MOT17-02-FRCNN, `--mode all`, 300 frames, joint/base: `7.616 FPS`.
- Same checkpoint, joint/final: `8.268 FPS`.
- Maximum correction magnitude: approximately `0.1925 < delta_max=0.2`.
- Both modes recorded `tgr_calls=0`, `replay_events=0`, and finite gate/correction/strength statistics.

## All-mode seven-sequence tracker evaluation

All cases used tracker seed 10000 and the cache under `outputs/agentguard/detection_cache/MOT17/all`. The all-seven aggregate contains five training sequences and is supplementary to the leakage-free 02/11 holdout.

| Case | HOTA | MOTA | IDF1 | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline raw | 0.770209 | 0.885963 | 0.873123 | 0.773406 | 0.770913 | 173.572 |
| Base-only raw | 0.775749 | 0.887406 | 0.882472 | 0.776737 | 0.778548 | 46.552 |
| Joint/base raw | 0.774259 | 0.886186 | 0.879614 | 0.777317 | 0.775063 | 10.851 |
| Joint/final raw | 0.774036 | 0.886382 | 0.879450 | 0.776108 | 0.775783 | 10.283 |
| Baseline post | 0.774442 | 0.894993 | 0.876322 | 0.779094 | 0.774255 | 172.308 |
| Base-only post | 0.780843 | 0.899516 | 0.887558 | 0.783065 | 0.783122 | 46.858 |
| Joint/base post | 0.777999 | 0.896133 | 0.882560 | 0.781559 | 0.779006 | 10.604 |
| Joint/final post | 0.778059 | 0.897451 | 0.883852 | 0.780884 | 0.779745 | 10.627 |

Joint/final versus strict base-only is `-0.001713` HOTA raw and `-0.002784` HOTA post. Joint/final versus joint/base is `-0.000223` raw and `+0.000060` post. Seed42 therefore does not trigger seeds 123/3407. No HOTA improvement is claimed for TSRM.

Per-sequence full metrics and resource logs are stored under:

```text
outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/evaluation_all7/logs/
outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/evaluation_all7/resources/
```

All joint resource logs have `tgr_calls=0`, `replay_events=0`, and maximum correction below `0.2`.

## Reproduction commands

```bash
ROOT="$(pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" -m agentguard.cli build_iwg_tsrm_data \
  --dataset MOT17 --mode all --candidate-types A \
  --event-cache-root outputs/agentguard/event_cache_v3_iwg_v2 \
  --detection-cache-root outputs/agentguard/detection_cache \
  --label-dir outputs/agentguard/experiments/iwg_tsrm_v1_phase_a/labels \
  --output-dir outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/dataset \
  --split-policy explicit_sequence_holdout \
  --val-sequences MOT17-02-FRCNN MOT17-11-FRCNN \
  --window-size 16 --window-stride 4 --max-frame-gap 30

scripts/agentguard/run_iwg_tsrm_smoke.sh
scripts/agentguard/run_iwg_tsrm_holdout.sh base
scripts/agentguard/run_iwg_tsrm_holdout.sh joint

RESUME_FROM="outputs/agentguard/experiments/iwg_tsrm_v2_holdout_seed42/checkpoints/iwg_tsrm_last.pt" \
  scripts/agentguard/run_iwg_tsrm_holdout.sh joint

scripts/agentguard/eval_iwg_tsrm_holdout.sh
```

The evaluation script expands to all eight baseline/base-only/joint-base/joint-final raw/post commands with `--dataset MOT17 --mode all` and all seven FRCNN sequences.

## Remaining risks

- Strong sequence-domain overfitting remains. Base train loss fell `0.689658 -> 0.538256` while validation worsened after the first completed epoch. Joint train final loss also improved while validation degraded.
- MOT17-04 dominates the window count; overlapping windows amplify sequence imbalance. This was not addressed because it would change the frozen research configuration.
- Revision strength trends toward saturation on later training epochs. The selected first-epoch checkpoint is less extreme, but TSRM did not beat the strict base-only tracker.
- The all-seven aggregate includes training sequences; only 02/11 are leakage-free validation sequences.
- Seeds 123 and 3407 and E3-E5/E8 ablations were intentionally not run because seed42 joint/final did not improve on strict base-only.

## Existing pilot preservation

`outputs/agentguard/experiments/iwg_v2_a_only_100e_seed42/pipeline.exit_code` remains `0`. The implementation, v2 data, training, smoke, and evaluation used new directories and did not reuse or overwrite the pilot directory.
