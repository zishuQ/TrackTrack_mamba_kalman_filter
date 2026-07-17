# IWG + TSRM Joint Training Delivery Report

## 1. Phase checkpoints

- Phase A: `c600495316b1057ac3a8f66cf64107e58683f389`
- Phase B: `c0eee2f6cfcd937f7d703964591eb38799081937`
- Phase C: `7de85555a8bc022cfc0a7961290ef840066cb276`
- Phase D: `25d85a857a478fd63ea567e285b8e6d68efc8828`
- Initial Phase E closeout: `68e0c0feb189960aa835ea2c817914df47498fb3`
- Offline/online IWG warm-up parity fix: `7f03fb8632b6603f7c40ec0f0be159f76eae6e4e`
- Endpoint-only temporal supervision: `bf825e1cfa11e3955d91145fe12d45992570d785`
- Recoverable endpoint pipeline: `d80c018ae1e8ee4daffa1011173aee936cefbb07`
- Temporal-to-IWG gradient isolation experiment: `96fd62badefdcb69ff97d49c45e418d5d3787694`

## 2. Files and responsibilities

- `agentguard/src/agentguard/data/label_schema.py`: rollout label v3 descriptor and hash.
- `agentguard/src/agentguard/rollout_labels.py`, `v0_pipeline.py`, and `rollout/`: benefit direction and v3 soft/safe/cue/risk targets.
- `agentguard/src/agentguard/datasets/joint_window_dataset.py`: full matched/unmatched timelines, exact FRCNN split, five IWG warm-up events, 16 formal events, endpoint mask, and normalization.
- `agentguard/src/agentguard/models/iwg.py`: causal batched sequence IWG, cue head, and risk head.
- `agentguard/src/agentguard/models/tsrm.py`: causal TCN, reset-aware GRU, fusion, bounded correction, and dynamics head.
- `agentguard/src/agentguard/models/iwg_tsrm.py`: 21-event IWG forward, 16-event TSRM crop, unmatched sentinel, and value-preserving temporal-to-IWG gradient scaling.
- `agentguard/src/agentguard/training/loss_iwg_tsrm.py`: masked production base/final/residual/dynamics/revision losses with temporal losses restricted to the final valid endpoint.
- `agentguard/src/agentguard/training/train_iwg_tsrm.py`: sequence-balanced, one-forward/one-backward/one-optimizer trainer, resume, metrics, checkpoint validation, and model schema v4.
- `agentguard/src/agentguard/cli/__init__.py`: dataset build, base/joint training, validation, and gradient-scale CLI.
- `agentguard/src/agentguard/runtime/manager.py`: checkpoint-sized buffers, pre-IWG gap reset, streaming joint inference, unmatched history, and no replay.
- `3. Tracker/run.py`, `3. Tracker/trackers/tracker.py`, and `3. Tracker/integrations/agentguard/`: `joint` mode, strict checkpoint loading, online buffers, and tracker integration.
- `scripts/agentguard/aggregate_iwg_tsrm_holdout.py`: machine-readable TrackEval aggregation restricted to 02/11.
- `scripts/agentguard/run_iwg_tsrm_smoke.sh`, `run_iwg_tsrm_holdout.sh`: recoverable smoke/formal training with provenance.
- `scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh`: strict raw evaluation with collision-free tracker suffixes.
- `scripts/agentguard/run_iwg_tsrm_gradient_isolation_pipeline.sh`: v4 smoke, seed42 joint training, and strict raw evaluation.
- `agentguard/tests/test_iwg_tsrm_*.py`: schema, timeline, causality, endpoint supervision, production gradients, streaming parity, runtime reset, and no-replay coverage.

## 3. Schema migration and split

- Label schema: v3, SHA256 `716df3064d31baa842202df71fc0302337a341c3fcfa2753fe6dd337c4d4c03e`.
- Joint dataset schema: v3, SHA256 `2f61edd55fe7c123f9d3b7c3b658745332df449d0fd81bf95081b76eb27b1462`.
- Base checkpoint schema: `agentguard_iwg_base_v3`.
- Final joint checkpoint schema: `agentguard_iwg_tsrm_v4`.
- Joint v4 strictly records `temporal_iwg_gradient_scale`; v3 joint checkpoints are rejected by the v4 validator.
- Compact event cache remains v3; scalar63 feature semantics and hash are unchanged.
- IWG input has 21 positions: five segment-local warm-up events plus 16 formal events. Only the last 16 IWG outputs enter TSRM.
- `L_final`, `L_residual`, and `L_revision` supervise only the final valid streaming-equivalent endpoint. Base/policy/cue/risk losses use all labeled formal events.
- Train sequences: `MOT17-04-FRCNN`, `MOT17-05-FRCNN`, `MOT17-09-FRCNN`, `MOT17-10-FRCNN`, `MOT17-13-FRCNN`.
- Validation sequences: `MOT17-02-FRCNN`, `MOT17-11-FRCNN`.
- The builder rejects any split other than those exact disjoint FRCNN lists; DPM/SDP cache directories cannot enter training.

## 4. Verification

Commands:

```bash
git diff --check
bash -n scripts/agentguard/run_iwg_tsrm_smoke.sh \
  scripts/agentguard/run_iwg_tsrm_holdout.sh \
  scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh \
  scripts/agentguard/run_iwg_tsrm_gradient_isolation_pipeline.sh
/home/shang/miniconda3/bin/python -m compileall -q \
  agentguard/src agentguard/tests scripts/agentguard \
  '3. Tracker/integrations/agentguard' '3. Tracker/trackers/tracker.py'
/home/shang/miniconda3/bin/python -m pytest \
  agentguard/tests/test_iwg_tsrm_loss.py \
  agentguard/tests/test_iwg_tsrm_checkpoint.py \
  agentguard/tests/test_iwg_tsrm_model.py \
  agentguard/tests/test_iwg_tsrm_runtime.py -q
/home/shang/miniconda3/bin/python -m pytest agentguard/tests -q
```

Results: related suite `19 passed`; complete AgentGuard suite `465 passed`. The production-loss test proves that gradient scale 0 leaves forward values unchanged, gives finite nonzero gradients to TSRM, and blocks temporal/final gradients from IWG. Existing production-loss tests give finite nonzero gradients to encoder, Transformer, policy, IWG residual, cue, risk, TCN, GRU, fusion, delta, strength, and dynamics branches. Offline 21-event and online streaming token/base/final parity is checked at absolute error `<=1e-6`.

## 5. Dataset and smoke

- Dataset metadata: `outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset/metadata.json`.
- Dataset metadata SHA256: `7ad742c80bc58e0b8ebeabd9bd1c916df83cebd7cb789aea101952e23db5f326`.
- Label file-set SHA256: `90f8565049f7f73844969d6ff0a60c98878be95380ee10a58906050f45f4939d`.
- Train/val windows: `23787 / 8524`; complete timeline includes matched and unmatched events.
- v4 smoke checkpoint: `outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_smoke_seed42/checkpoints/iwg_tsrm_last.pt`.
- v4 smoke checkpoint SHA256: `7a87389a93779cce1a28160e95db7a1fd8f6172fb222ec9b2e3647d4b2c30135`.
- Two-epoch train total loss: `1.569915 -> 1.423426`.
- Two-epoch train final loss: `0.627966 -> 0.524068`.
- Two-epoch val final loss: `0.556458 -> 0.529690`.
- Validation was finite and schema-valid; correction remained below `delta_max=0.2` with zero saturation.

## 6. Formal checkpoints and training behavior

- Strict base-only: `outputs/agentguard/experiments/iwg_base_v5_balanced_holdout_seed42/checkpoints/iwg_base_best.pt`.
- Base SHA256: `07131dc2635452b1ffe976a5c3139feff5c4d127b75ed56c7e5619c2c09e10e5`.
- Base best epoch: 0; validation base loss/MAE: `0.663226 / 0.235309` in checkpoint validation.
- Gradient-isolated joint: `outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42/checkpoints/iwg_tsrm_best.pt`.
- Joint SHA256: `769a9ff3fe794d93c16a0e994d18f598059c64fa9126c53e823d7df53cb90e39`.
- Joint best epoch: 1; validation base/final loss: `0.692974 / 0.533857`.
- Joint validation final MAE: `0.248893`; reported base-to-final endpoint error improvement: `0.100889`.
- Joint validation correction mean absolute: `0.151855`; revision strength mean: `0.951764`.
- Base ran 16 epochs because epoch 0 was best and early-stop patience was 15. Joint ran 17 because epoch 1 was best. This is not a one-epoch training failure: train losses continued improving while disjoint validation worsened, demonstrating rapid sequence-domain overfitting.

## 7. Runtime verification

- The earlier 300-frame MOT17-02/all runtime smoke measured joint/base `7.616 FPS` and joint/final `8.268 FPS`; maximum correction was about `0.1925 < 0.2`.
- v4 subsequently ran both complete 02/11 sequences: base-only `53.796 FPS`, joint/base `9.875 FPS`, joint/final `9.743 FPS`.
- v4 resource logs recorded maximum correction below `0.2`, `tgr_calls=0`, `replay_events=0`, and `replay_steps=0`.
- Gradient scaling changes backward behavior only; forward/runtime architecture is unchanged.

## 8. Strict holdout raw tracker results

All cases use `--dataset MOT17 --mode all`, the all-mode detection cache, only disjoint validation sequences 02/11, and tracker seed 10000. Baseline txt was reused and re-aggregated on exactly the same sequences. These are the decision metrics; the previous seven-sequence aggregate includes five training sequences and is supplementary only.

| Case | HOTA | AssA | DetA | IDF1 | MOTA | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.662992 | 0.632866 | 0.699331 | 0.751367 | 0.806082 | 167.181 |
| Strict base-only | 0.670665 | 0.644852 | 0.701955 | 0.770616 | 0.812792 | 53.796 |
| Joint checkpoint / base output | 0.674309 | 0.646407 | 0.707669 | 0.777622 | 0.811115 | 9.875 |
| Joint checkpoint / TSRM final | 0.664785 | 0.633102 | 0.702343 | 0.759096 | 0.806939 | 9.743 |

Per-sequence HOTA:

| Case | MOT17-02-FRCNN | MOT17-11-FRCNN |
|---|---:|---:|
| Baseline | 0.584388 | 0.799199 |
| Strict base-only | 0.597200 | 0.798902 |
| Joint/base | 0.602873 | 0.799142 |
| Joint/final | 0.587199 | 0.798799 |

Paired conclusions:

- Strict base-only versus baseline: `+0.007672` HOTA.
- Joint/base versus strict base-only: `+0.003644` HOTA. Gradient isolation removed the earlier joint representation regression.
- Joint/final versus joint/base: `-0.009523` HOTA.
- Joint/final versus strict base-only: `-0.005880` HOTA.
- Offline safe-target error improvement did not translate to tracking quality. TSRM learned a large predominantly positive correction and harmed association.
- No TSRM HOTA improvement is claimed.

Machine-readable manifest:

```text
outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42/
  evaluation_strict_raw/strict_holdout_raw_manifest.json
```

## 9. Reproduction commands

```bash
ROOT="$(pwd)"
PY="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/3. Tracker:${ROOT}/agentguard/src:${PYTHONPATH:-}"

"${PY}" -m agentguard.cli build_iwg_tsrm_data \
  --dataset MOT17 --mode all --candidate-types A \
  --event-cache-root outputs/agentguard/event_cache_v3_iwg_v2 \
  --detection-cache-root outputs/agentguard/detection_cache \
  --label-dir outputs/agentguard/labels/iwg_rg_cma/MOT17/nsa_candidate_a_json \
  --output-dir outputs/agentguard/experiments/iwg_tsrm_v3_endpoint_holdout_seed42/dataset \
  --split-policy explicit_sequence_holdout \
  --val-sequences MOT17-02-FRCNN MOT17-11-FRCNN \
  --window-size 16 --window-stride 4 --max-frame-gap 30

scripts/agentguard/run_iwg_tsrm_holdout.sh base

RUN_ROOT=outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42 \
TEMPORAL_IWG_GRADIENT_SCALE=0.0 \
  scripts/agentguard/run_iwg_tsrm_holdout.sh joint

RUN_ROOT=outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42 \
TEMPORAL_IWG_GRADIENT_SCALE=0.0 \
RESUME_FROM=outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42/checkpoints/iwg_tsrm_last.pt \
  scripts/agentguard/run_iwg_tsrm_holdout.sh joint

RUN_ROOT=outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42/evaluation_strict_raw \
BASE_CKPT=outputs/agentguard/experiments/iwg_base_v5_balanced_holdout_seed42/checkpoints/iwg_base_best.pt \
JOINT_CKPT=outputs/agentguard/experiments/iwg_tsrm_v4_gradiso_holdout_seed42/checkpoints/iwg_tsrm_best.pt \
TRACKER_PREFIX=gradiso_endpoint \
  scripts/agentguard/eval_iwg_tsrm_holdout_raw.sh
```

The provenance files under each run contain the fully expanded commands. The user later restricted the research decision to pre-post raw results; v4 post evaluation was therefore not run.

## 10. Remaining work and risks

- TSRM final fails the seed42 promotion gate. Seeds 123/3407 and v4 post evaluation were intentionally not run.
- The safe rollout target is misaligned with tracker HOTA: validation favors a mean positive correction around `+0.15`, while tracker association degrades substantially.
- Revision strength is high (`0.952` offline, about `0.745` in the online resource snapshots), so the learned correction is not conservative despite the revision penalty.
- Disjoint validation degrades after epoch 0/1 for both training modes. More epochs do not solve this; they worsen domain overfitting.
- E3-E5/E8 ablations remain unrun because the full TSRM did not pass the seed42 gate. Blind architecture-width or loss-weight search is not justified by current evidence.
- A next research iteration should change the temporal supervision objective or constrain correction application using tracker-grounded evidence. It should not treat lower offline safe-target MAE as sufficient evidence.

## 11. Existing pilot preservation and git state

- `outputs/agentguard/experiments/iwg_v2_a_only_100e_seed42/pipeline.exit_code` remains `0` with timestamp `2026-07-13 01:52:57 +0800`, predating all endpoint/v4 runs.
- All new data, checkpoints, tracker suffixes, logs, and manifests use separate v3/v4 directories. The IWG-v2 pilot was not overwritten or reused.
- At report preparation, the branch was `feature/agentguard-iwg-tsrm-joint`, tracking `origin/feature/agentguard-iwg-tsrm-joint`, ahead by three commits and behind by zero.
