# AgentGuard RG-CMA Cleanup Phase 0 Baseline

This is the controlled audit record for Phase 0 of
`AGENTGUARD_RG_CMA_CLEANUP_TODO.md`.

## Repository state

- Branch: `feature/agentguard-iwg-tsrm-joint`
- HEAD: `f3cbda9b98e1d291459da3fa30d444b6e683207b`
- Audit base from the TODO: `135e453881c79c6dcec7de855c39d72e9dd9f7ec`
- HEAD is five commits ahead of the audit base.
- The range from the audit base to HEAD contains 74 changed paths, with
  3,625 insertions and 7,878 deletions. The range already removed several
  legacy paths, so the TODO deletion list is treated as a candidate list only.

The pre-existing worktree changes at the start of Phase 0 were:

```text
 M agentguard/IWG_RG_CMA_COMMAND_GUIDE.md
 M agentguard/src/agentguard/cli/__init__.py
 M agentguard/src/agentguard/training/train_iwg_rg_cma.py
 M agentguard/tests/test_iwg_rg_cma_checkpoint.py
?? scripts/agentguard/run_iwg_rg_cma_group_lr_mot17_finetune50.sh
?? scripts/agentguard/run_iwg_rg_cma_mot20_lossv2_best_schedule.sh
```

These paths are not part of the Phase 0 commit.

## Repository conventions and environment

- No `AGENTS.md` exists at `/home/shang`, `/home/shang/workspace`, or the
  repository root.
- Package metadata and the test entry point are in
  `agentguard/pyproject.toml`.
- The package uses `setuptools`; the project script is `agentguard.cli:main`.
- Pytest configuration uses `testpaths = ["tests"]`.
- The repository `.venv` is a uv-created Python 3.10 environment, but its
  Python symlink points to the missing path
  `/home/shang/.local/share/uv/python/cpython-3.10.18-linux-x86_64-gnu/bin/python3.10`.
- Baseline commands therefore used the available system Miniconda Python
  3.12.7 with `PYTHONPATH=agentguard/src` and the repository's existing
  dependencies: pytest 9.1.1, torch 2.12.1+cpu, numpy 2.5.1, and scipy 1.18.0.
- TrackTrack `run.py --help` cannot start in this environment because the
  optional `sklearn` dependency is unavailable. No dependency file was changed.

## Baseline commands

The following commands were run before any Phase 0 repository edit:

```text
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src python -m pytest --collect-only -q
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m agentguard.cli --help
```

Results:

- Python compilation passed.
- Test collection found 454 tests.
- Full baseline: 447 passed, 7 failed, 3 warnings.
- AgentGuard CLI help completed and still exposed the legacy command groups
  `build_student_v0_data`, `train_student_v0`, Teacher/Verifier commands, and
  `iwg-attn` commands. This is an expected cleanup finding, not a Phase 0
  failure.

The seven pre-existing full-suite failures are all in
`tests/test_detection_cache_compact_event_cache.py`. They try to load the
missing path `scripts/agentguard/00_split_detection_cache.py` and fail with
`FileNotFoundError`. No Phase 0 source edit caused these failures.

Focused baseline results with `PYTHONPATH=src:'../3. Tracker'`:

```text
tests/test_iwg_rg_cma_model.py tests/test_iwg_rg_cma_checkpoint.py
tests/test_iwg_rg_cma_runtime.py tests/test_iwg_v2.py
35 passed, 3 warnings

tests/test_runtime.py tests/test_runtime_adapter_initialization.py
tests/test_gate_endpoint_branches.py tests/test_online_off_equivalence.py
tests/test_checkpoint_normalization_loading.py
52 passed

tests/test_iwg_attn_dataset.py tests/test_rollout_labels.py
tests/test_motion_rollout.py tests/test_appearance_rollout.py
tests/test_features.py tests/test_feature_dimension.py
tests/test_scalar_features_not_default_zero.py
99 passed

tests/test_cache_roundtrip.py plus the compact-cache tests not requiring the
missing split script
15 passed, 7 deselected
```

## Deterministic fixture summary

The synthetic IWG + RG-CMA runtime fixture produced:

- track IDs: `[1, 2]`
- association fixture result: `[[0, 0], [1, 1]]`, with no unmatched tracks
  or detections
- scalar feature input shape: 63
- event embedding shape: 128
- motion and appearance token shapes: 64 each
- base, final, applied gate shapes: `[2]` each
- policy, cue, and risk shapes: `[5]`, `[3]`, and `[4]`
- attention shapes: motion `[4, 6]`, appearance `[4, 6]`, and cross-modal
  `[4, 3, 3]`
- unmatched event gate and base gate: `[0.0, 0.0]`
- unmatched policy sentinel: `[0.0, 0.0, 0.0, 1.0, 0.0]`
- off-mode event buffers and window buffers remained empty
- off-mode statistics after one event: `total_events=1`, `iwg_calls=0`,
  `tgr_calls=0`, `replay_events=0`

## Data and checkpoint smoke baselines

Real compact data and matching checkpoints are present. The following
commands completed:

```text
PYTHONPATH=agentguard/src python -m agentguard.cli \
  validate_iwg_attn_checkpoint \
  --checkpoint outputs/agentguard/experiments/iwg_rg_cma_mot20_lossv2_best_schedule_bound005_seed42_bs1024_100e/checkpoints/iwg_rg_cma_last.pt \
  --dataset-dir outputs/agentguard/datasets/iwg_rg_cma/MOT20/nsa_all_v3_compact \
  --device cpu --max-batches 1
```

Validation returned `status=valid`, `epoch=100`, `samples=64`, model schema
`agentguard_iwg_rg_cma_v1`, correction bound `0.05`, and maximum absolute
correction `0.0436826907`.

The CPU-only smoke trainer also completed one epoch over four samples, two
optimizer steps, and wrote `/tmp/agentguard-phase0-smoke-checkpoint/iwg_rg_cma_last.pt`.
The resolved optimizer groups were `base_iwg=0.0001` and `rg_cma=0.0001`.

## Reference matrix

| Symbol or path group | Current definition | Current production callers | Current tests | Planned handling |
|---|---|---|---|---|
| `TGR` | `agentguard/src/agentguard/models/tgr.py`, `training/train_tgr.py` | `3. Tracker/trackers/tracker.py`, `runtime/manager.py`, legacy CLI | runtime, replay, TGR, and stage tests | Remove only after current Runtime path is isolated; Phase 5/6 |
| `TSRM` / `IWGTSRM` | No corresponding model file remains in current HEAD | residual names in legacy docs/scripts and old compatibility references | old replay and phase tests | Re-scan after each phase; do not infer from the base TODO list |
| `iwg-attn` / `IWGAttn` | `datasets/iwg_attn_dataset.py`; combined model is `models/iwg_rg_cma.py` | current CLI, `3. Tracker/run.py`, adapter, config bridge, training and experiment scripts | dataset, model, checkpoint, and runtime tests | Rename to `iwg-rg-cma` in Phase 2/3 |
| `CheckpointManager` / replay | `runtime/checkpoint.py`, `runtime/replay.py` | `runtime/manager.py`, adapter and `replay_backend.py` | replay, checkpoint-roll, runtime, snapshot tests | Remove callers first, then implementation in Phase 3/5/6 |
| `restore_state` / `revised_gate` | Track and event/serialization contracts | replay backend, Runtime, adapter | replay and serialization tests | Preserve `snapshot_state`; remove only after zero replay callers |
| `StudentV0` / `Teacher` / `Verifier` | `v0_pipeline.py`, `teacher/`, `verifier/`, old data modules | current monolithic CLI still registers the old commands; compact label builder still imports `v0_pipeline` | teacher, verifier, label fusion, old cache tests | Extract current rollout builder in Phase 1; delete old pipeline in Phase 5/6 |
| `joint_window_dataset.py` | Already absent from current worktree and HEAD | no current RG-CMA import found | no current direct test import | Phase 1 extracts helpers from the current Dataset instead of deleting this path |
| `build_compact_rollout_labels_for_sequence` | `agentguard/src/agentguard/v0_pipeline.py:167` | compact label writer and CLI | rollout label tests | Move to `data/rollout_label_builder.py` in Phase 1 |
| v3 label fields | `data/label_schema.py`, compact writer, Dataset and old IWG path | current Dataset/loss/checkpoint contract still consume multiple diagnostic fields | `test_label_schema_v3.py`, `test_iwg_v2.py` | Define compact schema v4 in Phase 4 |
| `EventCacheReader` / legacy serialization | `data/cache_reader.py`, `contracts/serialization.py` | cache reader exports, adapter fallback, old CLI | cache and serialization tests | Keep only compact reader in Phase 5/6 |
| Mamba shadow/distillation | `motion/mamba_shadow.py`, `data/mamba_distill.py`, and CLI Mamba module are already absent from current HEAD | no current consumer found by the full scan | no current Mamba test file in HEAD | Do not delete any additional Mamba path; reassess before Phase 5 |

## Static scan snapshot

The required scans were run across `agentguard/src`, `agentguard/tests`,
`3. Tracker`, and `scripts`. The current production residuals include TGR
Runtime/replay branches, `iwg-attn` names, `CheckpointManager`, old cache and
serialization exports, and the Student-V0/Teacher/Verifier CLI/data chain.
The exact file lists are intentionally not treated as deletion authorization;
each symbol will be re-scanned immediately before its phase-specific removal.

## Phase 0 gate

- Baseline branch, SHA, status, environment and test results recorded.
- Full reference scan completed after detecting divergence from the audit base.
- No production behavior was modified.
- Existing data, checkpoint, off-mode, capture/cache, model, and training smoke
  paths have executable baseline evidence, with known environment/test gaps
  recorded above.
- This file is the only Phase 0 file to be committed.
