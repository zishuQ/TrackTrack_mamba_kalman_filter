# AgentGuard Implementation Summary

## 362 tests passing (0 failures) | Stage 1 + Stage 2 wired and tested

---

## What is AgentGuard?

A motion/appearance gating network that works alongside TrackTrack (CVPR 2025). TrackTrack decides which detection pairs with which track. AgentGuard decides, for each accepted match, how much motion info and how much appearance info to write into the track state. Both gates are continuous values in [0, 1].

AgentGuard does NOT: change TrackTrack's matches, re-run Hungarian assignment, output new Track/Detection IDs, modify detection boxes, touch NMS/TPA/TAI, or call multimodal LLMs online.

---

## File Layout

```
TrackTrack/
├── 3. Tracker/
│   ├── run.py                              # CLI entry (+ agentguard args)
│   ├── trackers/
│   │   ├── tracker.py                      # Tracker.update() (5 AG insertion points)
│   │   ├── track.py                        # Track class (+ update_with_gates etc.)
│   │   ├── utils.py                        # iterative_assignment() (+ return_meta)
│   │   ├── kalman_filter.py                # NSA KF (unchanged)
│   │   └── cmc.py                          # CMC warp (apply_cmc, unchanged)
│   └── integrations/agentguard/            # Bridge layer (only this dir knows both sides)
│       ├── adapter.py                      # AgentGuardTrackerAdapter
│       ├── converters.py                   # Track ↔ TrackStateSnapshot conversions
│       ├── config_bridge.py                # argparse → runtime config
│       ├── hooks.py                        # Lifecycle stubs
│       └── replay_backend.py               # Real TrackTrackReplayBackend (no ReplayEngine)
│
├── agentguard/
│   ├── pyproject.toml
│   ├── configs/                            # runtime.yaml, data.yaml, training.yaml, splits/
│   └── src/agentguard/
│       ├── contracts/                      # Data protocol (enums, states, events, outputs)
│       ├── features/                       # 63-dim scalar feature builder
│       ├── models/                         # EventEncoder, IWG (per-frame), TGR (temporal)
│       ├── runtime/                        # Online manager, buffers, replay plans, stats
│       ├── motion/                         # Pure NumPy NSA KF (parity with TrackTrack KF)
│       ├── data/                           # Event cache, GT reader/matching, candidates,
│       │                                   #   identity vote, future oracle, event sink
│       ├── rollout/                        # Motion/appearance benefit, TGR window labels
│       ├── labels/                         # Soft targets, dataset-level tau, policy targets
│       ├── datasets/                       # IWG/TGR PyTorch datasets, samplers, split validation
│       ├── training/                       # IWG/TGR training loops, metrics, checkpointing
│       ├── oracle/                         # OracleGateProvider (hard gates from rollout labels)
│       ├── teacher/                        # Bailian client, event selector (PLACEHOLDER)
│       ├── verifier/                       # Cross-event scoring, label fusion (PLACEHOLDER)
│       ├── evaluation/                     # TrackEval wrapper
│       ├── hard_events/                    # (NOT IMPLEMENTED - needs V1 failure stats)
│       └── cli/                            # 12 subcommands for full pipeline
│
├── scripts/agentguard/                     # 14 shell scripts (00-13 pipeline)
└── outputs/agentguard/                     # All runtime outputs
```

---

## Modified TrackTrack Files (minimal diff)

| File | Changes |
|------|---------|
| `track.py` | Added `snapshot_state()`, `restore_state(snapshot)`, `update_with_gates(frame_id, detection, mg, ag)`. Endpoint branches for exact 0.0/1.0 gates. NaN/Inf validation. |
| `utils.py` | `iterative_assignment()` gained `return_meta=False` parameter. When True, extra dict with iou/cos/angle/raw_cost/final_cost/assignment_round/threshold/source matrices. False preserves original behavior exactly. |
| `tracker.py` | 5 minimal insertion points in `update()` and `update_without_detections()`: (1) begin_frame, (2) pre-CMC snapshots, (3) post-predict snapshots, (4) gated updates for mature tracks, (5) finalize_first_stage for TGR replay. One CMC read per frame via `_current_warp`. Non-mature tracks and AgentGuard-off path identical to original. Formal checkpoint protocol with scalar_dim=63, event_dim=128, policy_prototypes 5x2, normalization mean/std shape 63, feature_schema_sha256 validation. No fallback: iwg/full mode errors if model/checkpoint missing. `runtime.init_feature_builder()` called exactly once. |
| `run.py` | Added `--agentguard-mode off/iwg/full`, `--iwg-checkpoint`, `--tgr-checkpoint`, `--agentguard-device`. Output folder naming appends `_agentguard_iwg` or `_agentguard_full`. Fixed parser.error NameError. |

---

## Architecture: How Gating Works Online

### Mature Track Definition
A track is "mature" for AgentGuard when: `state in {Tracked, Lost} AND len(history) >= 6`. Non-mature tracks get the original `Track.update()`.

### Per-Frame Processing Order
```
begin_frame (save image dims)
→ save frame_start_state snapshots (pre-CMC) for mature tracks
→ CMC warp (ONE read: stored in self._current_warp)
→ Kalman predict
→ save pre_update_state snapshots (post-CMC+predict) for mature tracks
→ first-stage association (return_meta=True when AG enabled)
→ for each accepted match:
    if mature: build event → IWG inference → update_with_gates
    if not mature: original Track.update()
→ for each unmatched track:
    if mature: build unmatched event → record to buffers → mark_lost
    if not mature: mark_lost
→ finalize_first_stage:
    full mode: TGR inference on full windows → build ReplayPlans → replay into live Tracks
→ second-stage association (unchanged)
→ TAI init (unchanged)
```

### Gate Computation
```
IWG: 6 events (current+5 history, left-padded) → Transformer(2 layers, d=128) →
     policy_head(5) → softmax → policy⊕P → g_mix
     gate_residual_head(2) → tanh
     gate = clip(g_mix + 0.15·tanh(residual), 0, 1)

TGR: 4-event window → EventEncoder → +IWG policy+gate projections → Transformer →
     per-position residual_head(2)
     revised_gate = clip(gate + 0.5·tanh(residual), 0, 1)
     (no-detection positions forced to [0,0])
```

### Motion Gate Effect (in update_with_gates)
```
mean = mean_prior + mg * (mean_full - mean_prior)
cov = (1-mg) * cov_prior + mg * cov_full  → symmetrize
effective_box = (1-mg) * pred_box + mg * det_box
Exact endpoints: mg=1.0 → direct full update, mg=0.0 → keep prior
```

### Appearance Gate Effect
```
feat_full = EMA(track.feat, det.feat, det.score)
feat = (1-ag) * old_feat + ag * feat_full  → L2 normalize
Exact endpoints: ag=1.0 → full EMA, ag=0.0 → keep old
```

### TGR Replay (full mode only)
```
1. Runtime builds ReplayPlan(dict[track_id → checkpoint+steps+revised_gates])
2. Adapter gets plans, finds live Track objects (errors if missing)
3. TrackTrackReplayBackend: restore checkpoint → for each step:
   apply_warp via apply_cmc → Track.predict() → update_with_gates() or mark_lost()
4. Live Track mutated in-place to post-replay state. No ReplayEngine used.
```

### Checkpoint Rolling (M1)
Checkpoint roll is executed by the adapter through the real replay backend:
- save current live snapshot
- restore the oldest window checkpoint
- replay only the oldest event through real `Track` methods
- save that result as the next checkpoint
- pop the oldest window event and remove its old checkpoint
- restore the saved live snapshot

---

## Milestone 1 Changes

### cli/__init__.py
- `PROJECT_ROOT` resolved via `Path(__file__).resolve().parents[4]` with asserts for `agentguard/pyproject.toml` and `3. Tracker/`.
- `MODE_TO_SPLIT` mapping routes mode strings to split names via `_resolve_split()`.
- All `_PROJECT_ROOT` references replaced with `PROJECT_ROOT` (str).

### runtime/manager.py
- `event_sink` explicitly `None` in `__init__`.
- `EventFeatureBuilder` guarded by `_feature_builder_initialised` flag — never re-initialised.
- Full mode: missing TGR model or checkpoint raises `RuntimeError` (no silent skip).
- `finalize_first_stage` returns `dict[int, ReplayPlan]` and records pending checkpoint-roll metadata for the adapter.
- `run_iwg_inference`: iwg/full mode errors on missing model instead of silently returning dummy values.

### adapter.py
- `event_sink` uses `getattr(agentguard_runtime, 'event_sink', None)`.
- No longer calls `runtime.init_feature_builder()` (tracker does it once).
- Replay plan missing live track: raises `RuntimeError` instead of silently continuing.
- Adapter performs checkpoint roll with the real replay backend while preserving the current live track state.

### tracker.py
- Formal checkpoint protocol: requires `model_state_dict`, `reid_dim`, `scalar_dim`, `event_dim`, `policy_prototypes`, `normalization_mean`, `normalization_std`, `feature_schema_sha256`.
- Validation: `scalar_dim==63`, `event_dim==128`, `policy_prototypes` shape `(5,2)`, mean/std shape `(63,)`, ReID dimension cross-checked against weight shapes.
- State dict load order: `model_state_dict` → `state_dict` → entire checkpoint.
- No fallback: `iwg`/`full` mode errors if checkpoints are missing (no random/no-model).
- `runtime.init_feature_builder(reid_dim=checkpoint_reid_dim, norm_stats=checkpoint_norm_stats)` called exactly once.
- `_current_warp` stored per frame (one CMC read); disabled GMC → identity.

### replay_backend.py
- Completely rewritten: no `ReplayEngine` dependency.
- Restores live track through native `restore_state()` when available.
- For each step: applies warp via `apply_cmc`, calls `live_track.predict()`, then either `update_with_gates()` (matched) or `mark_lost()` (unmatched).
- `_ReplayDetection` wrapper bridges `DetectionObservation` to `Track.update_with_gates` interface.
- Unmatched replay preserves `end_frame_id`.

### oracle/gate_provider.py
- Hard gate: `1.0 if benefit >= -1e-6 else 0.0`.
- Benefit sign unified: `B = L_skip - L_write` (positive = write beneficial).

### labels/__init__.py
- Docstring updated to explicitly state benefit sign `B = L_skip - L_write`.

---

## 63-Dimensional Scalar Features

Built by a single `EventFeatureBuilder.build()` used by BOTH online inference and offline data construction.

| Indices | Category | Content |
|---------|----------|---------|
| 0-7 | Pair costs | IoU sim/dist, cos dist, conf dist, angle dist, raw/final cost |
| 8 | Assignment | min(round,10)/10 |
| 9-12 | Detection | score, source one-hot (high/low/nms_del) |
| 13-18 | Row stats | min, 2nd_min, gap, mean, std, normalized entropy |
| 19-24 | Col stats | Same for detection column |
| 25 | Overlap | Max IoU with other same-frame dets |
| 26 | Flag | has_detection |
| 27-30 | Geometry | (det-pred) residuals and size ratio |
| 31-38 | Normalized positions | pred and det boxes in image coords |
| 39-42 | KF velocity | mean[4:8] normalized by pred size |
| 43-46 | KF uncertainty | log of P diagonal elements |
| 47-54 | Velocity | 4×2 velocity flat (C-order) |
| 55-62 | Context | track score, recent obs scores, history len, gap, state one-hot |

No-detection events: indices 0-25 (association) and 27-42 (detection geometry) are zeroed; KF features (43-54) and context (55-62) preserved.

---

## Training Pipeline (Student-V0)

### IWG Training
- Loss: `BCE(gate, target) + 0.1·KL(policy_target||policy_pred) + 0.1·|gate-target|²`
- Per-sample weighted (not batch-total scalar multiplied)
- AdamW, lr=3e-4, cosine scheduler, AMP, grad clip 5, early stop patience 5
- Position embeddings: learned `(1, 6/4, 128)` Parameter, normal init std=0.02

### TGR Training
- Frozen EventEncoder + IWG, only TGR trained
- Loss: `MaskedBCE(revised_gate, target_window) + 0.02·|delta_g|²`
- AdamW, lr=2e-4, same scheduler/AMP/clip/patience

### Checkpoint Schema
```python
{
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "epoch": int,
    "reid_dim": int,
    "scalar_dim": 63,
    "event_dim": 128,
    "policy_prototypes": [...],
    "normalization_mean": ndarray,
    "normalization_std": ndarray,
    "feature_schema_sha256": str,
    "config": {...},
}
```

### Label Construction Flow
```
1. Read cached events + GT matches
2. TrackIdentityVoteState: resolve target_gt_id per (sequence, gt_id) key
3. Identity prototypes: per (seq, gt_id), score≥0.6 ∧ IoU≥0.7, top 70% cosine
4. A/B/C candidates: A=accepted, B=wrong-ID lowest cost, C=same-ID low-quality
5. FutureOracle: per event, save current_gt_box + up to 5 future gt_boxes
6. Motion rollout: write/skip branches → B = L_skip - L_write
7. Appearance rollout: same branches, B = L_skip - L_write
8. Dataset-level tau: median(|B| for B≠0), clipped to [1e-3, 0.1]
9. Soft targets: sigmoid(benefit/tau) for Student training
10. Hard oracle gates: 1 if B ≥ -1e-6, 0 otherwise
11. TGR window labels: enumerate 16 sequences per window
```

### Event IDs (Deterministic, no UUID)
```
Base:     {dataset}/{sequence}/{frame_id:06d}/{track_id:06d}
Candidate:{base}/A, {base}/B, {base}/C
Window:   {sequence}/{track_id}/{start_frame}-{end_frame}/{variant}
```

---

## Detection + Event Cache Protocol

### Directory Structure
```
outputs/agentguard/detection_cache/{dataset}/{split}/{sequence}/
├── manifest.json
├── frame_offsets.npy
├── boxes.npy
├── scores.npy
├── features.npy
├── sources.npy
├── class_ids.npy
├── target_frame_offsets.npy
└── target_detection_indices.npy

outputs/agentguard/event_cache/{dataset}/{split}/{sequence}/
├── manifest.json
├── frames_00000.pt
├── events_00000.pt
├── states_00000.pt
└── associations_00000.pt
```

### Atomicity
1. Write to `{sequence}.incomplete/` temp directory
2. `manifest.json` starts with `"complete": false`
3. All shards written and verified
4. `manifest.json` updated to `"complete": true`
5. Atomic rename to final directory

---

## Test Suite

All tests verified in this pass (390 total, 0 failures):

- `60` Milestone 1 focused tests passed
- `26` Stage 1 + Stage 2 focused tests passed
- `51` core regression tests passed
- `253` other existing tests passed

### Stage 1 + Stage 2 New Tests (3 new files)

| File | Topics |
|------|--------|
| `test_stage1_full_replay_and_context.py` | TGR replay mutates live Track, unmatched calls mark_lost only (end_frame_id unchanged), all-one gating reproduces baseline, ReplayEngine excluded from backend method, checkpoint roll preserves live fields, AssociationContext built/wired with full matrix snapshots, no_reid fallback, detection_to_update_input shape (D,), serialize/deserialize preserves association_context |
| `test_stage2_cache_and_serialization.py` | Cache atomic write .incomplete→rename, skip when complete + hashes match, rebuild stale .incomplete, max_frames limit/unlimited, manifest stores processed_frames/total_sequence_frames/truncated, payload type validation, round-trip preserves association_context, unmatched no-detection features, cache_root path responsibility, ReID dim not hardcoded, track feature shape (D,) unification, writer uses fixed root, matched/unmatched counting, empty shard skip, association_context counted in manifest |
| `test_detection_cache_compact_event_cache.py` | Per-sequence detection split offsets, mmap frame slices, compact event records without detection feature, compact states without full history/ReID feature, association matrices saved once per frame |

### Milestone 1 Tests (10 new files)

| File | Topics |
|------|--------|
| `test_cli_project_root.py` | PROJECT_ROOT pathlib resolution, mode-to-split routing |
| `test_runtime_adapter_initialization.py` | event_sink=None, FeatureBuilder once, adapter no re-init, full-mode errors |
| `test_checkpoint_normalization_loading.py` | State dict load order, scalar_dim=63, event_dim=128, policy_prototypes 5×2, norm shape (63,), reid_dim consistency |
| `test_replay_backend_calls_real_track_methods.py` | Backend calls predict/update_with_gates/mark_lost, no ReplayEngine in source |
| `test_full_replay_updates_live_track.py` | Matched+unmatched replay mutates state correctly |
| `test_all_one_replay_matches_tracktrack.py` | Gate=1.0 replay preserves full-write behavior |
| `test_checkpoint_roll_preserves_live_state.py` | Runtime plans-only, no ReplayEngine in source |
| `test_unmatched_replay_preserves_end_frame_id.py` | end_frame_id preserved across unmatched steps, no-detection enters TGR window |
| `test_single_warp_consumption.py` | One CMC read per frame, identity when disabled |
| `test_benefit_sign_definition.py` | B = L_skip - L_write, hard gate B≥-1e-6, sigmoid(B/tau) |

### Existing Categories (26 files)

| Category | Files | Topics |
|----------|-------|--------|
| Track gating | `test_update_all_one.py`, `test_update_all_zero.py`, `test_gate_endpoint_branches.py` | Gate=1 equivalence, gate=0 behavior, NaN/Inf rejection |
| Snapshot/restore | `test_snapshot_restore.py` | Deep copy semantics, array isolation |
| Association | `test_association_meta_equivalence.py` | return_meta correctness, shape validation |
| Features | `test_feature_dimension.py`, `test_features.py`, `test_scalar_features_not_default_zero.py` | 63-dim shape, default None, index semantics |
| KF parity | `test_nsa_numpy_parity.py` | NSA KalmanFilter vs TrackTrack KF (1e-7 tolerance) |
| Cache | `test_cache_roundtrip.py` | Write-read cycle, serialization |
| GT/Identity | `test_gt_vote_no_leakage.py`, `test_event_id_deterministic.py` | Current-frame voting isolation, deterministic IDs |
| Candidates | `test_candidate_abc.py` | A/B/C construction |
| Rollout | `test_motion_rollout.py`, `test_appearance_rollout.py`, `test_rollout_labels.py` | Benefit computation, GT/oracle separation, TGR window enumeration |
| Runtime | `test_runtime.py`, `test_online_off_equivalence.py` | Buffer ops, replay plans, off-mode identity |
| Teacher | `test_teacher_schema.py`, `test_teacher_request_cache.py`, `test_teacher_fallback.py` | Pydantic validation, cache hashing, Flash→Plus upgrade |
| Verifier | `test_verifier_abstain.py`, `test_cross_sequence_grouping.py`, `test_label_fusion.py` | Abstain logic, grouping constraints, label fusion math |

---

## Shell Scripts (User Pipeline)

| Script | Status | What It Does |
|--------|--------|-------------|
| `00_verify_environment.sh` | READY | Checks datasets, GT, pickles, CMC, output dir |
| `01_run_baseline.sh` | READY | Run baseline tracking |
| `02_cache_events.sh` | READY | Cache compact events via real tracker adapter and mmap detection cache |
| `03_validate_cache.sh` | READY | Validate compact event cache shards |
| `04_build_rollout_labels.sh` | READY | Compute rollout motion/appearance labels |
| `05_run_oracle.sh` | READY | Run Oracle IWG/Full with hard gates |
| `06_build_student_v0_data.sh` | READY | Build IWG/TGR PyTorch datasets |
| `07_train_student_v0.sh` | READY | Train IWG → freeze → train TGR |
| `08_select_teacher_events.sh` | **DISABLED** | Exits with error until V0 validated |
| `09_build_evidence_packets.sh` | **DISABLED** | Exits with error |
| `10_run_bailian_teacher.sh` | **DISABLED** | Exits with error |
| `11_verify_and_fuse_teacher.sh` | **DISABLED** | Exits with error |
| `12_build_student_v1_data.sh` | **DISABLED** | Exits with error |
| `13_train_student_v1.sh` | **DISABLED** | Exits with error |

---

## What is PLACEHOLDER / NOT READY

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Stage 1 Full-mode TGR Replay | **WIRED** | TrackTrackReplayBackend mutates live Track; unmatched preserves end_frame_id; all-one reproduces baseline |
| Stage 2 Event Cache | **WIRED** | `MOT17-04-FRCNN` all-mode 200-frame schema-v2 compact cache validated; lazy reader restores state/detection/association from shard offsets |
| Bailian Teacher API | Placeholder | Client, schema, evidence packet APIs exist but are not validated end-to-end |
| Verifier | Placeholder | Local replay, cross-event scoring, label fusion code exists but untested end-to-end |
| Student-V1 | Not ready | Dataset and training code exists, disabled until V0 is validated |
| Hard events | Not implemented | Requires Student-V1 real failure statistics |
| Oracle real tracker CLI | Partial | `OracleGateProvider` class exists but CLI integration for real tracking is not complete |

---

## Key Design Rules Enforced (Stage 1 + 2 updates)

1. **AgentGuard does not re-match**: It only gates already-accepted detections
2. **Single feature builder**: `EventFeatureBuilder.build()` used by both online and offline paths
3. **No hardcoded ReID dim**: Read from checkpoint metadata or real detections/cache, validated against weight shapes
4. **Normalization stats in checkpoint**: Online inference loads training-computed mean/std
5. **Deterministic IDs**: No UUID anywhere in the data pipeline
6. **Fail fast**: Missing TGR/checkpoint/features/track → RuntimeError in iwg/full mode
7. **Exact endpoint handling**: Gate=1.0 and gate=0.0 bypass interpolation for exact baseline equivalence
8. **Rollout separates GT and oracle**: GT boxes for loss evaluation, oracle detections for KF update
9. **Per-sample loss weighting**: Not batch-total scalar multiplied
10. **Cache atomicity**: `.incomplete/` temp dir → manifest complete=false → all shards validated → manifest complete=true → atomic rename
11. **No ReplayEngine online**: Replay backend mutates real Track via apply_cmc/predict/update_with_gates/mark_lost
12. **One FeatureBuilder init**: `runtime.init_feature_builder()` called exactly once in tracker.__init__
13. **Formal checkpoint schema**: scalar_dim=63, event_dim=128, policy_prototypes (5,2), norm shape (63,)
14. **One CMC read per frame**: `self._current_warp` stored; disabled = identity, enabled = actual warp
15. **Benefit sign unified**: B = L_skip - L_write; hard gate 1 for B ≥ -1e-6; soft sigmoid(B/tau)
16. **Unmatched replay preserves end_frame_id**: unmapped steps call `mark_lost` only, `assert end_frame_id` unchanged
17. **Capture-only mode**: `agentguard_mode=off + capture_agentguard_events=True` captures real events during association-time without changing baseline trajectories
18. **Stable detection indexing**: HIGH + LOW + NMS_DELETED_HIGH merged into one frame_detection_index pool
19. **AssociationContext wired into TrackEvent**: full cost row/col, detection_overlap_row, accepted_detection_index, num_tracks, num_detections, reid_available; matrices copied before iterative matching mutates them
20. **Cache path responsibility**: Detection cache lives under `outputs/agentguard/detection_cache/<dataset>/<split>/<sequence>`; compact event cache lives under `outputs/agentguard/event_cache/<dataset>/<split>/<sequence>`
21. **Serialization validation**: recursive `_validate_payload` ensures no custom objects in shard payloads
22. **Manifest completeness**: stores `processed_frames`, `total_sequence_frames`, `truncated`, `complete`, config/schema hashes; existing complete dir with matching hashes is skipped
23. **ReID shape unification**: `np.asarray(..., dtype=np.float32/64).reshape(-1)` at boundaries; derive reid_dim from real data, never hardcode 2048
24. **CLI split mapping**: `MODE_TO_SPLIT = {'train_custom':'train','val_custom':'val','train':'train','val':'val'}`; never index split YAML with train_custom/val_custom
25. **MOT17 detection cache split**: `scripts/agentguard/00_split_detection_cache.py` converts the monolithic detection/FastReID pickle once into per-sequence `.npy` arrays with mmap reader support
26. **No pickle in cache_events by default**: `cache_events` requires per-sequence detection cache unless `--allow-pickle-fallback` is explicitly passed
27. **Compact event cache schema v2**: frame records store `frame_id`, `frame_index`, `effective_warp` and `warp_matrix`; event records store shard/offset refs for states and association records
28. **Compact reader**: `CompactEventCacheReader` lazily restores Track states, accepted detections from mmap FastReID cache, and association row/column context without loading the full sequence
29. **Streaming validation**: `validate_cache` reads compact shards incrementally and validates detection indices through mmap detection cache
30. **Sequence enumeration**: `cache_events` reads detection-cache manifest for `all`, filters train/val by base video id, and supports detector suffix filtering
31. **Memory profile verified**: `MOT17-04-FRCNN` all-mode 5/20/50/200-frame cache run completed with 200-frame peak RSS 471.9 MB and 8048 events

---

## Build Info

- **Python**: 3.10+
- **Dependencies**: numpy, torch, scipy, lap, pydantic, trackeval
- **Tests**: 396 passing, 0 failures (`PYTHONPATH='.:3. Tracker:agentguard/src' python -m pytest agentguard/tests -q`)
