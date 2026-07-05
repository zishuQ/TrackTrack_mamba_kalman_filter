# AgentGuard Implementation Summary

## Commit: `ef3ba93` | Branch: `agent` | 296 tests passing (0 failures)

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
│       └── replay_backend.py               # Real TrackTrackReplayBackend
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
| `tracker.py` | 5 minimal insertion points in `update()` and `update_without_detections()`: (1) begin_frame, (2) pre-CMC snapshots, (3) post-predict snapshots, (4) gated updates for mature tracks, (5) finalize_first_stage for TGR replay. Non-mature tracks and AgentGuard-off path identical to original. |
| `run.py` | Added `--agentguard-mode off/iwg/full`, `--iwg-checkpoint`, `--tgr-checkpoint`, `--agentguard-device`. Output folder naming appends `_agentguard_iwg` or `_agentguard_full`. Fixed parser.error NameError. |

---

## Architecture: How Gating Works Online

### Mature Track Definition
A track is "mature" for AgentGuard when: `state in {Tracked, Lost} AND len(history) >= 6`. Non-mature tracks get the original `Track.update()`.

### Per-Frame Processing Order
```
begin_frame (save image dims)
→ save frame_start_state snapshots (pre-CMC) for mature tracks
→ CMC warp
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
     policy_head(5) → softmax → policy®P → g_mix
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
2. Adapter gets plans, finds live Track objects
3. TrackTrackReplayBackend: restore checkpoint → for each step:
   apply_warp → Track.predict() → update_with_gates() or mark_lost()
4. Live Track is now at the correct post-replay state
```

### Checkpoint Rolling
```
Window reaches 4 events → TGR runs →
   save current live snapshot
   → restore old checkpoint → replay oldest event (only) using replay_backend
   → save replay result as new checkpoint for next-oldest event
   → delete oldest event from window
   → restore original live snapshot (unchanged)
```

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
    "reid_dim": int,           # read from data, not hardcoded
    "scalar_dim": 63,
    "event_dim": 128,
    "policy_prototypes": [...],
    "normalization_mean": ndarray,   # saved during training
    "normalization_std": ndarray,    # loaded for online inference
    "config": {...},
}
```

### Label Construction Flow
```
1. Read cached events + GT matches
2. TrackIdentityVoteState: resolve target_gt_id per (sequence, gt_id) key
   (reliable when ≥3 votes with ≥75% majority)
3. Identity prototypes: per (seq, gt_id), score≥0.6 ∧ IoU≥0.7, top 70% cosine
4. A/B/C candidates: A=accepted, B=wrong-ID lowest cost, C=same-ID low-quality
   (NMS_del > low_score > high_score, within each: highest GT IoU)
5. FutureOracle: per event, save current_gt_box + up to 5 future gt_boxes, 
   oracle detections (best IoU match in frozen pool), and warp matrices
6. Motion rollout: write/skip branches → current+5 future frames.
   Loss uses gt_boxes (not oracle dets). Update uses oracle dets with real scores.
7. Appearance rollout: same branches, EMA with oracle features, loss vs prototype
8. Dataset-level tau: median(|B| for B≠0), clipped to [1e-3, 0.1]
9. Soft targets: sigmoid(benefit/tau) for Student training
10. Hard oracle gates: B>0→1, B<0→0, |B|≤1e-6→1 (for Oracle evaluation)
11. TGR window labels: enumerate 16 sequences per window, +B-contaminated variants
```

### Event IDs (Deterministic, no UUID)
```
Base:     {dataset}/{sequence}/{frame_id:06d}/{track_id:06d}
Candidate:{base}/A, {base}/B, {base}/C
Window:   {sequence}/{track_id}/{start_frame}-{end_frame}/{variant}
          variants: ORIGINAL, B_AT_2, B_AT_3, B_AT_2_3
```

---

## Event Cache Protocol

### Directory Structure
```
outputs/agentguard/cache/{dataset}/{split}/{sequence}/
├── manifest.json      (schema_version, complete flag, dimensions, stats)
├── frames_00000.pt    (per-frame: image_path, detections, warp_matrix)
├── events_00000.pt    (sharded: TrackEvents serialized via serialize_event)
└── identity_prototypes.pt
```

### Atomicity
1. Write to `{sequence}.incomplete/` temp directory
2. `manifest.json` starts with `"complete": false`
3. All shards written and verified
4. `manifest.json` updated to `"complete": true`
5. Atomic rename to final directory

### Schema Validation
- `schema_version` in manifest → reject on mismatch
- `reid_dim` checked → reject on mismatch
- `complete: false` → reject (incomplete cache)

---

## Test Suite

**296 tests, 0 failures** (run with `PYTHONPATH=src:'../3. Tracker' python -m pytest -q`)

### Test Categories (26 files)

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
| `02_cache_events.sh` | READY | Cache events with EventSink |
| `03_validate_cache.sh` | READY | Generate validation stats |
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

All scripts use portable Python path (`PYTHON_BIN` env var, `SCRIPT_DIR`-based resolution).

---

## What is PLACEHOLDER / NOT READY

| Subsystem | Status | Notes |
|-----------|--------|-------|
| Bailian Teacher API | Placeholder | Client, schema, evidence packet APIs exist but are not validated end-to-end |
| Verifier | Placeholder | Local replay, cross-event scoring, label fusion code exists but untested end-to-end |
| Student-V1 | Not ready | Dataset and training code exists, disabled until V0 is validated |
| Hard events | Not implemented | Requires Student-V1 real failure statistics |
| Oracle real tracker CLI | Partial | `OracleGateProvider` class exists but CLI integration for real tracking is not complete |

---

## Key Design Rules Enforced

1. **AgentGuard does not re-match**: It only gates already-accepted detections
2. **Single feature builder**: `EventFeatureBuilder.build()` used by both online and offline paths
3. **No hardcoded ReID dim**: Read from checkpoint metadata or model weight shape
4. **Normalization stats in checkpoint**: Online inference loads training-computed mean/std
5. **Deterministic IDs**: No UUID anywhere in the data pipeline
6. **Fail fast**: Missing TGR/checkpoint/features → RuntimeError, not silent zero
7. **Exact endpoint handling**: Gate=1.0 and gate=0.0 bypass interpolation for exact baseline equivalence
8. **Rollout separates GT and oracle**: GT boxes for loss evaluation, oracle detections for KF update
9. **Per-sample loss weighting**: Not batch-total scalar multiplied
10. **Cache atomicity**: Temp directory → complete flag → atomic rename

---

## Build Info

- **Python**: 3.10+
- **Dependencies**: numpy, torch, scipy, lap, pydantic, trackeval
- **Commit**: `ef3ba93` on branch `agent`
- **Remote**: `git@github.com:zishuQ/TrackTrack_mamba_kalman_filter.git`
- **Last verified**: 2026-07-05, 296 tests all passing
