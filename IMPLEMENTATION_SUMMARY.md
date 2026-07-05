# AgentGuard Implementation Status

## Current State (2026-07-05)

The codebase is under active repair. Below is the current status of each subsystem.

### Online Core
- Track gated update (`update_with_gates`): IMPLEMENTED, endpoint branches added
- IWG per-frame inference: IMPLEMENTED (needs normalization stats from checkpoint)
- TGR temporal revision: IMPLEMENTED with real live Track restore
- Checkpoint rolling: IMPLEMENTED with live state preservation
- No-detection frame handling: IMPLEMENTED

### Event Cache
- EventSink protocol: IMPLEMENTED
- CacheEventSink (atomic writes): IMPLEMENTED
- Serialization-based storage: IMPLEMENTED
- Deterministic event IDs: IMPLEMENTED (no UUID)

### GT & Identity
- GT matching (Hungarian IoU): IMPLEMENTED
- Identity voting (TrackIdentityVoteState): IMPLEMENTED
- Future Oracle (target_gt_id): IMPLEMENTED
- Identity prototypes (seq, gt_id key): IMPLEMENTED

### Candidates
- A/B/C candidate builder: IMPLEMENTED (needs frame detection pool)
- AssociationContext (full row/column): DATA CLASS EXISTS

### Rollout Labels
- Motion/appearance rollout: IMPLEMENTED (GT vs oracle separation, current frame, masks)
- TGR window labels: IMPLEMENTED
- Hard oracle gates: IMPLEMENTED
- Dataset-level tau: IMPLEMENTED

### Oracle Evaluation
- Hard gate provider: IMPLEMENTED
- Real tracker execution: PENDING (CLI integration)

### Student-V0
- Datasets: IMPLEMENTED (label alignment by event_id)
- Training: IMPLEMENTED (per-sample loss weights)
- Position embeddings: IMPLEMENTED

### Bailian Teacher
- PLACEHOLDER ONLY (disabled)

### Verifier
- PLACEHOLDER ONLY (disabled)

### Hard Events
- NOT IMPLEMENTED (by design - requires Student-V1 failure stats)

## Test Status
Tests are being reorganized. Core tests pass. New integration tests added.
