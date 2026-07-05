# Baseline Audit – AgentGuard Implementation

## 1. `run.py` Parameter Entry Points

- `--pickle_dir`: Detection + ReID feature cache directory (default: `../outputs/2. det_feat/`)
- `--output_dir`: Tracking result output directory (default: `../outputs/3. track/`)
- `--data_dir`: Dataset root directory
- `--dataset`: Dataset name (MOT17, MOT20, SportsMOT)
- `--mode`: Split mode (val, val_custom, train_custom, all, test)
- `--seed`: Random seed (default: 10000)
- `--use_post`: Enable post-processing (AFLink for DanceTrack, GBI for MOT, linear interp for SportsMOT)
- `--print-per-sequence-metrics`: Print per-sequence HOTA/MOTA/IDF1
- `--min_len`: Minimum history length for Tracked state (default: 3)
- `--min_box_area`: Minimum box area for output filtering (default: 100)
- `--max_time_lost`: Max frames before track removal (auto-set from fps * 2)
- `--penalty_p`: Low-score detection penalty (default: 0.20)
- `--penalty_q`: NMS-deleted detection penalty (default: 0.40)
- `--reduce_step`: Match threshold reduction per iteration (default: 0.05)
- `--tai_thr`: Track-aware NMS threshold (default: 0.55)
- `--no-reid`: Disable ReID cosine distance, use IoU-only
- `--kf-type`: Kalman filter variant (nsa, kf, ekf, ukf), default: nsa
- `--noise-pos-std`, `--noise-size-std`, `--drop-rate`: Noise injection for robustness testing
- `--disable-gmc`: Disable global motion compensation
- `--tracker-suffix`: Suffix for output folder name
- `--sequences`: Filter to specific sequences
- `--det_thr`, `--init_thr`, `--match_thr`: Set per-dataset/per-sequence in `set_parameters()`

## 2. Detection & ReID Cache Format

Format: Pickle dictionary with structure:
```python
{
    vid_name: {
        frame_id: numpy.ndarray (N, 2054)
        # columns 0-3: x1, y1, x2, y2
        # column 4: detection score
        # column 5: class_id
        # columns 6+: ReID feature vector (2048-dim)
    }
}
```

Two cache files per split:
- `{prefix}_0.95.pickle`: All detections at score >= 0.95
- `{prefix}_0.80.from_{prefix}_0.95.idx.pickle`: Index file mapping 0.80 threshold detections
  - NMS-deleted detections are reconstructed by comparing 0.80 and 0.95 sets via IoU < 0.97

The 0.80 detections are loaded as `detections`, 0.95 as `detections_95`.

## 3. One-Frame `Tracker.update()` Complete Call Order

```
1. self.frame_id += 1

2. Find NMS-deleted detections:
   dets_del = find_deleted_detections(dets, dets_95)

3. Convert raw arrays to Track objects:
   dets = [Track(args, d) for d in dets]
   dets_del = [Track(args, d) for d in dets_del]

4. Split detections by score:
   dets_high = [d for d in dets if d.score > args.det_thr]
   dets_low  = [d for d in dets if d.score <= args.det_thr]
   dets_del_high = [d for d in dets_del if d.score > args.det_thr]

5. Split tracks:
   tracked_lost = [t for t in self.tracks if state in (Tracked, Lost)]
   new = [t for t in self.tracks if state == New]

6. Camera Motion Compensation:
   warp_matrix = self.cmc.get_warp_matrix()     # read from GMC file
   apply_cmc(tracked_lost, warp_matrix)          # warp mean/covariance
   apply_cmc(new, warp_matrix)

7. Kalman predict:
   [t.predict() for t in tracked_lost]
   [t.predict() for t in new]

8. First-stage association (Tracked/Lost + all dets):
   dets_all = dets_high + dets_low + dets_del_high
   matches, u_tracks, u_dets = iterative_assignment(
       tracked_lost, dets_high, dets_low, dets_del_high,
       match_thr, penalty_p, penalty_q, reduce_step, frame_id,
       no_reid=...
   )

9. Update matched tracks:
   for t, d in matches:
       tracked_lost[t].update(self.frame_id, dets[d])

10. Mark unmatched tracks as lost:
    for t in u_tracks:
        tracked_lost[t].mark_lost()

11. Remaining high-score detections:
    dets_high_left = [dets[i] for i in u_dets if i < len(dets_high)]

12. Second-stage association (New tracks + remaining high detections):
    matches, u_tracks, u_dets = iterative_assignment(
        new, dets_high_left, [], [],
        match_thr, penalty_p, penalty_q, reduce_step, frame_id,
        no_reid=...
    )

13. Update matched new tracks:
    for t, d in matches:
        new[t].update(self.frame_id, dets_high_left[d])

14. Mark unmatched new tracks as removed:
    for t in u_tracks:
        new[t].mark_removed()

15. Remove lost tracks too old:
    for track in self.tracks:
        if frame_id - end_frame_id > max_time_lost:
            track.mark_removed()

16. Filter removed tracks:
    self.tracks = [t for t in self.tracks if t.state != Removed]

17. Initialize new tracks (Track-Aware NMS):
    self.init_tracks(dets_high_left[udx] for udx in u_dets)

18. Return Tracked tracks:
    return [t for t in self.tracks if t.state == Tracked]
```

## 4. CMC Call Location

`tracker.py` line 55-58:
```python
warp_matrix = self.cmc.get_warp_matrix()
if not self.disable_gmc:
    apply_cmc(tracked_lost, warp_matrix)
    apply_cmc(new, warp_matrix)
```

CMC reads one line per frame from a pre-computed GMC file.
`apply_cmc()` warps `mean` and `covariance` of all tracks using the 2x3 warp matrix.

CMC is called BEFORE Kalman predict.

## 5. Kalman Predict Call Location

`tracker.py` lines 61-62:
```python
[t.predict() for t in tracked_lost]
[t.predict() for t in new]
```

`Track.predict()` (track.py line 90-97):
- Zeroes velocity of w/h if not Tracked (DanceTrack only)
- Calls `self.kalman_filter.predict(mean, covariance)`

## 6. First Stage Tracked/Lost Association Location

`tracker.py` lines 66-70:
```python
dets = dets_high + dets_low + dets_del_high
matches, u_tracks, u_dets = iterative_assignment(
    tracked_lost, dets_high, dets_low, dets_del_high,
    args.match_thr, args.penalty_p, args.penalty_q,
    args.reduce_step, self.frame_id,
    no_reid=getattr(args, 'no_reid', False))
```

## 7. Successful Match `Track.update()` Call Location

`tracker.py` lines 73-74:
```python
for t, d in matches:
    tracked_lost[t].update(self.frame_id, dets[d])
```

## 8. Unmatched Track Processing Location

`tracker.py` lines 77-78:
```python
for t in u_tracks:
    tracked_lost[t].mark_lost()
```

## 9. Second Stage New Track Association Location

`tracker.py` lines 85-88:
```python
matches, u_tracks, u_dets = iterative_assignment(
    new, dets_high_left, [], [], args.match_thr,
    args.penalty_p, args.penalty_q, args.reduce_step, self.frame_id,
    no_reid=getattr(args, 'no_reid', False))
```

## 10. TAI (Track-Aware NMS) Location

`tracker.py` line 108 (inside `init_tracks()`):
```python
def init_tracks(self, dets):
    tracks = [t for t in self.tracks if t.state == Tracked or t.state == New]
    iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
    scores = np.array([d.score for d in dets])
    allow_indices = track_aware_nms(iou_sim, scores, len(tracks), args.tai_thr, args.init_thr)
    ...
```

## 11. Result File Save Location

`run.py` line 179-180:
```python
result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
write_results(result_filename, results)
```

Format per line: `{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1`

Output folder: `outputs/3. track/<tracker_name>/`
- `tracker_name` determined from pickle basename
- Appended with `_<kf_type>` if not NSA
- Appended with `_<suffix>` if `--tracker-suffix` set

## 12. TrackEval Call Method

`run.py` line 251-255 (inside `evaluate()`):
```python
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
res, _ = evaluator.evaluate(dataset_list, metrics_list)
```

Evaluates HOTA, DetA, AssA, MOTA, IDF1.

## 13. `Track.update()` Modified Members

In `track.py` lines 99-120:
```python
def update(self, frame_id, detection):
    # KF update -> self.mean, self.covariance
    # Feature EMA -> self.feat
    # History append -> self.history[frame_id]
    # Velocity recompute -> self.velocity
    # Box update -> self.box
    # Score update -> self.score
    # End frame -> self.end_frame_id
    # State transition -> self.state (Tracked if len(history) >= min_len, else New)
```

Full list of modified members:
1. `self.mean` – KF posterior mean (8,)
2. `self.covariance` – KF posterior covariance (8,8)
3. `self.feat` – EMA-updated ReID feature (1, D)
4. `self.history[frame_id]` – [box, score, mean, cov, feat]
5. `self.velocity` – averaged velocity over delta_t frames (4, 2)
6. `self.box` – current detection box (4,)
7. `self.score` – current detection score
8. `self.end_frame_id` – current frame ID
9. `self.state` – Tracked or New

## 14. `iterative_assignment()` Current Input and Return Values

**Input:**
```python
def iterative_assignment(tracks, dets_high, dets_low, dets_del_high,
                         match_thr, penalty_p, penalty_q,
                         reduce_step, frame_id, d_t=3, no_reid=False):
```

- `tracks`: List of Track objects
- `dets_high`: List of high-score detection Track objects
- `dets_low`: List of low-score detection Track objects
- `dets_del_high`: List of NMS-deleted high-score Track objects
- `match_thr`: Initial match threshold
- `penalty_p`: Cost penalty for low-score detections
- `penalty_q`: Cost penalty for NMS-deleted detections
- `reduce_step`: Threshold reduction per iteration
- `frame_id`: Current frame for angle distance computation
- `d_t`: Delta time for velocity computation (default: 3)
- `no_reid`: Disable ReID (default: False)

**Return:**
```python
return matches, u_tracks, u_dets
```
- `matches`: List of [track_idx, det_idx] pairs
- `u_tracks`: List of unmatched track indices
- `u_dets`: List of unmatched detection indices

## 15. Baseline Kalman Filter Type

Default: **NSA Kalman Filter** (`KalmanFilter` class in `kalman_filter.py`)
- 8-dim state: [cx, cy, w, h, vx, vy, vw, vh]
- 4-dim measurement: [cx, cy, w, h]
- NSA = Noise Scale Adaptive: measurement noise scaled by `(1 - confidence)`
- Created via `create_kalman_filter('nsa')`
- Other options: 'kf' (StandardKalmanFilter), 'ekf' (ExtendedKalmanFilter), 'ukf' (UnscentedKalmanFilter)

## 16. Residual Old Implementations Search

Search executed:
```bash
grep -RIn -E "state_guard|StateGuard|IWG|TGR|update_gated|update_with_gates|snapshot_state|replay" .
```

**Result: No matches found** (excluding .venv, outputs, __pycache__).

No residual StateGuard, IWG, TGR, or replay implementations exist in the codebase.
No old training data code exists.
No old imports need to be removed.

Mamba variant (`run_mamba.py`, `tracker_mamba.py`, `track_mamba.py`) and ALT variant
(`run_alt.py`, `alt_kalman_filter_wrapper.py`) are independent tracking implementations
that do not contain AgentGuard-related code. They are excluded from AgentGuard scope.
