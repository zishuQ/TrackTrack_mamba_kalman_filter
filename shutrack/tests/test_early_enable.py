"""Early-enable New-path integration with real Track and association code."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from shutrack.integration.tracktrack import ShutrackTrackerAdapter
from shutrack.runtime.adapter import ShutrackRuntime, ShutrackRuntimeConfig
from shutrack.runtime.checkpoint_loader import load_runtime_model
from shutrack.tests.conftest import write_tiny_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_DIR = REPO_ROOT / "3. Tracker"
if str(TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKER_DIR))

from trackers.track import Track, TrackCounter  # noqa: E402
from trackers.utils import iterative_assignment  # noqa: E402


def _track_args() -> SimpleNamespace:
    return SimpleNamespace(kf_type="nsa", data_path="MOT17", min_len=3)


def _detection_array(box, score: float = 0.9) -> np.ndarray:
    return np.concatenate(
        [np.asarray(box, dtype=np.float64), np.array([score, 0.0]), np.ones(8)]
    )


def _make_detection(frame_detection_index: int, box) -> Track:
    detection = Track(_track_args(), _detection_array(box))
    detection.frame_detection_index = int(frame_detection_index)
    return detection


def _adapter(tmp_path) -> tuple[ShutrackTrackerAdapter, ShutrackRuntime]:
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(
        model,
        checkpoint,
        device="cpu",
        config=ShutrackRuntimeConfig(min_observations=6),
    )
    args = SimpleNamespace(img_w=640, img_h=480, shutrack_early_enable=True)
    return ShutrackTrackerAdapter(args, "tiny", runtime=runtime), runtime


def test_early_enable_second_records_third_gates_and_reads_history(tmp_path):
    adapter, runtime = _adapter(tmp_path)
    track_args = _track_args()
    counter = TrackCounter()
    track = Track(track_args, _detection_array([100, 100, 140, 180]))
    track.initiate(1, counter)
    assert track.observation_count == 1
    assert track.state == 0  # New

    # Match 2: New-path pool association, original update, evidence recorded.
    detection2 = _make_detection(0, [102, 102, 142, 182])
    pool2 = [detection2]
    adapter.begin_frame(2, 640, 480, detection_pool=pool2)
    matches, u_tracks, u_dets, meta2 = iterative_assignment(
        [track], pool2, [], [], 0.5, 0.2, 0.4, 0.05, 2,
        no_reid=False, return_meta=True,
    )
    assert [(int(t), int(d)) for t, d in matches] == [(0, 0)]
    assert not u_tracks and not u_dets
    handled = adapter.process_new_track_match(
        track, pool2[0], 0, 0, meta2, np.eye(2, 3), 2
    )
    assert handled is True
    assert track.observation_count == 2
    assert runtime.stats.early_gate_updates == 0
    assert len(runtime.event_buffers[track.track_id]) == 1
    evidence2 = runtime.event_buffers[track.track_id][-1]

    # Match 3: first gate application, still inside the New path.
    detection3 = _make_detection(0, [104, 104, 144, 184])
    pool3 = [detection3]
    adapter.begin_frame(3, 640, 480, detection_pool=pool3)
    matches, u_tracks, u_dets, meta3 = iterative_assignment(
        [track], pool3, [], [], 0.5, 0.2, 0.4, 0.05, 3,
        no_reid=False, return_meta=True,
    )
    assert [(int(t), int(d)) for t, d in matches] == [(0, 0)]
    assert meta3["detection_indices"].tolist() == [0]
    handled = adapter.process_new_track_match(
        track, pool3[0], 0, 0, meta3, np.eye(2, 3), 3
    )
    assert handled is True
    assert runtime.stats.early_gate_updates == 1
    assert track.observation_count == 3
    assert track.state == 1  # TrackTrack confirmation rule: now Tracked
    assert len(runtime.event_buffers[track.track_id]) == 2
    assert runtime.event_buffers[track.track_id][-2] is evidence2

    # The model-visible window at match 3 has the match-2 event as history.
    window = runtime.window_for(
        track.track_id, runtime.event_buffers[track.track_id][-1]
    )
    assert window.history_mask.tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(window.history_evidence[-1], evidence2.evidence)
    assert window.current_competition.shape == (3,)

    # Candidate-track removal cleans the buffer.
    adapter.remove_track(track.track_id)
    assert track.track_id not in runtime.event_buffers


def test_early_enable_disabled_does_not_handle_new_matches(tmp_path):
    checkpoint_path = tmp_path / "best.pt"
    write_tiny_checkpoint(checkpoint_path)
    model, checkpoint = load_runtime_model(checkpoint_path, device="cpu")
    runtime = ShutrackRuntime(model, checkpoint, device="cpu")
    adapter = ShutrackTrackerAdapter(
        SimpleNamespace(img_w=640, img_h=480, shutrack_early_enable=False),
        "tiny",
        runtime=runtime,
    )
    track = Track(_track_args(), _detection_array([100, 100, 140, 180]))
    track.initiate(1, TrackCounter())
    detection = _make_detection(0, [102, 102, 142, 182])
    adapter.begin_frame(2, 640, 480, detection_pool=[detection])
    _, _, _, meta = iterative_assignment(
        [track], [detection], [], [], 0.5, 0.2, 0.4, 0.05, 2,
        no_reid=False, return_meta=True,
    )
    assert adapter.process_new_track_match(
        track, detection, 0, 0, meta, np.eye(2, 3), 2
    ) is False
    assert not runtime.event_buffers
