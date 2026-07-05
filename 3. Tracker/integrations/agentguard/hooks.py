"""Empty hook stubs called from ``tracker.py``.

Each hook receives the ``AgentGuardTrackerAdapter`` instance and relevant
context.  These stubs are no-ops by default; they can be replaced at runtime
(e.g. by monkey-patching or by setting them on the adapter) without modifying
the tracker source code.
"""


def on_frame_start(frame_id, adapter):
    """Called at the very beginning of each frame, before any tracking logic.

    Parameters
    ----------
    frame_id : int
        Current frame number (1-indexed).
    adapter : AgentGuardTrackerAdapter
        The active adapter instance.
    """


def on_accepted_match(track, detection, event):
    """Called after a track-detection match has been accepted.

    Parameters
    ----------
    track : Track
        The matched track.
    detection : Track
        The matched detection.
    event : TrackEvent
        The fully populated event for this match.
    """


def on_unmatched_track(track, event):
    """Called for every track that was not matched in the current frame.

    Parameters
    ----------
    track : Track
        The unmatched track.
    event : TrackEvent
        The event built for this unmatched track.
    """


def on_after_first_stage(tracks, matches):
    """Called after first-stage association has finished.

    Parameters
    ----------
    tracks : list of Track
        All tracks considered during the first stage.
    matches : list of (int, int)
        List of (track_idx, detection_idx) pairs from first-stage matching.
    """


def on_track_removed(track):
    """Called just before a track is removed from the tracker's active list.

    Parameters
    ----------
    track : Track
        The track about to be removed.
    """
