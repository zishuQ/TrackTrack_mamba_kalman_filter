import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from agentguard.contracts.events import TrackEvent

def test_scalar_features_defaults_to_none():
    evt = TrackEvent(
        event_id="test", dataset="MOT17", sequence="seq",
        frame_id=1, track_id=1, image_width=1920, image_height=1080,
        has_detection=True
    )
    assert evt.scalar_features is None, "scalar_features should default to None, not zeros"
