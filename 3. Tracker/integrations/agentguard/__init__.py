"""AgentGuard integration package for TrackTrack.

This package bridges TrackTrack's internal ``Track`` / ``Tracker`` objects
with AgentGuard's data protocol.  Only code inside this directory is allowed
to know both TrackTrack internals and AgentGuard contracts.

Export
------
Top-level convenience imports:

- :class:`.AgentGuardTrackerAdapter`  — main adapter class.
- :func:`.export_track_state`         — Track → TrackStateSnapshot.
- :func:`.export_detection`           — Track → DetectionObservation.
- :func:`.export_association_pair`    — meta dict → AssociationPairFeatures.
- :func:`.build_runtime_config`       — args → AgentGuard config dict.
"""

from .adapter import AgentGuardTrackerAdapter
from .converters import (
    export_track_state,
    export_detection,
    export_association_pair,
)
from .config_bridge import build_runtime_config

__all__ = [
    "AgentGuardTrackerAdapter",
    "export_track_state",
    "export_detection",
    "export_association_pair",
    "build_runtime_config",
]
