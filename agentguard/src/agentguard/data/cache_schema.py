from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class CacheManifest:
    """Metadata describing a single cached sequence.

    Stored as ``manifest.json`` alongside the sharded ``.pt`` files.
    """

    schema_version: int = 2
    dataset: str = ""
    split: str = ""
    sequence: str = ""
    num_frames: int = 0
    num_events: int = 0
    num_matched_events: int = 0
    num_unmatched_events: int = 0
    num_candidates: int = 0
    num_detection_records: int = 0
    num_association_contexts: int = 0
    processed_frames: int = 0
    total_sequence_frames: int = 0
    reid_dim: int = 0
    image_width: int = 1920
    image_height: int = 1080
    complete: bool = False
    truncated: bool = False
    source_commit: str = ""
    config_sha256: str = ""
    feature_schema_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheManifest":
        """Reconstruct from a dictionary (e.g. loaded from a JSON file)."""
        return cls(
            schema_version=int(data.get("schema_version", 2)),
            dataset=str(data.get("dataset", "")),
            split=str(data.get("split", "")),
            sequence=str(data.get("sequence", "")),
            num_frames=int(data.get("num_frames", 0)),
            num_events=int(data.get("num_events", 0)),
            num_matched_events=int(data.get("num_matched_events", 0)),
            num_unmatched_events=int(data.get("num_unmatched_events", 0)),
            num_candidates=int(data.get("num_candidates", 0)),
            num_detection_records=int(data.get("num_detection_records", 0)),
            num_association_contexts=int(data.get("num_association_contexts", 0)),
            processed_frames=int(data.get("processed_frames", 0)),
            total_sequence_frames=int(data.get("total_sequence_frames", 0)),
            reid_dim=int(data.get("reid_dim", 0)),
            image_width=int(data.get("image_width", 1920)),
            image_height=int(data.get("image_height", 1080)),
            complete=bool(data.get("complete", False)),
            truncated=bool(data.get("truncated", False)),
            source_commit=str(data.get("source_commit", "")),
            config_sha256=str(data.get("config_sha256", "")),
            feature_schema_sha256=str(data.get("feature_schema_sha256", "")),
        )
