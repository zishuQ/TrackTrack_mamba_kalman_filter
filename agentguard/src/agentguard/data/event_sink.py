"""Event sink protocol and cache implementation for AgentGuard event recording."""

from typing import Protocol, runtime_checkable, Any
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Serialization validation
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = (dict, list, str, int, float, bool, type(None), np.ndarray, np.generic)

try:
    import torch
    _ALLOWED_TYPES = _ALLOWED_TYPES + (torch.Tensor,)
except ImportError:
    pass


def _validate_payload(obj: Any, path: str = "<root>") -> None:
    """Recursively validate that *obj* contains only allowed types."""
    if obj is None:
        return
    if isinstance(obj, (str, int, float, bool, np.integer, np.floating)):
        return
    if isinstance(obj, np.ndarray):
        return
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return
    except ImportError:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int)):
                raise TypeError(
                    f"dict key at {path} is {type(k).__name__}, expected str or int"
                )
            _validate_payload(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _validate_payload(v, f"{path}[{i}]")
    else:
        raise TypeError(
            f"Disallowed type {type(obj).__name__} at {path}. "
            f"Allowed: dict, list, str, int, float, bool, None, np.ndarray, torch.Tensor"
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EventSink(Protocol):
    """Protocol for recording frame-level tracking events.

    Implementations receive events as they are generated during
    real tracker processing, not post-hoc reconstruction.
    """

    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        """Called once per frame after association is complete."""
        ...

    def on_sequence_start(
        self, sequence: str, reid_dim: int, image_width: int, image_height: int
    ) -> None:
        """Called at the start of a new sequence."""
        ...

    def on_sequence_end(self) -> dict:
        """Called at the end of a sequence. Returns manifest info."""
        ...


# ---------------------------------------------------------------------------
# CacheEventSink — writes events to disk with atomic .incomplete semantics
# ---------------------------------------------------------------------------

class CacheEventSink:
    """EventSink that writes events to disk cache shards.

    Path responsibility
    -------------------
    The ``cache_root`` is the fixed base directory. This class appends
    ``<dataset>/<split>/<sequence>/`` internally. Callers MUST NOT append
    dataset/split/sequence themselves.

    Atomicity
    ---------
    1. Write to ``<sequence>.incomplete/`` temp directory.
    2. Initial ``manifest.json`` has ``"complete": false``.
    3. All shards written and validated.
    4. Final ``manifest.json`` updated with true counts and ``"complete": true``.
    5. Atomic rename to final directory.
    6. Existing final dir with ``complete=true`` and matching schema/config hashes
       is skipped. Stale ``.incomplete`` directories are deleted and rebuilt.

    Parameters
    ----------
    cache_root : str
        Fixed base directory (e.g. ``outputs/agentguard/cache``).
        This class appends ``<dataset>/<split>/<sequence>/``.
    dataset : str
        Dataset name (e.g. ``MOT17``).
    split : str
        Split name (e.g. ``train``, ``val``).
    sequence : str
        Sequence name.
    shard_size : int
        Maximum events per shard file.
    max_frames : int
        0 = unlimited. Otherwise stop after this many frames.
    config_hash : str
        SHA256 of the cache config. Used for completeness checks.
    schema_hash : str
        SHA256 of the feature schema. Used for completeness checks.
    """

    def __init__(
        self,
        cache_root: str,
        dataset: str,
        split: str,
        sequence: str,
        shard_size: int = 10000,
        max_frames: int = 0,
        config_hash: str = "",
        schema_hash: str = "",
        source_commit: str = "",
    ):
        self.cache_root = Path(cache_root)
        self.dataset = dataset
        self.split = split
        self.sequence = sequence
        self.shard_size = shard_size
        self.max_frames = max_frames
        self.config_hash = config_hash
        self.schema_hash = schema_hash
        self.source_commit = source_commit

        self._frame_records: list[dict] = []
        self._events: list[dict] = []
        self._shard_idx = 0
        self._reid_dim: int = 0
        self._image_width: int = 0
        self._image_height: int = 0
        self._num_frames: int = 0
        self._num_matched_events: int = 0
        self._num_unmatched_events: int = 0
        self._num_detection_records: int = 0
        self._num_association_contexts: int = 0
        self._total_sequence_frames: int = 0
        self._truncated: bool = False
        self._temp_dir: Path | None = None
        self._final_dir: Path | None = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _seq_dir(self) -> Path:
        """Final sequence directory: cache_root/<dataset>/<split>/<sequence>/"""
        return self.cache_root / self.dataset / self.split / self.sequence

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def on_sequence_start(
        self,
        sequence: str,
        reid_dim: int,
        image_width: int,
        image_height: int,
    ) -> None:
        self._reid_dim = reid_dim
        self._image_width = image_width
        self._image_height = image_height

        final_dir = self._seq_dir()
        self._final_dir = final_dir
        temp_dir = final_dir.parent / f"{sequence}.incomplete"

        # Check if a complete final directory already exists
        if final_dir.exists():
            existing_manifest = final_dir / "manifest.json"
            if existing_manifest.is_file():
                try:
                    with open(existing_manifest) as f:
                        existing = json.load(f)
                    if existing.get("complete", False):
                        existing_config = existing.get("config_sha256", "")
                        existing_schema = existing.get("feature_schema_sha256", "")
                        if existing_config == self.config_hash and existing_schema == self.schema_hash:
                            self._temp_dir = None
                            return
                except (json.JSONDecodeError, KeyError):
                    pass
            # Incomplete or hash mismatch — remove and rebuild
            shutil.rmtree(final_dir)

        # Remove any stale .incomplete dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        os.makedirs(temp_dir, exist_ok=True)
        self._temp_dir = temp_dir

        # Write initial manifest with complete=false
        self._write_manifest(complete=False)

    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        if self.max_frames > 0 and self._num_frames >= self.max_frames:
            self._truncated = True
            return
        if self._temp_dir is None:
            return

        # Validate frame record and events
        _validate_payload(frame_record, "frame_record")
        for evt in events:
            _validate_payload(evt, "frame_event")

        self._frame_records.append(frame_record)
        self._events.extend(events)
        self._num_frames += 1

        # Count event types
        for evt in events:
            if evt.get("has_detection"):
                self._num_matched_events += 1
                if evt.get("association_context") is not None:
                    self._num_association_contexts += 1
            else:
                self._num_unmatched_events += 1

        # Flush shards when full (skip empty chunks)
        while len(self._events) >= self.shard_size:
            self._flush_events()

    def on_sequence_end(self) -> dict:
        if self._temp_dir is None:
            return {
                "sequence": self.sequence,
                "num_frames": self._num_frames,
                "num_events": 0,
            }

        # Preserve externally pre-set _total_sequence_frames when valid
        if self._total_sequence_frames <= 0:
            self._total_sequence_frames = self._num_frames

        # Flush remaining events (skip empty)
        self._flush_events(final=True)
        self._flush_frames()

        # Compute num_detection_records from flushed frame records
        self._num_detection_records = sum(
            len(fr.get("detections", [])) for fr in self._frame_records
        )

        # Write final manifest with true counts
        self._write_manifest(complete=True)

        # Validate shard readability before atomic rename
        self._validate_shards()

        self._atomic_rename()

        total_events = self._num_matched_events + self._num_unmatched_events
        return {
            "sequence": self.sequence,
            "num_frames": self._num_frames,
            "num_events": total_events,
            "num_matched_events": self._num_matched_events,
            "num_unmatched_events": self._num_unmatched_events,
            "num_detection_records": self._num_detection_records,
            "num_association_contexts": self._num_association_contexts,
            "reid_dim": self._reid_dim,
            "truncated": self._truncated,
        }

    # ------------------------------------------------------------------
    # Internal flush / write helpers
    # ------------------------------------------------------------------

    def _flush_events(self, final: bool = False) -> None:
        if self._temp_dir is None:
            return

        chunk_size = self.shard_size
        if final:
            chunk_size = max(len(self._events), 1)

        while len(self._events) >= chunk_size:
            chunk = list(self._events[:chunk_size])
            self._events = self._events[chunk_size:]
            if not chunk:
                break
            import torch
            torch.save(chunk, self._temp_dir / f"events_{self._shard_idx:05d}.pt")
            self._shard_idx += 1

        # Final flush: save any remaining events
        if final and self._events:
            import torch
            torch.save(list(self._events), self._temp_dir / f"events_{self._shard_idx:05d}.pt")
            self._shard_idx += 1
            self._events = []

    def _flush_frames(self) -> None:
        if self._temp_dir is None:
            return
        import torch
        os.makedirs(self._temp_dir, exist_ok=True)
        torch.save(self._frame_records, self._temp_dir / "frames_00000.pt")

    def _write_manifest(self, complete: bool) -> None:
        if self._temp_dir is None:
            return
        total_events = self._num_matched_events + self._num_unmatched_events
        manifest = {
            "schema_version": 2,
            "dataset": self.dataset,
            "split": self.split,
            "sequence": self.sequence,
            "num_frames": self._num_frames,
            "num_events": total_events,
            "num_matched_events": self._num_matched_events,
            "num_unmatched_events": self._num_unmatched_events,
            "num_candidates": 0,
            "num_detection_records": self._num_detection_records,
            "num_association_contexts": self._num_association_contexts,
            "processed_frames": self._num_frames,
            "total_sequence_frames": self._total_sequence_frames,
            "reid_dim": self._reid_dim,
            "image_width": self._image_width,
            "image_height": self._image_height,
            "complete": complete,
            "truncated": self._truncated,
            "config_sha256": self.config_hash,
            "feature_schema_sha256": self.schema_hash,
        }
        if self.source_commit:
            manifest["source_commit"] = self.source_commit
        with open(self._temp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _validate_shards(self) -> None:
        """Verify every written shard file is readable before atomic rename."""
        if self._temp_dir is None:
            return
        import torch
        temp_dir = self._temp_dir
        for fname in sorted(os.listdir(temp_dir)):
            if fname.endswith(".pt"):
                try:
                    torch.load(temp_dir / fname, weights_only=False)
                except Exception as e:
                    raise RuntimeError(
                        f"Shard {fname} in {temp_dir} is unreadable: {e}"
                    )

    def _atomic_rename(self) -> None:
        if self._temp_dir is None or self._final_dir is None:
            return
        if self._final_dir.exists():
            shutil.rmtree(self._final_dir)
        os.rename(str(self._temp_dir), str(self._final_dir))
        self._temp_dir = None


class NullEventSink:
    """EventSink that discards all events (for online-only operation)."""

    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        pass

    def on_sequence_start(self, *args, **kwargs) -> None:
        pass

    def on_sequence_end(self) -> dict:
        return {}
