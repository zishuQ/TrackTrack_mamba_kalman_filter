from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch

from agentguard.data.cache_schema import CacheManifest


class EventCacheReader:
    """Reads cached events from disk.

    Expects the directory layout produced by :class:`CacheEventSink`::

        <cache_dir>/
        ├── manifest.json
        ├── frames_00000.pt
        ├── events_00000.pt
        ├── events_00001.pt
        ├── candidates_00000.pt   (optional stage-2)
        ├── candidates_00001.pt   (optional stage-2)
        └── identity_prototypes.pt (optional stage-2)

    Parameters
    ----------
    cache_dir : str
        Path to the sequence-level cache directory (the one containing
        ``manifest.json``).
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    #  Path helpers
    # ------------------------------------------------------------------

    def _resolve(self, filename: str) -> str:
        path = os.path.join(self.cache_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Expected cache file not found: {path}"
            )
        return path

    def _glob_shards(self, prefix: str) -> List[str]:
        """Return sorted paths of all shards matching ``<prefix>_*.pt``."""
        matching: List[str] = []
        for fname in os.listdir(self.cache_dir):
            if fname.startswith(f"{prefix}_") and fname.endswith(".pt"):
                matching.append(os.path.join(self.cache_dir, fname))
        matching.sort()
        if not matching:
            raise FileNotFoundError(
                f"No shard files found matching '{prefix}_*.pt' in {self.cache_dir}"
            )
        return matching

    def _glob_shards_optional(self, prefix: str) -> List[str]:
        """Return sorted paths of shards matching ``<prefix>_*.pt`` or empty list."""
        matching: List[str] = []
        for fname in os.listdir(self.cache_dir):
            if fname.startswith(f"{prefix}_") and fname.endswith(".pt"):
                matching.append(os.path.join(self.cache_dir, fname))
        matching.sort()
        return matching

    def _resolve_optional(self, filename: str) -> Optional[str]:
        """Return path if *filename* exists, otherwise ``None``."""
        path = os.path.join(self.cache_dir, filename)
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------
    #  Public read methods
    # ------------------------------------------------------------------

    def read_manifest(self) -> CacheManifest:
        """Load and return the ``CacheManifest`` from ``manifest.json``."""
        path = self._resolve("manifest.json")
        with open(path, "r") as f:
            data = json.load(f)
        return CacheManifest.from_dict(data)

    def read_events(self) -> List[Any]:
        """Load all events from sharded ``.pt`` files.

        Events are deserialised from JSON-compatible dicts back to
        ``TrackEvent`` objects (see
        :func:`agentguard.contracts.serialization.deserialize_events`).

        Returns
        -------
        list
            Concatenated list of all events across all shards, preserving
            the original shard ordering.
        """
        from agentguard.contracts.serialization import deserialize_events

        shard_paths = self._glob_shards("events")
        all_events: List[Any] = []
        for sp in shard_paths:
            chunk = torch.load(sp, weights_only=False)
            if isinstance(chunk, list):
                all_events.extend(chunk)
            else:
                all_events.append(chunk)
        # Deserialise raw dicts back to TrackEvent objects
        return deserialize_events(all_events)

    def read_candidates(self) -> List[Any]:
        """Load all candidates from sharded ``.pt`` files.

        Returns an empty list when no candidates shards exist (optional
        stage-2 file).

        Returns
        -------
        list
            Concatenated list of all candidates across all shards.
        """
        shard_paths = self._glob_shards_optional("candidates")
        all_candidates: List[Any] = []
        for sp in shard_paths:
            chunk = torch.load(sp, weights_only=False)
            if isinstance(chunk, list):
                all_candidates.extend(chunk)
            else:
                all_candidates.append(chunk)
        return all_candidates

    def read_identity_prototypes(self) -> Dict[int, Any]:
        """Load identity prototypes.

        Returns an empty dict when the optional stage-2 file does not exist.

        Returns
        -------
        dict
            Mapping ``track_id -> prototype_vector``.
        """
        path = self._resolve_optional("identity_prototypes.pt")
        if path is None:
            return {}
        data = torch.load(path, weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected identity_prototypes.pt to contain a dict, "
                f"got {type(data).__name__}"
            )
        return data

    def read_frames(self) -> Any:
        """Load per-frame metadata.

        Accepts both the list payload (from ``CacheEventSink``) and the
        legacy dict payload (from ``EventCacheWriter``), returning the data
        as-stored.

        Returns
        -------
        list or dict
            Per-frame metadata in the stored format.
        """
        path = self._resolve("frames_00000.pt")
        data = torch.load(path, weights_only=False)
        if not isinstance(data, (list, dict)):
            raise TypeError(
                f"Expected frames_00000.pt to contain a list or dict, "
                f"got {type(data).__name__}"
            )
        return data
