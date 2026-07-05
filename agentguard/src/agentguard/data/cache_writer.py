from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch

from agentguard.data.cache_schema import CacheManifest


class EventCacheWriter:
    """Writes cached events to disk in sharded ``.pt`` files.

    Output structure per sequence::

        outputs/agentguard/cache/<dataset>/<split>/<sequence>/
        ├── manifest.json
        ├── frames.pt
        ├── events_00000.pt
        ├── events_00001.pt
        ├── candidates_00000.pt
        ├── candidates_00001.pt
        └── identity_prototypes.pt

    Parameters
    ----------
    output_dir : str
        Root directory under which the ``<dataset>/<split>/<sequence>/``
        subdirectory will be created.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    #  Path helpers
    # ------------------------------------------------------------------

    def _seq_dir(self, manifest: CacheManifest) -> str:
        return os.path.join(
            self.output_dir,
            manifest.dataset,
            manifest.split,
            manifest.sequence,
        )

    def _ensure_seq_dir(self, manifest: CacheManifest) -> str:
        d = self._seq_dir(manifest)
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    #  Public write methods
    # ------------------------------------------------------------------

    def write_manifest(self, manifest: CacheManifest) -> str:
        """Write ``manifest.json`` and return its path."""
        seq_dir = self._ensure_seq_dir(manifest)
        path = os.path.join(seq_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        return path

    def write_events(
        self,
        events: List[Any],
        shard_size: int = 10000,
        manifest: Optional[CacheManifest] = None,
    ) -> List[str]:
        """Write events to sharded ``.pt`` files.

        Events are serialised to JSON-compatible dicts before saving
        (see :func:`agentguard.contracts.serialization.serialize_events`).

        Parameters
        ----------
        events : list
            Sequence of event objects (must be picklable via ``torch.save``).
        shard_size : int
            Maximum number of events per shard.
        manifest : CacheManifest, optional
            If provided, the shard directory is derived from it.

        Returns
        -------
        list of str
            Paths of the written shard files.
        """
        from agentguard.contracts.serialization import serialize_events

        if manifest is None:
            manifest = CacheManifest()
        seq_dir = self._ensure_seq_dir(manifest)
        paths: List[str] = []
        serialized = serialize_events(events)
        for shard_idx, start in enumerate(range(0, len(serialized), shard_size)):
            chunk = serialized[start : start + shard_size]
            shard_path = os.path.join(seq_dir, f"events_{shard_idx:05d}.pt")
            torch.save(chunk, shard_path)
            paths.append(shard_path)
        return paths

    def write_candidates(
        self,
        candidates: List[Any],
        shard_size: int = 10000,
        manifest: Optional[CacheManifest] = None,
    ) -> List[str]:
        """Write candidate data to sharded ``.pt`` files.

        Parameters
        ----------
        candidates : list
            Sequence of candidate objects (must be picklable).
        shard_size : int
            Maximum number per shard.
        manifest : CacheManifest, optional
            If provided, the shard directory is derived from it.

        Returns
        -------
        list of str
            Paths of the written shard files.
        """
        if manifest is None:
            manifest = CacheManifest()
        seq_dir = self._ensure_seq_dir(manifest)
        paths: List[str] = []
        for shard_idx, start in enumerate(range(0, len(candidates), shard_size)):
            chunk = candidates[start : start + shard_size]
            shard_path = os.path.join(seq_dir, f"candidates_{shard_idx:05d}.pt")
            torch.save(chunk, shard_path)
            paths.append(shard_path)
        return paths

    def write_identity_prototypes(
        self,
        prototypes: Dict[int, Any],
        manifest: Optional[CacheManifest] = None,
    ) -> str:
        """Write identity prototypes as a single ``.pt`` file.

        Parameters
        ----------
        prototypes : dict
            Mapping ``track_id -> prototype_vector`` (ndarray or tensor).
        manifest : CacheManifest, optional
            If provided, the shard directory is derived from it.

        Returns
        -------
        str
            Path of the written file.
        """
        if manifest is None:
            manifest = CacheManifest()
        seq_dir = self._ensure_seq_dir(manifest)
        path = os.path.join(seq_dir, "identity_prototypes.pt")
        torch.save(prototypes, path)
        return path

    def write_frames(
        self,
        frame_data: Dict[int, Any],
        manifest: Optional[CacheManifest] = None,
    ) -> str:
        """Write per-frame metadata as ``frames.pt``.

        Parameters
        ----------
        frame_data : dict
            Mapping ``frame_id -> per-frame metadata dict`` (must be picklable).
        manifest : CacheManifest, optional
            If provided, the shard directory is derived from it.

        Returns
        -------
        str
            Path of the written file.
        """
        if manifest is None:
            manifest = CacheManifest()
        seq_dir = self._ensure_seq_dir(manifest)
        path = os.path.join(seq_dir, "frames.pt")
        torch.save(frame_data, path)
        return path
