"""Event sink protocol and cache implementation for AgentGuard event recording."""

from typing import Protocol, runtime_checkable, Any
import numpy as np
from pathlib import Path

@runtime_checkable
class EventSink(Protocol):
    """Protocol for recording frame-level tracking events.
    
    Implementations receive events as they are generated during
    real tracker processing, not post-hoc reconstruction.
    """
    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        """Called once per frame after association is complete.
        
        Args:
            frame_record: Dict with keys:
                frame_id, image_path, image_width, image_height,
                warp_matrix, detections (list of detection dicts)
            events: List of event dicts, each with:
                event_id, frame_id, track_id, frame_start_state,
                pre_update_state, has_detection, accepted_detection_index,
                association_row, scalar_features, track_feature,
                detection_feature, target_gt_id, detection_gt_id
        """
        ...

    def on_sequence_start(self, sequence: str, reid_dim: int, image_width: int, image_height: int) -> None:
        """Called at the start of a new sequence."""
        ...

    def on_sequence_end(self) -> dict:
        """Called at the end of a sequence. Returns manifest info."""
        ...


class CacheEventSink:
    """EventSink that writes events to disk cache shards."""
    
    def __init__(self, output_dir: str, dataset: str, split: str, sequence: str, shard_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.split = split
        self.sequence = sequence
        self.shard_size = shard_size
        self._frame_records: list[dict] = []
        self._events: list[dict] = []
        self._shard_idx = 0
        self._reid_dim = 0
        self._image_width = 0
        self._image_height = 0
        self._num_frames = 0
    
    def on_sequence_start(self, sequence: str, reid_dim: int, image_width: int, image_height: int) -> None:
        self._reid_dim = reid_dim
        self._image_width = image_width
        self._image_height = image_height
        import os
        seq_dir = self.output_dir / self.dataset / self.split / sequence
        temp_dir = seq_dir.parent / f"{sequence}.incomplete"
        os.makedirs(temp_dir, exist_ok=True)
        self._temp_dir = temp_dir
    
    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        self._frame_records.append(frame_record)
        self._events.extend(events)
        self._num_frames += 1
        while len(self._events) >= self.shard_size:
            self._flush_events()
    
    def on_sequence_end(self) -> dict:
        self._flush_events()
        self._flush_frames()
        self._write_manifest()
        self._atomic_rename()
        return {
            "sequence": self.sequence,
            "num_frames": self._num_frames,
            "num_events": sum(1 for _ in self._iter_all_shards('events')),
        }
    
    def _flush_events(self):
        import torch
        chunk = list(self._events[:self.shard_size])
        self._events = self._events[self.shard_size:]
        import os
        os.makedirs(self._temp_dir, exist_ok=True)
        torch.save(chunk, self._temp_dir / f"events_{self._shard_idx:05d}.pt")
        self._shard_idx += 1
    
    def _flush_frames(self):
        import torch
        import os
        os.makedirs(self._temp_dir, exist_ok=True)
        torch.save(self._frame_records, self._temp_dir / "frames_00000.pt")
    
    def _write_manifest(self):
        import json
        manifest = {
            "schema_version": 2,
            "dataset": self.dataset,
            "split": self.split,
            "sequence": self.sequence,
            "num_frames": self._num_frames,
            "num_events": 0,
            "num_candidates": 0,
            "reid_dim": self._reid_dim,
            "image_width": self._image_width,
            "image_height": self._image_height,
            "complete": True,
            "truncated": False,
        }
        with open(self._temp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _atomic_rename(self):
        import os
        import shutil
        final_dir = self.output_dir / self.dataset / self.split / self.sequence
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.rename(self._temp_dir, final_dir)
    
    def _iter_all_shards(self, prefix):
        import os
        for i in range(self._shard_idx):
            path = self._temp_dir / f"{prefix}_{i:05d}.pt"
            if path.exists():
                import torch
                data = torch.load(path, weights_only=False)
                for item in data:
                    yield item


class NullEventSink:
    """EventSink that discards all events (for online-only operation)."""
    def on_frame(self, frame_record: dict, events: list[dict]) -> None:
        pass
    def on_sequence_start(self, *args, **kwargs) -> None:
        pass
    def on_sequence_end(self) -> dict:
        return {}
