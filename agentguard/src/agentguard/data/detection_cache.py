from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np


DetectionView = Literal["source", "target"]


class SequenceDetectionCache:
    """Memory-mapped per-sequence detection + FastReID cache.

    The source view stores the physical detections and features once.  The
    target view, when present, stores only global detection indices into the
    source arrays.
    """

    def __init__(self, sequence_dir: str | Path) -> None:
        self.sequence_dir = Path(sequence_dir)
        with open(self.sequence_dir / "manifest.json", "r") as f:
            self.manifest = json.load(f)

        self.frame_offsets = np.load(
            self.sequence_dir / "frame_offsets.npy",
            mmap_mode="r",
        )
        self.boxes = np.load(self.sequence_dir / "boxes.npy", mmap_mode="r")
        self.scores = np.load(self.sequence_dir / "scores.npy", mmap_mode="r")
        self.features = np.load(self.sequence_dir / "features.npy", mmap_mode="r")
        self.sources = np.load(self.sequence_dir / "sources.npy", mmap_mode="r")
        self.class_ids = np.load(self.sequence_dir / "class_ids.npy", mmap_mode="r")

        self.target_frame_offsets = None
        self.target_detection_indices = None
        target_offsets_path = self.sequence_dir / "target_frame_offsets.npy"
        target_indices_path = self.sequence_dir / "target_detection_indices.npy"
        if target_offsets_path.is_file() and target_indices_path.is_file():
            self.target_frame_offsets = np.load(target_offsets_path, mmap_mode="r")
            self.target_detection_indices = np.load(target_indices_path, mmap_mode="r")

    @property
    def reid_dim(self) -> int:
        return int(self.manifest["reid_dim"])

    @property
    def num_frames(self) -> int:
        return int(self.manifest["num_frames"])

    @property
    def num_detections(self) -> int:
        return int(self.manifest["num_detections"])

    def _slice_indices(self, frame_id: int, view: DetectionView) -> np.ndarray:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= self.num_frames:
            raise IndexError(
                f"frame_id={frame_id} outside [0, {self.num_frames}) for {self.sequence_dir}"
            )
        if view == "target" and self.target_frame_offsets is not None:
            start = int(self.target_frame_offsets[frame_id])
            end = int(self.target_frame_offsets[frame_id + 1])
            return self.target_detection_indices[start:end]

        start = int(self.frame_offsets[frame_id])
        end = int(self.frame_offsets[frame_id + 1])
        return np.arange(start, end, dtype=np.int64)

    def get_frame(self, frame_id: int, view: DetectionView = "source") -> Dict[str, Any]:
        indices = self._slice_indices(frame_id, view)
        return {
            "frame_id": int(frame_id),
            "detection_indices": indices,
            "boxes": self.boxes[indices],
            "scores": self.scores[indices],
            "features": self.features[indices],
            "sources": self.sources[indices],
            "class_ids": self.class_ids[indices],
        }

    def get_frame_array(self, frame_id: int, view: DetectionView = "source") -> np.ndarray | None:
        frame = self.get_frame(frame_id, view=view)
        n = int(frame["boxes"].shape[0])
        if n == 0:
            return None
        class_ids = frame["class_ids"].astype(np.float32, copy=False).reshape(n, 1)
        return np.concatenate(
            [
                frame["boxes"].astype(np.float32, copy=False),
                frame["scores"].astype(np.float32, copy=False).reshape(n, 1),
                class_ids,
                frame["features"].astype(np.float32, copy=False),
            ],
            axis=1,
        )

    def detection_range(self, frame_id: int, view: DetectionView = "source") -> tuple[int, int]:
        indices = self._slice_indices(frame_id, view)
        if indices.size == 0:
            return (-1, -1)
        return (int(indices[0]), int(indices[-1]) + 1)

    def close(self) -> None:
        for name in (
            "frame_offsets",
            "boxes",
            "scores",
            "features",
            "sources",
            "class_ids",
            "target_frame_offsets",
            "target_detection_indices",
        ):
            arr = getattr(self, name, None)
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()
            setattr(self, name, None)


def sequence_cache_dir(
    root: str | Path,
    dataset: str,
    split: str,
    sequence: str,
) -> Path:
    return Path(root) / dataset / split / sequence

