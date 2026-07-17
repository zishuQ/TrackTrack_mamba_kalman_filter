from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class GTReader:
    """Reads ground-truth annotations from MOT-format ``gt.txt`` files.

    MOT format (per line)::

        frame_id, track_id, x1, y1, w, h, 1, 1, 1

    Only rows with ``visibility > 0`` and standard class labels are kept.

    Provides per-frame GT boxes with track IDs.

    Parameters
    ----------
    data_dir : str
        Root directory containing sequence subdirectories with ``gt/gt.txt``.
    sequence_name : str
        Name of the sequence (subdirectory under *data_dir*).
    """

    def __init__(self, data_dir: str, sequence_name: str) -> None:
        self.data_dir = data_dir
        self.sequence_name = sequence_name
        self._gt_by_frame: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        self._gt_box_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        self._load()

    # ------------------------------------------------------------------
    #  Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load ``gt.txt`` for this sequence.

        Expected at ``<data_dir>/<sequence_name>/gt/gt.txt``.
        """
        gt_path = os.path.join(self.data_dir, self.sequence_name, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"GT file not found: {gt_path}  "
                f"(expected at {{data_dir}}/{{sequence}}/gt/gt.txt)"
            )

        self._gt_by_frame.clear()
        self._gt_box_by_frame.clear()
        data = np.loadtxt(gt_path, delimiter=",", dtype=np.float64)

        if data.ndim == 1:
            # Only a single annotation line
            data = data.reshape(1, -1)

        for row in data:
            frame_id = int(row[0])
            track_id = int(row[1])
            x1 = float(row[2])
            y1 = float(row[3])
            w = float(row[4])
            h = float(row[5])
            # Row[6] is typically 1 (class), row[7] is 1 (visibility)
            # We skip rows with zero or negative width/height
            if w <= 0 or h <= 0:
                continue

            box = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float64)
            self._gt_by_frame.setdefault(frame_id, []).append((box, track_id))
            self._gt_box_by_frame.setdefault(frame_id, {})[track_id] = box

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def get_gt_for_frame(self, frame_id: int) -> List[Tuple[np.ndarray, int]]:
        """Return ground-truth annotations for a single frame.

        Parameters
        ----------
        frame_id : int
            1-based frame index (as used in the MOT gt.txt file).

        Returns
        -------
        list of (ndarray, int)
            Each element is ``(x1y1x2y2 box, gt_track_id)``.
        """
        return self._gt_by_frame.get(frame_id, [])

    def get_gt_box(self, frame_id: int, track_id: int) -> Optional[np.ndarray]:
        """Return one track box in O(1) time for a known frame."""
        return self._gt_box_by_frame.get(int(frame_id), {}).get(int(track_id))

    def get_all_frame_ids(self) -> List[int]:
        """Return sorted list of all frame IDs present in the GT."""
        return sorted(self._gt_by_frame.keys())

    def get_all_track_ids(self) -> List[int]:
        """Return sorted list of all unique ground-truth track IDs."""
        ids: set[int] = set()
        for entries in self._gt_by_frame.values():
            for _, tid in entries:
                ids.add(tid)
        return sorted(ids)

    def get_gt_track(
        self, track_id: int
    ) -> Dict[int, np.ndarray]:
        """Return all frames for a given GT track ID.

        Parameters
        ----------
        track_id : int
            Ground-truth track ID.

        Returns
        -------
        dict
            ``{frame_id: x1y1x2y2 box}`` mapping for this track.
        """
        result: Dict[int, np.ndarray] = {}
        for fid, entries in self._gt_by_frame.items():
            for box, tid in entries:
                if tid == track_id:
                    result[fid] = box
        return result
