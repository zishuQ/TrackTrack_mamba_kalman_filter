from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.data.gt_reader import GTReader


class FutureOracleBuilder:
    """For each reliable-identity event, saves:

    * ``current_gt_box`` — (4,) x1y1x2y2 of the GT box at the event frame.
    * ``future_gt_boxes`` — list [1..5] of GT boxes in the next frames.
    * ``future_oracle_detections`` — list [1..5] of the best-IoU detection
      (among frozen candidates) matching the GT box at each future frame.
    * ``future_warp_matrices`` — list [1..5] of warp matrices for each
      future frame.

    The **oracle** detection is the frozen detection candidate with the
    highest IoU to the GT box at that frame.  The GT box itself is **never**
    used as an oracle.

    Parameters
    ----------
    warp_by_frame : dict
        Mapping ``frame_id -> warp_matrix`` (``np.ndarray`` of shape ``(2, 3)``).
    """

    _MAX_FUTURE_STEPS: int = 5
    _ORACLE_IOU_MIN: float = 0.3

    def __init__(self, warp_by_frame: Dict[int, np.ndarray]) -> None:
        self.warp_by_frame = warp_by_frame

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def build(
        self,
        event: TrackEvent,
        gt_reader: GTReader,
        all_frame_detections: Dict[int, List[Tuple[np.ndarray, float, int]]],
        frame_ids: List[int],
    ) -> Dict[str, Any]:
        """Build future oracle data for an event.

        Parameters
        ----------
        event : TrackEvent
            The event for which to build the oracle.
        gt_reader : GTReader
            Ground-truth reader for the current sequence.
        all_frame_detections : dict
            Maps ``frame_id -> list of (x1y1x2y2 box, score, detection_index)``.
            These are the frozen detection candidates per frame.
        frame_ids : list of int
            Sorted list of all frame IDs in the sequence (used for
            determining future frames).

        Returns
        -------
        dict
            Keys:

            * ``current_gt_box`` — (4,) ndarray or ``None``.
            * ``future_gt_boxes`` — list of (4,) ndarray (length 0-5).
            * ``future_oracle_detections`` — list of (4,) ndarray or ``None``.
            * ``future_warp_matrices`` — list of (2, 3) ndarray.
        """
        current_frame = event.frame_id
        future_frames = self._get_future_frames(current_frame, frame_ids)

        # Current GT box
        current_gt = self._find_gt_for_track(gt_reader, current_frame, event.track_id)

        future_gt_boxes: List[np.ndarray] = []
        future_oracle_detections: List[Optional[np.ndarray]] = []
        future_warp_matrices: List[np.ndarray] = []

        for ff in future_frames:
            # GT box
            gt_box = self._find_gt_for_track(gt_reader, ff, event.track_id)
            future_gt_boxes.append(gt_box)

            # Oracle detection (best IoU to GT among frozen detections)
            oracle_box = self._find_oracle_detection(
                gt_box, all_frame_detections.get(ff, [])
            )
            future_oracle_detections.append(oracle_box)

            # Warp matrix
            wm = self.warp_by_frame.get(ff, np.eye(2, 3, dtype=np.float64))
            future_warp_matrices.append(wm)

        return {
            "current_gt_box": current_gt,
            "future_gt_boxes": future_gt_boxes,
            "future_oracle_detections": future_oracle_detections,
            "future_warp_matrices": future_warp_matrices,
        }

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_future_frames(
        current_frame: int,
        frame_ids: List[int],
        max_steps: int = 5,
    ) -> List[int]:
        """Return up to *max_steps* future frame IDs after *current_frame*.

        Returns frames from the sorted *frame_ids* list that are strictly
        greater than *current_frame*, up to *max_steps* in number.
        """
        future: List[int] = []
        for fid in frame_ids:
            if fid > current_frame:
                future.append(fid)
                if len(future) >= max_steps:
                    break
        return future

    @staticmethod
    def _find_gt_for_track(
        gt_reader: GTReader,
        frame_id: int,
        track_id: int,
    ) -> Optional[np.ndarray]:
        """Return the GT box for *track_id* at *frame_id*, or ``None``."""
        entries = gt_reader.get_gt_for_frame(frame_id)
        for box, tid in entries:
            if tid == track_id:
                return box
        return None

    @staticmethod
    def _find_oracle_detection(
        gt_box: Optional[np.ndarray],
        detections: List[Tuple[np.ndarray, float, int]],
    ) -> Optional[np.ndarray]:
        """Return the detection box with the highest IoU to *gt_box*.

        Only detections with IoU >= ``_ORACLE_IOU_MIN`` are considered.
        Returns ``None`` if there is no GT box or no qualifying detection.
        """
        if gt_box is None or not detections:
            return None

        best_box: Optional[np.ndarray] = None
        best_iou = 0.0

        for det_box, score, det_idx in detections:
            iou = FutureOracleBuilder._iou(gt_box, det_box)
            if iou >= FutureOracleBuilder._ORACLE_IOU_MIN and iou > best_iou:
                best_iou = iou
                best_box = det_box

        return best_box

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Intersection-over-Union between two ``(4,)`` boxes (x1y1x2y2)."""
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union
