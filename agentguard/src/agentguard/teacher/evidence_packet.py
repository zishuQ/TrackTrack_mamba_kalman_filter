from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentguard.contracts.events import TrackEvent
from agentguard.data.gt_reader import GTReader


class EvidencePacketBuilder:
    """Builds evidence packets for Teacher annotation.

    Each event generates the following directory structure::

        <output_dir>/event_<event_id>/
        ├── event.json
        ├── contact_sheet.jpg
        ├── appearance_gallery.jpg
        ├── prompt.json
        └── manifest.json

    Parameters
    ----------
    output_dir : str
        Root directory where evidence packets are written.
    image_dir : str
        Directory containing the original sequence frames (images).
    """

    def __init__(self, output_dir: str, image_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.image_dir = Path(image_dir)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def build_packet(
        self,
        event: TrackEvent,
        gt_reader: GTReader,
        identity_prototypes: Dict[int, np.ndarray],
    ) -> str:
        """Build an evidence packet for a single event.

        Parameters
        ----------
        event : TrackEvent
            The event to build a packet for.
        gt_reader : GTReader
            Ground-truth reader for the current sequence.
        identity_prototypes : dict
            Mapping ``track_id -> prototype vector`` (ReID prototypes).

        Returns
        -------
        str
            Path to the created event directory.
        """
        event_dir = self.output_dir / f"event_{event.event_id}"
        event_dir.mkdir(parents=True, exist_ok=True)

        # 1. event.json
        event_data = self._build_event_json(event)
        self._write_json(event_dir / "event.json", event_data)

        # 2. contact_sheet.jpg — stub (image assembly depends on dataset format)
        contact_sheet_path = event_dir / "contact_sheet.jpg"
        self._build_contact_sheet(event, contact_sheet_path)

        # 3. appearance_gallery.jpg — stub
        gallery_path = event_dir / "appearance_gallery.jpg"
        self._build_appearance_gallery(event, identity_prototypes, gallery_path)

        # 4. prompt.json
        prompt_data = self._build_prompt_json(event, event_data)
        self._write_json(event_dir / "prompt.json", prompt_data)

        # 5. manifest.json
        manifest = self._build_manifest(event, event_data)
        self._write_json(event_dir / "manifest.json", manifest)

        return str(event_dir)

    # ------------------------------------------------------------------
    #  event.json builder
    # ------------------------------------------------------------------

    def _build_event_json(self, event: TrackEvent) -> Dict[str, Any]:
        """Build the serialisable event data dictionary.

        Includes:
        - track_age, lost_gap, detection_score, detection_source
        - iou_distance, appearance_distance, angle_distance
        - raw_cost, final_cost, candidate_margin
        - row_entropy, col_entropy
        - motion_rollout_target, appearance_rollout_target
        - future_oracle_coverage
        """
        scalar = event.scalar_features

        data: Dict[str, Any] = {
            "event_id": event.event_id,
            "dataset": event.dataset,
            "sequence": event.sequence,
            "frame_id": event.frame_id,
            "track_id": event.track_id,
            "image_width": event.image_width,
            "image_height": event.image_height,
            "has_detection": event.has_detection,
        }

        # Association-derived fields
        assoc = event.association
        if assoc is not None:
            data["iou_distance"] = assoc.iou_distance
            data["cosine_distance"] = assoc.cosine_distance
            data["angle_distance"] = assoc.angle_distance
            data["raw_cost"] = assoc.raw_cost
            data["final_cost"] = assoc.final_cost
            data["detection_source"] = int(assoc.detection_source)
        else:
            data["iou_distance"] = -1.0
            data["cosine_distance"] = -1.0
            data["angle_distance"] = -1.0
            data["raw_cost"] = -1.0
            data["final_cost"] = -1.0
            data["detection_source"] = -1

        # Detection score
        if event.has_detection and event.detection is not None:
            data["detection_score"] = event.detection.score
        else:
            data["detection_score"] = -1.0

        # Track context from scalar features (indices 55+)
        # track_score at scalar[55] (relative to block start)
        data["track_age"] = self._compute_track_age(event)
        data["lost_gap"] = self._compute_lost_gap(event)

        # Row / column stats from scalar features
        # Indices: 13=row_min, 14=row_2nd, 15=row_diff, 18=row_entropy
        #          19=col_min, 20=col_2nd, 21=col_diff, 24=col_entropy
        if scalar.size >= 25:
            row_min = float(scalar[13])
            row_2nd = float(scalar[14])
            data["candidate_margin"] = row_2nd - row_min
            data["row_entropy"] = float(scalar[18])
            data["col_entropy"] = float(scalar[24])
        else:
            data["candidate_margin"] = -1.0
            data["row_entropy"] = -1.0
            data["col_entropy"] = -1.0

        # Rollout targets (may be attached to event)
        data["motion_rollout_target"] = getattr(event, "motion_rollout_target", None)
        data["appearance_rollout_target"] = getattr(
            event, "appearance_rollout_target", None
        )

        # Future oracle coverage
        data["future_oracle_coverage"] = getattr(
            event, "future_oracle_coverage", None
        )

        return data

    # ------------------------------------------------------------------
    #  Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt_json(
        self,
        event: TrackEvent,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the structured prompt for the LLM Teacher.

        The prompt includes the event data, visual references, and
        instructions for classification.
        """
        system_prompt = (
            "You are a tracking quality assessment assistant. "
            "Analyse the provided evidence and classify the event type, "
            "rate cue reliability (motion, appearance, box), and suggest "
            "a policy distribution. Respond in JSON format."
        )

        user_prompt_parts: List[str] = [
            "## Event Information",
            f"- Sequence: {event_data.get('sequence', 'unknown')}",
            f"- Frame: {event_data.get('frame_id', -1)}",
            f"- Track ID: {event_data.get('track_id', -1)}",
            f"- Detection source: {event_data.get('detection_source', -1)}",
            f"- Detection score: {event_data.get('detection_score', -1.0):.4f}",
            f"- IoU distance: {event_data.get('iou_distance', -1.0):.4f}",
            f"- Cosine distance: {event_data.get('cosine_distance', -1.0):.4f}",
            f"- Raw cost: {event_data.get('raw_cost', -1.0):.4f}",
            f"- Final cost: {event_data.get('final_cost', -1.0):.4f}",
            f"- Candidate margin: {event_data.get('candidate_margin', -1.0):.4f}",
            f"- Row entropy: {event_data.get('row_entropy', -1.0):.4f}",
            f"- Col entropy: {event_data.get('col_entropy', -1.0):.4f}",
            "",
            "## Visual References",
            "- Contact sheet shows the track history (past frames).",
            "- Appearance gallery shows identity prototypes and detection crops.",
            "",
            "## Instructions",
            "Classify the event into one or more of these types:",
            "- CLEAN_OBSERVATION, BOX_JITTER, PARTIAL_BOX, MOTION_OUTLIER,",
            "  APPEARANCE_CONTAMINATION, HEAVY_OCCLUSION,",
            "  MOTION_APPEARANCE_CONFLICT, IDENTITY_AMBIGUITY,",
            "  LOST_REAPPEARANCE, INSUFFICIENT_EVIDENCE",
            "",
            "Provide cue reliability scores (0-1) for motion, appearance, box.",
            "Provide a 5-way policy distribution (FULL_WRITE, MOTION_ONLY, ",
            "APPEARANCE_ONLY, HOLD_BOTH, SOFT_CAUTION) that sums to 1.0.",
            "Set 'request_temporal_revision' to true if you need more context.",
            "Set 'abstain' to true if you cannot reach a confident decision.",
            "Provide a confidence score (0-1) for your assessment.",
            "List evidence frame indices that support your decision.",
        ]

        return {
            "system": system_prompt,
            "user": "\n".join(user_prompt_parts),
            "images": {
                "contact_sheet": "contact_sheet.jpg",
                "appearance_gallery": "appearance_gallery.jpg",
            },
        }

    # ------------------------------------------------------------------
    #  Contact sheet (stub)
    # ------------------------------------------------------------------

    def _build_contact_sheet(
        self,
        event: TrackEvent,
        output_path: Path,
    ) -> None:
        """Build a contact sheet image showing the track history.

        This is a stub — in production, this would composite past-frame
        crops into a grid image.
        """
        # Placeholder: create an empty file (real implementation would use
        # PIL or OpenCV to assemble crops from self.image_dir).
        if not output_path.exists():
            # Create a minimal valid JPEG using numpy if possible;
            # otherwise just touch the file.
            try:
                import cv2 as cv

                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 32
                cv.putText(
                    canvas,
                    f"Contact sheet: {event.event_id}",
                    (20, 240),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                )
                cv.imwrite(str(output_path), canvas, [cv.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                # Fallback: touch file (will be generated later)
                output_path.touch()

    # ------------------------------------------------------------------
    #  Appearance gallery (stub)
    # ------------------------------------------------------------------

    def _build_appearance_gallery(
        self,
        event: TrackEvent,
        identity_prototypes: Dict[int, np.ndarray],
        output_path: Path,
    ) -> None:
        """Build an appearance gallery showing identity prototypes and
        detection crops.

        This is a stub — in production, this would layout prototype
        visualisations and detection thumbnails.
        """
        if not output_path.exists():
            try:
                import cv2 as cv

                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 32
                n_protos = len(identity_prototypes)
                cv.putText(
                    canvas,
                    f"Appearance gallery: {event.event_id}",
                    (20, 240),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                )
                cv.putText(
                    canvas,
                    f"Prototypes: {n_protos} identities",
                    (20, 270),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                )
                cv.imwrite(str(output_path), canvas, [cv.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                output_path.touch()

    # ------------------------------------------------------------------
    #  Manifest builder
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        event: TrackEvent,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a manifest JSON for the evidence packet."""
        return {
            "schema_version": 1,
            "event_id": event.event_id,
            "dataset": event.dataset,
            "sequence": event.sequence,
            "packet_version": "1.0",
            "files": [
                "event.json",
                "contact_sheet.jpg",
                "appearance_gallery.jpg",
                "prompt.json",
            ],
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_track_age(event: TrackEvent) -> int:
        """Estimate track age from the event's pre-update state history."""
        state = event.pre_update_state
        if state is not None and state.history:
            return len(state.history)
        return 0

    @staticmethod
    def _compute_lost_gap(event: TrackEvent) -> int:
        """Compute the gap (in frames) since the track was last observed."""
        state = event.pre_update_state
        if state is not None:
            gap = event.frame_id - state.end_frame_id
            return max(0, gap)
        return 0

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write *data* as pretty-printed JSON to *path*."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
