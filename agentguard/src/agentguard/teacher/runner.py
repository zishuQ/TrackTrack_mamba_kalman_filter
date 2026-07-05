from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from agentguard.contracts.events import TrackEvent
from agentguard.data.gt_reader import GTReader
from agentguard.teacher.bailian_client import BailianClient
from agentguard.teacher.evidence_packet import EvidencePacketBuilder
from agentguard.teacher.request_cache import RequestCache
from agentguard.teacher.response_schema import TeacherResponse


class TeacherRunner:
    """Orchestrates the full Teacher annotation pipeline.

    The pipeline:
    1. Build evidence packets for selected events.
    2. Call the Bailian API (with caching).
    3. Save teacher responses.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file.  Defaults to
        ``agentguard/configs/teacher.yaml``.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = self._load_config(config_path)
        self.client = BailianClient(
            config_path=self.config.get("bailian_config")
        )
        self.cache = RequestCache(
            cache_dir=self.config.get("cache_dir", "cache/teacher")
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(
        self,
        selected_events: List[Tuple[int, TrackEvent, str]],
        evidence_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Run the Teacher annotation pipeline.

        Parameters
        ----------
        selected_events : list of (int, TrackEvent, str)
            Events to process, as returned by
            :class:`~agentguard.teacher.event_selector.TeacherEventSelector`.
        evidence_dir : str
            Directory containing sequence images and GT data.
        output_dir : str
            Directory where outputs (evidence packets + results) are saved.

        Returns
        -------
        dict
            Summary of the run::

                {
                    "total_events": int,
                    "successful": int,
                    "failed": int,
                    "cached": int,
                    "results_dir": str,
                    "responses": list of dict,
                }
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        evidence_output = output_path / "evidence_packets"
        evidence_output.mkdir(parents=True, exist_ok=True)

        results_dir = output_path / "teacher_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        packet_builder = EvidencePacketBuilder(
            output_dir=str(evidence_output),
            image_dir=os.path.join(evidence_dir, "images"),
        )

        responses: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {
            "total_events": len(selected_events),
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "results_dir": str(results_dir),
            "responses": responses,
        }

        for priority, event, reason in selected_events:
            result = self._process_event(
                event=event,
                packet_builder=packet_builder,
                evidence_dir=evidence_dir,
            )
            responses.append(result)

            if result.get("cached", False):
                summary["cached"] += 1
            elif result.get("success", False):
                summary["successful"] += 1
            else:
                summary["failed"] += 1

        # Export results
        self._save_results(results_dir, responses)
        self.cache.export_usage()

        return summary

    # ------------------------------------------------------------------
    #  Internal processing
    # ------------------------------------------------------------------

    def _process_event(
        self,
        event: TrackEvent,
        packet_builder: EvidencePacketBuilder,
        evidence_dir: str,
    ) -> Dict[str, Any]:
        """Process a single event through the teacher pipeline."""
        result: Dict[str, Any] = {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "frame_id": event.frame_id,
            "track_id": event.track_id,
            "success": False,
            "cached": False,
            "response": None,
            "error": None,
        }

        try:
            # Build GT reader for the sequence
            gt_reader = GTReader(
                data_dir=os.path.join(evidence_dir, "gt"),
                sequence_name=event.sequence,
            )

            # Load identity prototypes (empty dict as fallback)
            identity_prototypes: Dict[int, np.ndarray] = {}
            proto_path = os.path.join(
                evidence_dir, "prototypes", f"{event.sequence}_prototypes.pt"
            )
            if os.path.isfile(proto_path):
                try:
                    import torch

                    identity_prototypes = torch.load(
                        proto_path, weights_only=False
                    )
                except Exception:
                    pass

            # Build evidence packet
            packet_path = packet_builder.build_packet(
                event=event,
                gt_reader=gt_reader,
                identity_prototypes=identity_prototypes,
            )

            # Load packet data for API call
            event_json_path = os.path.join(packet_path, "event.json")
            contact_sheet_path = os.path.join(packet_path, "contact_sheet.jpg")
            gallery_path = os.path.join(packet_path, "appearance_gallery.jpg")
            prompt_path = os.path.join(packet_path, "prompt.json")

            with open(prompt_path, "r") as f:
                prompt = json.load(f)

            # Check cache
            request_hash = self.cache.compute_hash(
                event_json=event_json_path,
                contact_sheet=contact_sheet_path,
                appearance_gallery=gallery_path,
                model=self.client.model_name,
                prompt_version=1,
            )
            cached_response = self.cache.get_cached(request_hash)
            if cached_response is not None:
                result["cached"] = True
                result["success"] = True
                result["response"] = cached_response.get("parsed_response")
                return result

            # Call API
            images = [
                p for p in [contact_sheet_path, gallery_path] if os.path.isfile(p)
            ]
            api_result = self.client.call_with_fallback(prompt, images)

            # Save to cache
            self.cache.save(
                request_hash=request_hash,
                event_id=event.event_id,
                model=api_result.get("model", self.client.model_name),
                raw_response=api_result.get("raw_response", ""),
                parsed_response=api_result.get("parsed_response"),
                prompt_tokens=api_result.get("tokens", {}).get("prompt", 0),
                completion_tokens=api_result.get("tokens", {}).get("completion", 0),
                total_tokens=api_result.get("tokens", {}).get("total", 0),
                latency=api_result.get("latency", 0.0),
                retry_count=0,
                fallback_used=api_result.get("fallback_used", False),
            )

            if api_result.get("success", False):
                result["success"] = True
                result["response"] = api_result.get("parsed_response")
            else:
                result["error"] = api_result.get("error", "Unknown error")

        except Exception as exc:
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    #  I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_results(results_dir: Path, responses: List[Dict[str, Any]]) -> None:
        """Save teacher results to disk."""
        results_path = results_dir / "teacher_responses.json"
        with open(results_path, "w") as f:
            json.dump(responses, f, indent=2, default=str)

        # Also save a summary CSV
        csv_path = results_dir / "teacher_responses.csv"
        try:
            import csv

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "event_id",
                        "sequence",
                        "frame_id",
                        "track_id",
                        "success",
                        "cached",
                        "error",
                    ]
                )
                for r in responses:
                    writer.writerow(
                        [
                            r.get("event_id", ""),
                            r.get("sequence", ""),
                            r.get("frame_id", ""),
                            r.get("track_id", ""),
                            r.get("success", False),
                            r.get("cached", False),
                            r.get("error", ""),
                        ]
                    )
        except Exception:
            pass  # CSV export is best-effort

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        """Load the teacher runner configuration."""
        if config_path is None:
            config_path = "agentguard/configs/teacher.yaml"
        if not os.path.isfile(config_path):
            return {
                "bailian_config": None,
                "cache_dir": "cache/teacher",
            }
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
