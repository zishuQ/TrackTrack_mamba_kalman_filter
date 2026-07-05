from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agentguard.teacher.response_schema import TeacherResponse


class RequestCache:
    """Caches teacher API calls to avoid redundant calls.

    Request hash includes:
    - model name
    - prompt version
    - event.json SHA256
    - contact_sheet SHA256
    - appearance_gallery SHA256

    Tracks token usage and costs.

    Parameters
    ----------
    cache_dir : str
        Directory where cache records are stored.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def get_cached(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """Return cached response if it exists.

        Parameters
        ----------
        request_hash : str
            SHA256 hash identifying the request.

        Returns
        -------
        dict or None
            The cached response dict, or ``None`` if not found.
        """
        cache_file = self.cache_dir / f"{request_hash}.json"
        if cache_file.is_file():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def save(
        self,
        request_hash: str,
        event_id: str,
        model: str,
        raw_response: str,
        parsed_response: Optional[TeacherResponse],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency: float,
        retry_count: int,
        fallback_used: bool,
    ) -> None:
        """Save a request record.

        Never saves the API key.

        Parameters
        ----------
        request_hash : str
            Unique request hash.
        event_id : str
            Event identifier.
        model : str
            Model name used.
        raw_response : str
            Raw text response from the API.
        parsed_response : TeacherResponse or None
            Parsed structured response.
        prompt_tokens : int
            Number of prompt (input) tokens.
        completion_tokens : int
            Number of completion (output) tokens.
        total_tokens : int
            Total token count.
        latency : float
            API latency in seconds.
        retry_count : int
            Number of retry attempts.
        fallback_used : bool
            Whether a fallback (Plus) model was used.
        """
        record: Dict[str, Any] = {
            "request_hash": request_hash,
            "event_id": event_id,
            "model": model,
            "raw_response": raw_response,
            "parsed_response": (
                parsed_response.model_dump() if parsed_response is not None else None
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "retry_count": retry_count,
            "fallback_used": fallback_used,
        }

        self.records.append(record)

        # Write to individual cache file
        cache_file = self.cache_dir / f"{request_hash}.json"
        with open(cache_file, "w") as f:
            json.dump(record, f, indent=2, default=str)

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute cost from config prices.

        Prices are read from ``agentguard/configs/bailian.yaml``.
        Returns 0 if prices are ``null`` or missing.

        Parameters
        ----------
        input_tokens : int
            Number of input (prompt) tokens.
        output_tokens : int
            Number of output (completion) tokens.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        prices = self._load_prices()
        if prices is None:
            return 0.0

        input_price = float(prices.get("input_price_per_1k", 0.0))
        output_price = float(prices.get("output_price_per_1k", 0.0))

        cost = (input_tokens / 1000.0) * input_price + (
            output_tokens / 1000.0
        ) * output_price
        return cost

    def export_usage(self) -> None:
        """Export usage statistics.

        Writes two files inside *cache_dir*:
        - ``teacher_api_usage.json`` — full usage data.
        - ``teacher_api_usage.csv`` — flattened usage table.
        """
        # JSON export
        json_path = self.cache_dir / "teacher_api_usage.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "total_requests": len(self.records),
                    "total_prompt_tokens": sum(
                        r.get("prompt_tokens", 0) for r in self.records
                    ),
                    "total_completion_tokens": sum(
                        r.get("completion_tokens", 0) for r in self.records
                    ),
                    "total_cost": sum(
                        self.compute_cost(
                            r.get("prompt_tokens", 0),
                            r.get("completion_tokens", 0),
                        )
                        for r in self.records
                    ),
                    "records": self.records,
                },
                f,
                indent=2,
                default=str,
            )

        # CSV export
        csv_path = self.cache_dir / "teacher_api_usage.csv"
        if self.records:
            fieldnames = [
                "request_hash",
                "event_id",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency",
                "retry_count",
                "fallback_used",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.records:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})

    # ------------------------------------------------------------------
    #  Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(
        event_json: str,
        contact_sheet: str,
        appearance_gallery: str,
        model: str,
        prompt_version: int = 1,
    ) -> str:
        """Compute a request hash from the input data.

        Parameters
        ----------
        event_json : str
            Serialised event JSON (or path).
        contact_sheet : str
            Contact sheet file path (or base64 content).
        appearance_gallery : str
            Appearance gallery file path (or base64 content).
        model : str
            Model name string.
        prompt_version : int
            Version of the prompt template used.

        Returns
        -------
        str
            Hex SHA256 digest.
        """
        hasher = hashlib.sha256()

        def _update_from_file_or_str(value: str) -> None:
            """Hash file contents if *value* is a path, otherwise hash directly."""
            if os.path.isfile(value):
                with open(value, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        hasher.update(chunk)
            else:
                hasher.update(value.encode("utf-8"))

        _update_from_file_or_str(event_json)
        _update_from_file_or_str(contact_sheet)
        _update_from_file_or_str(appearance_gallery)
        hasher.update(model.encode("utf-8"))
        hasher.update(str(prompt_version).encode("utf-8"))

        return hasher.hexdigest()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prices() -> Optional[Dict[str, float]]:
        """Load pricing information from the Bailian config."""
        config_path = "agentguard/configs/bailian.yaml"
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            prices = config.get("prices")
            if prices is None:
                return None
            return prices
        except Exception:
            return None
