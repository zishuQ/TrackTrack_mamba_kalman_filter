"""Test RequestCache hash computation and cache retrieval."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, "src")

from agentguard.teacher.request_cache import RequestCache
from agentguard.teacher.response_schema import TeacherResponse


class TestRequestCacheHash:
    """Tests for RequestCache.compute_hash."""

    def test_hash_is_deterministic(self):
        """Same inputs should produce the same hash."""
        h1 = RequestCache.compute_hash(
            event_json='{"a": 1}',
            contact_sheet="contact.jpg",
            appearance_gallery="gallery.jpg",
            model="gpt-4o",
            prompt_version=1,
        )
        h2 = RequestCache.compute_hash(
            event_json='{"a": 1}',
            contact_sheet="contact.jpg",
            appearance_gallery="gallery.jpg",
            model="gpt-4o",
            prompt_version=1,
        )
        assert h1 == h2

    def test_hash_changes_with_model(self):
        """Different model names should produce different hashes."""
        h1 = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="model-a", prompt_version=1,
        )
        h2 = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="model-b", prompt_version=1,
        )
        assert h1 != h2

    def test_hash_changes_with_prompt_version(self):
        """Different prompt versions should produce different hashes."""
        h1 = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="m", prompt_version=1,
        )
        h2 = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="m", prompt_version=2,
        )
        assert h1 != h2

    def test_hash_changes_with_event_json(self):
        """Different event JSON should produce different hashes."""
        h1 = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="m", prompt_version=1,
        )
        h2 = RequestCache.compute_hash(
            event_json='{"different": true}', contact_sheet="a.jpg",
            appearance_gallery="b.jpg", model="m", prompt_version=1,
        )
        assert h1 != h2

    def test_hash_is_sha256_hex(self):
        """Hash should be a valid 64-char hex SHA256 digest."""
        h = RequestCache.compute_hash(
            event_json="{}", contact_sheet="a.jpg", appearance_gallery="b.jpg",
            model="m", prompt_version=1,
        )
        assert len(h) == 64
        int(h, 16)  # should not raise

    def test_hash_with_file_content(self):
        """When inputs are file paths, hash should use file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evt_path = os.path.join(tmpdir, "event.json")
            with open(evt_path, "w") as f:
                json.dump({"key": "value"}, f)

            h = RequestCache.compute_hash(
                event_json=evt_path,
                contact_sheet="nonexistent.jpg",  # will hash the string
                appearance_gallery="nonexistent.jpg",
                model="m",
                prompt_version=1,
            )
            assert len(h) == 64


class TestRequestCacheRetrieval:
    """Tests for RequestCache.get_cached and save."""

    @pytest.fixture
    def cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_get_cached_nonexistent(self, cache_dir):
        """Getting a non-existent hash should return None."""
        cache = RequestCache(cache_dir=cache_dir)
        result = cache.get_cached("nonexistent_hash_1234567890abcdef")
        assert result is None

    def test_save_and_retrieve(self, cache_dir):
        """Saving then retrieving should return the same data."""
        cache = RequestCache(cache_dir=cache_dir)

        request_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        cache.save(
            request_hash=request_hash,
            event_id="evt_001",
            model="gpt-4o",
            raw_response='{"event_types": ["CLEAN_OBSERVATION"]}',
            parsed_response=TeacherResponse(
                event_types=["CLEAN_OBSERVATION"],
                confidence=0.85,
            ),
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency=0.5,
            retry_count=0,
            fallback_used=False,
        )

        cached = cache.get_cached(request_hash)
        assert cached is not None
        assert cached["request_hash"] == request_hash
        assert cached["event_id"] == "evt_001"
        assert cached["model"] == "gpt-4o"
        assert cached["prompt_tokens"] == 100
        assert cached["completion_tokens"] == 50
        assert cached["total_tokens"] == 150

    def test_save_overwrites(self, cache_dir):
        """Saving the same hash twice should overwrite."""
        cache = RequestCache(cache_dir=cache_dir)
        h = "a" * 64

        cache.save(h, "evt_001", "m", "resp1", None, 10, 5, 15, 0.1, 0, False)
        cache.save(h, "evt_002", "m", "resp2", None, 20, 10, 30, 0.2, 1, True)

        cached = cache.get_cached(h)
        assert cached["event_id"] == "evt_002"
        assert cached["raw_response"] == "resp2"

    def test_parsed_response_serialization(self, cache_dir):
        """Parsed TeacherResponse should survive JSON serialization round-trip."""
        cache = RequestCache(cache_dir=cache_dir)
        h = "b" * 64

        parsed = TeacherResponse(
            event_types=["MOTION_OUTLIER", "BOX_JITTER"],
            cue_reliability={"motion": 0.3, "appearance": 0.7, "box": 0.5},
            policy_prior={
                "FULL_WRITE": 0.4, "MOTION_ONLY": 0.2,
                "APPEARANCE_ONLY": 0.2, "HOLD_BOTH": 0.1, "SOFT_CAUTION": 0.1,
            },
            request_temporal_revision=True,
            confidence=0.75,
            abstain=False,
            evidence_frames=[100, 102, 105],
        )

        cache.save(h, "evt_003", "gpt-4o", "raw", parsed, 50, 25, 75, 0.3, 0, False)
        cached = cache.get_cached(h)

        assert cached is not None
        parsed_dump = cached["parsed_response"]
        assert parsed_dump is not None
        assert parsed_dump["event_types"] == ["MOTION_OUTLIER", "BOX_JITTER"]
        assert parsed_dump["confidence"] == 0.75
        assert parsed_dump["request_temporal_revision"] is True
        assert parsed_dump["evidence_frames"] == [100, 102, 105]

    def test_cache_creates_directory(self):
        """Cache directory should be created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_subdir = os.path.join(tmpdir, "nested", "cache", "dir")
            cache = RequestCache(cache_dir=cache_subdir)
            assert os.path.isdir(cache_subdir)

    def test_records_accumulate(self, cache_dir):
        """Each save should append to internal records list."""
        cache = RequestCache(cache_dir=cache_dir)
        assert len(cache.records) == 0

        for i in range(3):
            h = f"{i:064x}"
            cache.save(h, f"evt_{i}", "m", f"resp{i}", None, 10, 5, 15, 0.1, 0, False)

        assert len(cache.records) == 3

    def test_compute_cost(self, cache_dir):
        """compute_cost should return 0 when config is unavailable."""
        cache = RequestCache(cache_dir=cache_dir)
        cost = cache.compute_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0  # No config file in test environment
