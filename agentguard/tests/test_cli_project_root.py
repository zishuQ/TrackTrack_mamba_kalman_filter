"""Test that PROJECT_ROOT is resolved via pathlib and asserts for key directories."""

import sys
from unittest.mock import patch

import pytest


def test_project_root_is_pathlib():
    from agentguard.cli import PROJECT_ROOT
    from pathlib import Path

    p = Path(PROJECT_ROOT)
    assert p.is_dir(), f"PROJECT_ROOT {PROJECT_ROOT} is not a directory"
    assert (p / "agentguard" / "pyproject.toml").is_file(), (
        f"agentguard/pyproject.toml missing from {PROJECT_ROOT}"
    )
    assert (p / "3. Tracker").is_dir(), f"3. Tracker missing from {PROJECT_ROOT}"


def test_project_root_in_sys_path():
    from agentguard.cli import PROJECT_ROOT
    assert PROJECT_ROOT in sys.path, f"{PROJECT_ROOT} not in sys.path"


def test_mode_to_split_mapping():
    from agentguard.cli import MODE_TO_SPLIT, _resolve_split

    assert MODE_TO_SPLIT["val"] == "val"
    assert MODE_TO_SPLIT["val_custom"] == "val"
    assert MODE_TO_SPLIT["train_custom"] == "train"
    assert MODE_TO_SPLIT["train"] == "train"
    assert MODE_TO_SPLIT["all"] == "all"
    assert MODE_TO_SPLIT["test"] == "test"
    assert _resolve_split("val") == "val"
    assert _resolve_split("train_custom") == "train"
    assert _resolve_split("val_custom") == "val"

    with pytest.raises(ValueError):
        _resolve_split("nonexistent_mode")


def test_cache_dir_uses_resolved_split():
    from agentguard.cli import _cache_dir, PROJECT_ROOT
    import os
    path = _cache_dir("MOT17", "val")
    assert "val" in path
    assert "MOT17" in path


def test_candidate_type_defaults_are_a_only():
    from agentguard.cli import _label_mode_name, _parse_candidate_types

    assert _parse_candidate_types("") == {"A"}
    assert _parse_candidate_types("A") == {"A"}
    assert _label_mode_name("all", "A") == "all_a_only"
    assert _label_mode_name("all", "A,B,C") == "all"
