"""Unit tests for the appearance-template counterfactual accounting."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from appearance_template_experiment import summarize_sequence  # noqa: E402


def record(frame, det, labels, oracle, online):
    return {
        "frame": frame,
        "labels": labels,
        "oracle": oracle,
        "online": online,
        "actual": {
            1: {
                "det": det,
                "identity": 1,
                "b_oracle_source": 10,
                "b_online_source": 10,
            }
        },
    }


def test_status_and_change_accounting() -> None:
    labels = [1, 2, None]
    records = [
        record(1, 0, labels, {1: 1}, {1: 1}),  # A correct, B wrong -> break
        record(2, 1, labels, {1: 0}, {1: 0}),  # A wrong, B correct -> fix
        record(3, None, labels, {1: 0}, {1: 0}),  # A unmatched, B correct -> fix_miss
        record(4, 0, labels, {}, {}),  # A correct, B unmatched -> break_miss
        record(5, 2, labels, {1: 0}, {1: 0}),  # A matched but GT unknown
        record(6, 0, labels, None, None),  # B not evaluated
        record(7, 0, labels, {1: 2}, {1: 2}),  # B matched but GT unknown
    ]
    pilot = type("FakePilot", (), {"records": records})()
    summary = summarize_sequence(pilot)
    assert summary["identity_tracks"] == 7
    assert summary["identity_frames"] == 5
    assert summary["status"]["actual"] == {
        "correct": 4,
        "wrong": 1,
        "unmatched": 1,
        "unknown": 1,
    }
    for variant in ("oracle", "online"):
        assert summary["status"][variant] == {
            "correct": 3,
            "wrong": 1,
            "unmatched": 1,
            "unknown": 1,
            "no_b": 1,
        }
        assert summary["changes"][variant] == {
            "fix": 1,
            "break": 1,
            "fix_miss": 1,
            "break_miss": 1,
            "comparable": 2,
        }
