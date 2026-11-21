"""Label-semantics gates for the generator and formal scratch training."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import generate_trainval_paired as generator  # noqa: E402
import train_supervision_groups as trainer  # noqa: E402


def protocol(
    schema: int = 3,
    *,
    full_write: bool = True,
    rank_arrays: bool = True,
    column: bool = False,
    gap3: bool = False,
) -> dict:
    return {
        "schema": schema,
        "full_write": full_write,
        "anchor": False,
        "column": column,
        "gap3": gap3,
        "rank_arrays": rank_arrays,
        "teacher_sha256": None,
    }


def summary(protocol_dict: dict, *, complete: bool = True, split: str = "val") -> dict:
    return {"protocol": protocol_dict, "complete": complete, "split": split}


def test_generator_does_not_skip_old_schema_complete_shard() -> None:
    current = protocol()
    old = summary(protocol(schema=2), complete=True)
    assert generator.existing_summary_state(old, current) == "incompatible"


def test_generator_resumes_current_schema_complete_shard() -> None:
    current = protocol()
    assert generator.existing_summary_state(summary(current), current) == "complete"
    assert (
        generator.existing_summary_state(summary(current, complete=False), current)
        == "incomplete"
    )


def test_scratch_training_rejects_old_schema() -> None:
    with pytest.raises(ValueError, match="schema 3"):
        trainer.validate_replay_summary(
            summary(protocol(schema=2)), "motion_diou", True, "val", "seq"
        )


def test_scratch_training_accepts_current_schema() -> None:
    result = trainer.validate_replay_summary(
        summary(protocol()), "motion_diou", True, "val", "seq"
    )
    assert result["schema"] == 3


def test_resume_accepts_current_schema_without_full_write() -> None:
    result = trainer.validate_replay_summary(
        summary(protocol(full_write=False)), "appearance_rank", False, "val", "seq"
    )
    assert result["schema"] == 3


def test_scratch_training_rejects_split_mismatch() -> None:
    with pytest.raises(ValueError, match="split-matched"):
        trainer.validate_replay_summary(
            summary(protocol(), split="train"), "motion_diou", True, "val", "seq"
        )


def test_group_requirements_still_enforced() -> None:
    with pytest.raises(ValueError, match="rank features missing"):
        trainer.validate_replay_summary(
            summary(protocol(rank_arrays=False)), "appearance_rank", True, "val", "seq"
        )
    with pytest.raises(ValueError, match="column candidates missing"):
        trainer.validate_replay_summary(
            summary(protocol()), "appearance_row_column", True, "val", "seq"
        )
    with pytest.raises(ValueError, match="third-frame targets missing"):
        trainer.validate_replay_summary(
            summary(protocol()), "motion_gap3", True, "val", "seq"
        )
