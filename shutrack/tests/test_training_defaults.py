"""Mainline default regression checks (architecture, projection, staging)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from shutrack.training.defaults import (  # noqa: E402
    MAINLINE_APPEARANCE_PROJ_DIM, MAINLINE_ARCHITECTURE_VARIANT,
    MAINLINE_STAGE_CHUNK_SAMPLES,
)
import train_supervision_groups as trainer  # noqa: E402


def _bare_args(*extra):
    return trainer.build_parser().parse_args(
        ["--group", "appearance_row_column", *extra])


def test_bare_launch_selects_mainline_architecture_and_staging():
    args = _bare_args()
    variant, width = trainer.resolve_model_options(args)
    assert variant == MAINLINE_ARCHITECTURE_VARIANT
    assert width == MAINLINE_APPEARANCE_PROJ_DIM
    assert trainer.resolve_stage_chunk_samples(args, 1024) == MAINLINE_STAGE_CHUNK_SAMPLES


def test_mot20_formal_recipe_stages_the_selected_sequence_in_one_chunk():
    args = _bare_args("--dataset", "MOT20", "--shards", "4",
                      "--sequence-sampling", "sqrt-size")
    rows = range(117186)
    chunk_samples = trainer.resolve_stage_chunk_samples(args, 1024)
    assert list(trainer.feature_chunks(rows, chunk_samples, enabled=True)) == [rows]

    args.stage_chunk_samples = 16384
    chunk_samples = trainer.resolve_stage_chunk_samples(args, 1024)
    assert len(list(trainer.feature_chunks(rows, chunk_samples, enabled=True))) == 8


def test_mot20_full_pool_keeps_bounded_chunks():
    args = _bare_args("--dataset", "MOT20", "--shards", "1",
                      "--sequence-sampling", "sqrt-size")
    assert trainer.resolve_stage_chunk_samples(args, 1024) == MAINLINE_STAGE_CHUNK_SAMPLES


def test_training_rejects_retired_architecture():
    with pytest.raises(SystemExit):
        _bare_args("--architecture-variant", "legacy")


@pytest.mark.parametrize("flag", ["--stage-features", "--no-stage-features", "--preload-features"])
def test_legacy_loader_flags_are_hidden_and_normalized(flag):
    args = _bare_args(flag)
    assert args.legacy_loader_flags
    assert flag not in trainer.build_parser().format_help()
    assert trainer.resolve_stage_chunk_samples(args, 1024) == MAINLINE_STAGE_CHUNK_SAMPLES


def test_explicit_dropout_overrides_init_checkpoint():
    args = _bare_args("--dropout", "0.05")
    assert args.dropout == 0.05
    assert trainer.resolve_dropout(args, {"dropout": 0.0}) == 0.05


def test_init_checkpoint_dropout_survives_when_cli_silent():
    args = _bare_args()
    assert args.dropout is None
    assert trainer.resolve_dropout(args, {"dropout": 0.1}) == 0.1


def test_scratch_dropout_defaults_to_zero():
    assert trainer.resolve_dropout(_bare_args()) == 0.0


def test_query_film_model_creates_dropout_modules_when_enabled():
    import torch
    from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model

    model = ShutrackV2Model(ShutrackV2Config(
        reid_dim=8, appearance_proj_dim=MAINLINE_APPEARANCE_PROJ_DIM,
        architecture_variant=MAINLINE_ARCHITECTURE_VARIANT, dropout=0.05,
    ))
    drops = [(name, float(module.p)) for name, module in model.named_modules()
             if isinstance(module, torch.nn.Dropout) and module.p > 0]
    assert drops
    assert all(abs(probability - 0.05) < 1e-9 for _, probability in drops)
