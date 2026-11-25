"""Mainline default regression checks (architecture, projection, staging)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from shutrack.training.defaults import (  # noqa: E402
    LEGACY_APPEARANCE_PROJ_DIM, LEGACY_ARCHITECTURE_VARIANT,
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
    assert args.stage_features is True
    assert args.stage_chunk_samples == MAINLINE_STAGE_CHUNK_SAMPLES


def test_legacy_reproduction_opts_out_explicitly():
    args = _bare_args("--architecture-variant", LEGACY_ARCHITECTURE_VARIANT)
    variant, width = trainer.resolve_model_options(args)
    assert variant == LEGACY_ARCHITECTURE_VARIANT
    assert width == LEGACY_APPEARANCE_PROJ_DIM


def test_staging_can_only_be_disabled_for_performance_comparison():
    args = _bare_args("--no-stage-features")
    assert args.stage_features is False


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
