"""Mainline training defaults shared by every trainer entry point.

Experiment drivers must not restate these values; they rely on the CLI defaults.
Legacy reproduction runs opt out explicitly with ``--architecture-variant
legacy`` (which also resolves the projection width back to 64).
"""
from __future__ import annotations

MAINLINE_ARCHITECTURE_VARIANT = "query_film_bilinear"
MAINLINE_APPEARANCE_PROJ_DIM = 256
MAINLINE_STAGE_FEATURES = True
MAINLINE_STAGE_CHUNK_SAMPLES = 16384

LEGACY_ARCHITECTURE_VARIANT = "legacy"
LEGACY_APPEARANCE_PROJ_DIM = 64

__all__ = [
    "MAINLINE_ARCHITECTURE_VARIANT",
    "MAINLINE_APPEARANCE_PROJ_DIM",
    "MAINLINE_STAGE_FEATURES",
    "MAINLINE_STAGE_CHUNK_SAMPLES",
    "LEGACY_ARCHITECTURE_VARIANT",
    "LEGACY_APPEARANCE_PROJ_DIM",
]
