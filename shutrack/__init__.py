"""SHUTrack v2: Suppress Harmful Updates with HCE + RSC.

This package is intentionally self-contained.  It does not import the legacy
AgentGuard model, trainer, or runtime; the only read-only contact points with
the legacy data layer live in ``shutrack.data.legacy_adapter``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
