"""Small scalar-only metric accumulation for SHUTrack training/runtime."""

from __future__ import annotations

from typing import Any, Mapping

import torch


class ScalarAccumulator:
    """Weighted scalar means without storing per-event tensors."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def add(self, name: str, value: torch.Tensor | float, weight: float = 1.0) -> None:
        weight = float(weight)
        if weight <= 0.0:
            return
        scalar = float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        self._sums[name] = self._sums.get(name, 0.0) + scalar * weight
        self._weights[name] = self._weights.get(name, 0.0) + weight

    def mean(self, name: str, default: float = 0.0) -> float:
        weight = self._weights.get(name, 0.0)
        return float(self._sums.get(name, 0.0) / weight) if weight > 0 else float(default)

    def summary(self) -> dict[str, float]:
        return {
            name: self.mean(name)
            for name in sorted(self._sums)
        }

    def reset(self) -> None:
        self._sums.clear()
        self._weights.clear()


def component_scalars(
    components: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in components.items():
        if torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().float().cpu())
    return result


__all__ = ["ScalarAccumulator", "component_scalars"]
