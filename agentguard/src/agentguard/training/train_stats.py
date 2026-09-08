"""Device-side training statistic aggregation for IWG RG-CMA.

Ordinary epoch stats stay on the model device and are transferred once when
an epoch is logged. Expensive diagnostics can be skipped or sampled without
changing the training step.
"""

from __future__ import annotations

from typing import Any

import torch


EXPENSIVE_COMPONENT_KEYS = frozenset(
    {
        "correction_abs_p50",
        "correction_abs_p95",
    }
)
GROUP_GRAD_NORM_SUFFIX = "_grad_norm"
CORRECTION_QUANTILE_DEFINITION = "mean_of_batch_quantiles"


def resolve_diagnostic_interval(config: dict[str, Any]) -> int:
    raw = config.get("diagnostic_interval", 0)
    if raw is None or raw == "":
        interval = 0
    else:
        try:
            interval = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"diagnostic_interval must be a non-negative integer, got {raw!r}"
            ) from exc
    if interval < 0:
        raise ValueError(
            f"diagnostic_interval must be a non-negative integer, got {raw!r}"
        )
    return interval


def resolve_log_interval_setting(config: dict[str, Any]) -> int:
    raw = config.get("log_interval", 0)
    if raw is None or raw == "":
        setting = 0
    else:
        try:
            setting = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"log_interval must be a non-negative integer, got {raw!r}"
            ) from exc
    if setting < 0:
        raise ValueError(
            f"log_interval must be a non-negative integer, got {raw!r}"
        )
    return setting


def resolve_epoch_log_interval(setting: int, n_batches: int) -> int:
    if int(setting) <= 0:
        return max(1, int(n_batches) // 5)
    return int(setting)


def should_sample_diagnostics(batch_index: int, interval: int) -> bool:
    if int(interval) <= 0:
        return False
    return int(batch_index) % int(interval) == 0


def _is_expensive_metric_key(key: str) -> bool:
    if key in EXPENSIVE_COMPONENT_KEYS:
        return True
    return key.endswith(GROUP_GRAD_NORM_SUFFIX) and key != "grad_norm"


@torch.no_grad()
def logged_scalar_values(*values: torch.Tensor) -> list[float]:
    """Transfer a small stack of scalars once for intra-epoch progress logs."""
    if not values:
        return []
    stacked = torch.stack(
        [value.detach().reshape(()) for value in values]
    ).to(dtype=torch.float64)
    return [float(item) for item in stacked.cpu().tolist()]


class TrainingEpochStats:
    """Fixed-size, detached running totals for one training epoch."""

    def __init__(
        self,
        *,
        device: torch.device,
        context_size: int,
        cross_modal_token_count: int,
        uses_full_history_cma: bool = False,
        uses_bidirectional_history_cma: bool = False,
        diagnostic_interval: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.context_size = int(context_size)
        self.cross_modal_token_count = int(cross_modal_token_count)
        self.uses_full_history_cma = bool(uses_full_history_cma)
        self.uses_bidirectional_history_cma = bool(uses_bidirectional_history_cma)
        self.diagnostic_interval = int(diagnostic_interval)
        self.samples = 0
        self.batches = 0
        self.diagnostic_samples = 0
        self.diagnostic_batches = 0
        self._sums: dict[str, torch.Tensor] = {}
        self.channel_abs_error = torch.zeros(
            2, device=self.device, dtype=torch.float64
        )
        self.channel_correct = torch.zeros(
            2, device=self.device, dtype=torch.float64
        )
        self.channel_weight = torch.zeros(
            2, device=self.device, dtype=torch.float64
        )
        self.motion_positions = torch.zeros(
            self.context_size, device=self.device, dtype=torch.float64
        )
        self.appearance_positions = torch.zeros(
            self.context_size, device=self.device, dtype=torch.float64
        )
        self.appearance_to_motion_positions = torch.zeros(
            self.context_size, device=self.device, dtype=torch.float64
        )
        self.motion_to_appearance_positions = torch.zeros(
            self.context_size, device=self.device, dtype=torch.float64
        )
        self.appearance_to_motion_6x6_matrix = torch.zeros(
            (self.context_size, self.context_size),
            device=self.device,
            dtype=torch.float64,
        )
        self.motion_to_appearance_6x6_matrix = torch.zeros(
            (self.context_size, self.context_size),
            device=self.device,
            dtype=torch.float64,
        )
        self.cross_matrix = torch.zeros(
            (self.cross_modal_token_count, self.cross_modal_token_count),
            device=self.device,
            dtype=torch.float64,
        )
        self.motion_entropy = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )
        self.appearance_entropy = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )

    def _zero(self) -> torch.Tensor:
        return torch.zeros((), device=self.device, dtype=torch.float64)

    def _add_scalar(
        self, key: str, value: torch.Tensor | float, count: int
    ) -> None:
        if not torch.is_tensor(value):
            tensor = torch.tensor(
                float(value), device=self.device, dtype=torch.float64
            )
        else:
            tensor = value.detach().to(
                device=self.device, dtype=torch.float64
            ).reshape(())
        if key not in self._sums:
            self._sums[key] = self._zero()
        self._sums[key] += tensor * int(count)

    def _accumulate_attention(
        self, outputs: dict[str, torch.Tensor], count: int
    ) -> None:
        self.motion_positions += (
            outputs["motion_attention_weights"]
            .detach()
            .double()
            .mean(dim=1)
            .sum(dim=0)
        )
        self.appearance_positions += (
            outputs["appearance_attention_weights"]
            .detach()
            .double()
            .mean(dim=1)
            .sum(dim=0)
        )
        self.motion_entropy += (
            outputs["motion_attention_entropy"].detach().double().mean()
            * int(count)
        )
        self.appearance_entropy += (
            outputs["appearance_attention_entropy"].detach().double().mean()
            * int(count)
        )
        if "appearance_to_motion_attention_weights" in outputs:
            self.appearance_to_motion_positions += (
                outputs["appearance_to_motion_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
            )
            self.motion_to_appearance_positions += (
                outputs["motion_to_appearance_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
            )
        if "appearance_to_motion_6x6_attention_weights" in outputs:
            self.appearance_to_motion_6x6_matrix += (
                outputs["appearance_to_motion_6x6_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
            )
            self.motion_to_appearance_6x6_matrix += (
                outputs["motion_to_appearance_6x6_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
            )
        if "cross_modal_attention_weights" in outputs:
            self.cross_matrix += (
                outputs["cross_modal_attention_weights"]
                .detach()
                .double()
                .mean(dim=1)
                .sum(dim=0)
            )

    @torch.no_grad()
    def update(
        self,
        *,
        components: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        count: int,
        grad_norm: torch.Tensor | float | None,
        group_grad_norms: dict[str, torch.Tensor | float] | None = None,
        include_expensive: bool = True,
    ) -> None:
        count = int(count)
        self.samples += count
        self.batches += 1
        for key, value in components.items():
            if key in EXPENSIVE_COMPONENT_KEYS:
                continue
            self._add_scalar(key, value, count)
        if grad_norm is not None:
            self._add_scalar("grad_norm", grad_norm, count)
        valid_channels = torch.stack(
            [batch["valid_motion"], batch["valid_appearance"]], dim=-1
        )
        gate_weights = (
            valid_channels.to(dtype=torch.float32)
            * batch["sample_weight"].to(dtype=torch.float32).unsqueeze(-1)
        )
        refined = outputs["refined_gate"].detach()
        target = batch["safe_gate_target"]
        gate_error = (refined - target).abs()
        gate_correct = ((refined >= 0.5) == (target >= 0.5)).to(
            dtype=torch.float32
        )
        self.channel_abs_error += (gate_error * gate_weights).sum(dim=0).double()
        self.channel_correct += (gate_correct * gate_weights).sum(dim=0).double()
        self.channel_weight += gate_weights.sum(dim=0).double()
        if not include_expensive:
            return
        self.diagnostic_samples += count
        self.diagnostic_batches += 1
        for key in EXPENSIVE_COMPONENT_KEYS:
            if key in components:
                self._add_scalar(key, components[key], count)
        if group_grad_norms:
            for group_name, group_grad_norm in group_grad_norms.items():
                self._add_scalar(
                    f"{group_name}_grad_norm", group_grad_norm, count
                )
        if "motion_attention_weights" in outputs:
            self._accumulate_attention(outputs, count)

    def buffer_fingerprint(self) -> tuple[int, tuple[int, ...]]:
        """Stable identity of the fixed-size accumulator tensors."""
        tensor_ids = tuple(
            id(tensor)
            for tensor in (
                *self._sums.values(),
                self.channel_abs_error,
                self.channel_correct,
                self.channel_weight,
                self.motion_positions,
                self.appearance_positions,
                self.motion_entropy,
                self.appearance_entropy,
            )
        )
        return len(self._sums), tensor_ids

    def retains_graph(self) -> bool:
        tracked = [
            *self._sums.values(),
            self.channel_abs_error,
            self.channel_correct,
            self.channel_weight,
            self.motion_positions,
            self.appearance_positions,
            self.motion_entropy,
            self.appearance_entropy,
        ]
        return any(tensor.grad_fn is not None for tensor in tracked)

    @torch.no_grad()
    def as_metrics(self) -> dict[str, Any]:
        samples = max(int(self.samples), 1)
        diagnostic_samples = int(self.diagnostic_samples)
        diagnostic_divisor = max(diagnostic_samples, 1)
        packed: dict[str, torch.Tensor] = dict(self._sums)
        packed["_ch_err0"] = self.channel_abs_error[0]
        packed["_ch_err1"] = self.channel_abs_error[1]
        packed["_ch_cor0"] = self.channel_correct[0]
        packed["_ch_cor1"] = self.channel_correct[1]
        packed["_ch_w0"] = self.channel_weight[0]
        packed["_ch_w1"] = self.channel_weight[1]
        packed["_motion_entropy"] = self.motion_entropy
        packed["_appearance_entropy"] = self.appearance_entropy
        keys = list(packed)
        if keys:
            stacked = torch.stack(
                [packed[key].reshape(()) for key in keys]
            ).detach()
            cpu_values = stacked.cpu().tolist()
            scalars = {
                key: float(value) for key, value in zip(keys, cpu_values)
            }
        else:
            scalars = {}

        metrics: dict[str, Any] = {
            "samples": int(self.samples),
            "n_batches": int(self.batches),
            "diagnostic_interval": int(self.diagnostic_interval),
            "diagnostic_batches": int(self.diagnostic_batches),
            "diagnostic_samples": diagnostic_samples,
            "diagnostics_available": int(self.diagnostic_batches) > 0,
            "correction_quantile_definition": CORRECTION_QUANTILE_DEFINITION,
        }
        for key, value in scalars.items():
            if key.startswith("_"):
                continue
            if _is_expensive_metric_key(key):
                continue
            metrics[key] = value / samples
        for key in EXPENSIVE_COMPONENT_KEYS:
            if int(self.diagnostic_batches) == 0 or key not in scalars:
                metrics[key] = None
            else:
                metrics[key] = scalars[key] / diagnostic_divisor
        for key, value in scalars.items():
            if _is_expensive_metric_key(key) and key not in EXPENSIVE_COMPONENT_KEYS:
                metrics[key] = (
                    None
                    if int(self.diagnostic_batches) == 0
                    else value / diagnostic_divisor
                )
        if int(self.diagnostic_batches) == 0:
            metrics["base_iwg_grad_norm"] = None
            metrics["rg_cma_grad_norm"] = None
        weight0 = scalars.get("_ch_w0", 0.0)
        weight1 = scalars.get("_ch_w1", 0.0)
        metrics["gate_accuracy"] = (
            scalars.get("_ch_cor0", 0.0) + scalars.get("_ch_cor1", 0.0)
        ) / max(weight0 + weight1, 1.0)
        metrics["motion_mae"] = scalars.get("_ch_err0", 0.0) / max(weight0, 1.0)
        metrics["appearance_mae"] = scalars.get("_ch_err1", 0.0) / max(
            weight1, 1.0
        )
        if int(self.diagnostic_batches) == 0:
            metrics["motion_attention_entropy"] = None
            metrics["appearance_attention_entropy"] = None
            metrics["motion_history_attention"] = None
            metrics["appearance_history_attention"] = None
            if not self.uses_full_history_cma:
                metrics["cross_modal_attention"] = None
            if self.uses_bidirectional_history_cma:
                metrics["appearance_to_motion_history_attention"] = None
                metrics["motion_to_appearance_history_attention"] = None
            if self.uses_full_history_cma:
                metrics["appearance_to_motion_6x6_attention"] = None
                metrics["motion_to_appearance_6x6_attention"] = None
            return metrics

        metrics["motion_attention_entropy"] = (
            scalars["_motion_entropy"] / diagnostic_divisor
        )
        metrics["appearance_attention_entropy"] = (
            scalars["_appearance_entropy"] / diagnostic_divisor
        )
        metrics["motion_history_attention"] = (
            self.motion_positions / diagnostic_divisor
        ).detach().cpu().tolist()
        metrics["appearance_history_attention"] = (
            self.appearance_positions / diagnostic_divisor
        ).detach().cpu().tolist()
        if not self.uses_full_history_cma:
            metrics["cross_modal_attention"] = (
                self.cross_matrix / diagnostic_divisor
            ).detach().cpu().tolist()
        if self.uses_bidirectional_history_cma:
            metrics["appearance_to_motion_history_attention"] = (
                self.appearance_to_motion_positions / diagnostic_divisor
            ).detach().cpu().tolist()
            metrics["motion_to_appearance_history_attention"] = (
                self.motion_to_appearance_positions / diagnostic_divisor
            ).detach().cpu().tolist()
        if self.uses_full_history_cma:
            metrics["appearance_to_motion_6x6_attention"] = (
                self.appearance_to_motion_6x6_matrix / diagnostic_divisor
            ).detach().cpu().tolist()
            metrics["motion_to_appearance_6x6_attention"] = (
                self.motion_to_appearance_6x6_matrix / diagnostic_divisor
            ).detach().cpu().tolist()
        return metrics
