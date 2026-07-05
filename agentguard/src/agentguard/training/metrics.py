from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch


class MetricsTracker:
    """Track and compute training/validation metrics over multiple batches.

    Accumulates scalar loss components and prediction statistics, then
    computes aggregate summaries (mean, accuracy, etc.) via :meth:`compute`.

    Usage
    -----
    .. code-block:: python

        tracker = MetricsTracker()
        for batch in loader:
            loss_dict = {"loss": ..., "gate_loss": ..., ...}
            preds = {"gate": ..., "policy": ...}
            targets = {"motion_target": ..., ...}
            tracker.update(loss_dict, preds, targets)
        summary = tracker.compute()
        print(summary)
        tracker.reset()
    """

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        """Clear all accumulated values."""
        self._loss_sums: Dict[str, float] = defaultdict(float)
        self._loss_counts: Dict[str, int] = defaultdict(int)
        self._gate_preds: List[torch.Tensor] = []
        self._gate_targets: List[torch.Tensor] = []
        self._policy_preds: List[torch.Tensor] = []
        self._policy_targets: List[torch.Tensor] = []
        self._valid_motion: List[torch.Tensor] = []
        self._valid_appearance: List[torch.Tensor] = []
        self._sample_weights: List[torch.Tensor] = []
        self._n_batches: int = 0
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        loss_dict: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> None:
        """Accumulate metrics for one batch.

        Parameters
        ----------
        loss_dict : dict
            Scalar loss components keyed by name (e.g. ``"loss"``,
            ``"gate_loss"``, ``"policy_loss"``).  Each value must be a
            scalar ``torch.Tensor`` (detached from graph).
        preds : dict
            Model predictions.  Expected keys:

            - ``gate``: ``(B, 2)`` tensor of predicted gates.
            - ``policy_probs``: ``(B, 5)`` tensor of policy probabilities
              (optional — tracked if present).
        targets : dict
            Ground-truth targets.  Expected keys:

            - ``motion_target`` / ``appearance_target``: ``(B,)`` float.
            - ``policy_soft_target``: ``(B, 5)`` (optional).
            - ``valid_motion`` / ``valid_appearance``: ``(B,)`` bool.
            - ``sample_weight``: ``(B,)`` float.
        """
        self._n_batches += 1

        # Accumulate losses.
        for key, value in loss_dict.items():
            detached = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().mean().item()
            self._loss_sums[key] += detached
            self._loss_counts[key] += 1

        # Store predictions and targets for epoch-level metrics.
        if "gate" in preds:
            self._gate_preds.append(preds["gate"].detach().cpu())
        if "policy_probs" in preds:
            self._policy_preds.append(preds["policy_probs"].detach().cpu())

        self._gate_targets.append(
            torch.stack(
                [
                    targets.get("motion_target", torch.zeros_like(targets.get("valid_motion", torch.empty(0)))),
                    targets.get("appearance_target", torch.zeros_like(targets.get("valid_appearance", torch.empty(0)))),
                ],
                dim=-1,
            ).cpu()
        )

        if "policy_soft_target" in targets:
            self._policy_targets.append(targets["policy_soft_target"].cpu())

        if "valid_motion" in targets:
            self._valid_motion.append(targets["valid_motion"].cpu())
        if "valid_appearance" in targets:
            self._valid_appearance.append(targets["valid_appearance"].cpu())
        if "sample_weight" in targets:
            self._sample_weights.append(targets["sample_weight"].cpu())

        self._n_samples += len(
            targets.get("sample_weight", targets.get("valid_motion", torch.empty(0)))
        )

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """Aggregate all accumulated values into summary metrics.

        Returns
        -------
        dict
            Keys:
            - ``loss``: mean total loss.
            - ``gate_loss``: mean gate BCE.
            - ``policy_loss``: mean policy KL divergence (if tracked).
            - ``brier_loss``: mean Brier score.
            - ``gate_accuracy``: fraction of gates correctly predicted
              within 0.5 threshold.
            - ``motion_mae``: mean absolute error for motion gate.
            - ``appearance_mae``: mean absolute error for appearance gate.
            - ``n_batches``: total number of batches seen.
            - ``n_samples``: total number of samples.
        """
        metrics: Dict[str, float] = {}

        # Mean losses.
        for key in self._loss_sums:
            cnt = self._loss_counts.get(key, 1)
            metrics[key] = self._loss_sums[key] / max(cnt, 1)

        # Gate accuracy and MAE.
        if self._gate_preds and self._gate_targets:
            gate_pred = torch.cat(self._gate_preds, dim=0)   # (N, 2)
            gate_tgt = torch.cat(self._gate_targets, dim=0)  # (N, 2)

            # Binary accuracy at threshold 0.5.
            gate_acc = ((gate_pred > 0.5) == (gate_tgt > 0.5)).float().mean().item()
            metrics["gate_accuracy"] = gate_acc

            # Per-component MAE.
            mae = (gate_pred - gate_tgt).abs()
            metrics["motion_mae"] = mae[:, 0].mean().item()
            metrics["appearance_mae"] = mae[:, 1].mean().item()

        # Policy metrics.
        if self._policy_preds and self._policy_targets:
            policy_pred = torch.cat(self._policy_preds, dim=0)   # (N, 5)
            policy_tgt = torch.cat(self._policy_targets, dim=0)  # (N, 5)

            kl_div = (policy_tgt * (policy_tgt.clamp(min=1e-8).log() - policy_pred.clamp(min=1e-8).log())).sum(dim=-1)
            metrics["policy_kl"] = kl_div.mean().item()

            # Policy accuracy (argmax match).
            policy_acc = (policy_pred.argmax(dim=-1) == policy_tgt.argmax(dim=-1)).float().mean().item()
            metrics["policy_accuracy"] = policy_acc

        metrics["n_batches"] = self._n_batches
        metrics["n_samples"] = self._n_samples

        return metrics

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated state for a new epoch."""
        self._reset_state()
