"""Cosine learning-rate scheduler with linear warmup.

This module provides a single scheduler class that performs a linear warmup
followed by a cosine decay, updating the learning rate **per optimiser step**
(suitable for step-level logging and gradient-based scheduling).

Usage
-----
.. code-block:: python

    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=1000, total_steps=10000)
    for step in range(total_steps):
        loss.backward()
        optimizer.step()
        scheduler.step()          # <-- call after every optimizer.step()
"""

from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing scheduler with a linear warmup phase.

    The learning rate increases linearly from ``start_lr`` to the base LR
    (``optimizer.param_groups[0]['lr']``) over ``warmup_steps``, then decays
    following a cosine curve to ``eta_min`` over the remaining steps.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimiser.
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps (warmup + cosine).
    eta_min : float
        Minimum (final) learning rate (default 0.0).
    start_lr : float
        Initial learning rate at step 0 (default 0.0).
    last_epoch : int
        The index of the last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        start_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = max(total_steps, 1)
        self.eta_min = eta_min
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def get_lr(self) -> List[float]:
        """Compute the learning rate for the current step."""
        step = self.last_epoch + 1  # LRScheduler tracks last_epoch; step = epoch+1

        if step <= self.warmup_steps:
            # Linear warmup.
            if self.warmup_steps > 0:
                alpha = step / self.warmup_steps
                return [
                    base_lr * alpha + self.start_lr * (1.0 - alpha)
                    for base_lr in self.base_lrs
                ]
            return list(self.base_lrs)

        # Cosine decay.
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]
