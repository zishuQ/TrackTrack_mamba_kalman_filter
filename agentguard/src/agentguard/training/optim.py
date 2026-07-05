from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    """Build an AdamW optimiser for the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters will be optimised.
    lr : float
        Peak learning rate (default 3e-4).
    weight_decay : float
        Weight decay coefficient (default 1e-4).

    Returns
    -------
    AdamW
        Configured optimiser.
    """
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Build a cosine annealing scheduler with linear warmup.

    The learning rate linearly increases from 0 to the peak LR over
    ``warmup_steps``, then follows a cosine decay to 0 over the remaining
    steps.

    Parameters
    ----------
    optimizer : Optimizer
        The optimiser whose LR will be scheduled.
    num_training_steps : int
        Total number of optimiser steps in the training run.
    warmup_steps : int
        Number of linear warmup steps.

    Returns
    -------
    SequentialLR
        Combined warmup + cosine scheduler.
    """
    warmup_steps = max(warmup_steps, 0)
    if warmup_steps == 0:
        # No warmup — cosine only.
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0.0)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(num_training_steps - warmup_steps, 1),
        eta_min=0.0,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
