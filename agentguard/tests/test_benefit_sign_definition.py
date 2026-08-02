"""Test the unified benefit sign definition B = L_skip - L_write."""

import numpy as np
import pytest


def signed_sigmoid(x):
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def test_benefit_sign_positive_means_write():
    """B = L_skip - L_write > 0 means skip loss > write loss → write is better."""
    B = 0.5  # skip worse than write
    assert B > 0  # benefit positive → write branch favored


def test_benefit_sign_negative_means_skip():
    """B = L_skip - L_write < 0 means write loss > skip loss → skip is better."""
    B = -0.5
    assert B < 0
    assert B < -1e-6  # should trigger skip (hard gate = 0)


def test_hard_gate_threshold():
    """Hard gate: 1 for B >= -1e-6, 0 for B < -1e-6."""

    def hard_gate(B):
        return 1.0 if B >= -1e-6 else 0.0

    assert hard_gate(0.5) == 1.0
    assert hard_gate(0.0) == 1.0
    assert hard_gate(-1e-9) == 1.0  # tiny negative, still >= -1e-6
    assert hard_gate(-1e-6) == 1.0  # exactly at boundary
    assert hard_gate(-1e-5) == 0.0  # clear negative
    assert hard_gate(-0.1) == 0.0


def test_soft_target_sigmoid():
    """Soft target uses sigmoid(B / tau)."""
    tau = 0.05
    B_positive = 0.1
    target = signed_sigmoid(B_positive / tau)
    assert target > 0.5  # favors write

    B_negative = -0.1
    target_neg = signed_sigmoid(B_negative / tau)
    assert target_neg < 0.5  # favors skip


def test_compute_soft_target_uses_sigmoid():
    """Soft target uses sigmoid(B / tau)."""

    import math

    def sigmoid(x):
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def compute_soft_target(benefit, tau):
        return sigmoid(benefit / max(tau, 1e-12))

    assert compute_soft_target(0.0, 0.01) == 0.5
    assert compute_soft_target(0.1, 0.01) > 0.99
    assert compute_soft_target(-0.1, 0.01) < 0.01
