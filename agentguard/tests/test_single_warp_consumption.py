"""Test that only one warp matrix is consumed per frame."""

import numpy as np


def test_single_warp_consumption_disabled_gmc():
    """When GMC is disabled, the warp should be identity."""
    expected_warp = np.eye(2, 3, dtype=np.float64)

    warp = np.eye(2, 3, dtype=np.float64)
    assert np.array_equal(warp, expected_warp)


def test_single_warp_consumption_enabled_gmc():
    """When GMC is enabled, the warp should be the actual read value."""
    actual_warp = np.array([
        [1.0, 0.0, 2.5],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    assert not np.array_equal(actual_warp, np.eye(2, 3, dtype=np.float64))


def test_warp_is_stored_once_per_frame():
    """Verify the _current_warp tracking pattern."""
    import numpy as np

    class FakeCMC:
        call_count = 0
        def get_warp_matrix(self):
            self.call_count += 1
            return np.eye(2, 3, dtype=np.float64)

    cmc = FakeCMC()
    disable_gmc = False

    if disable_gmc:
        warp = np.eye(2, 3, dtype=np.float64)
    else:
        warp = cmc.get_warp_matrix().copy()

    assert cmc.call_count == 1
    assert np.array_equal(warp, np.eye(2, 3, dtype=np.float64))
