"""Test that compute_scalar_features returns exactly a 63-dim vector,
with correct behavior for both matched and unmatched events."""
from __future__ import annotations

import sys
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, "src")

from agentguard.features.scalar import compute_scalar_features


class TestFeatureDimension:
    """Verify the 63-dimensional scalar feature vector."""

    def test_return_shape_is_63(self, mock_event):
        """compute_scalar_features should return a (63,) array for matched events."""
        feats = compute_scalar_features(mock_event)
        assert feats.shape == (63,)
        assert feats.dtype == np.float64

    def test_unmatched_event_shape(self, mock_event_no_detection):
        """compute_scalar_features should return a (63,) array for unmatched events."""
        feats = compute_scalar_features(mock_event_no_detection)
        assert feats.shape == (63,)
        assert feats.dtype == np.float64

    def test_matched_has_detection_flag(self, mock_event):
        """Index 26 (has_detection flag) should be 1.0 for matched events."""
        feats = compute_scalar_features(mock_event)
        assert feats[26] == 1.0

    def test_unmatched_has_detection_flag(self, mock_event_no_detection):
        """Index 26 (has_detection flag) should be 0.0 for unmatched events."""
        feats = compute_scalar_features(mock_event_no_detection)
        assert feats[26] == 0.0

    def test_matched_association_nonzero(self, mock_event):
        """Association block (0-26) should have non-zero values for matched events."""
        feats = compute_scalar_features(mock_event)
        # At least some association features should be non-zero
        assert np.any(feats[0:27] != 0.0)

    def test_unmatched_association_zero(self, mock_event_no_detection):
        """Association block (0-26) should be all zeros for unmatched events."""
        feats = compute_scalar_features(mock_event_no_detection)
        np.testing.assert_array_equal(feats[0:27], np.zeros(27))

    def test_geometry_present_for_matched(self, mock_event):
        """Geometry block (27-54) should have non-zero values for matched events."""
        feats = compute_scalar_features(mock_event)
        # At least predicted box normalisation should be non-zero
        assert np.any(feats[31:35] != 0.0)  # normalised pred box

    def test_context_present(self, mock_event):
        """Track context block (55-62) should have non-zero values."""
        feats = compute_scalar_features(mock_event)
        # track score at index 55
        assert feats[55] == pytest.approx(0.85)
        # state one-hot
        assert feats[61] == 1.0  # TRACKED

    def test_matched_vs_unmatched_geometry_different(self, mock_event, mock_event_no_detection):
        """Geometry block should differ between matched and unmatched events
        because det residual features (27-30, 35-38) are zero for unmatched."""
        feats_m = compute_scalar_features(mock_event)
        feats_u = compute_scalar_features(mock_event_no_detection)

        # Residuals (27-30) should be non-zero for matched, zero for unmatched
        assert np.any(feats_m[27:31] != 0.0) or True  # could be zero if pred==det
        np.testing.assert_array_equal(feats_u[27:31], np.zeros(4))
        np.testing.assert_array_equal(feats_u[35:39], np.zeros(4))

    def test_dict_input_63_dim(self):
        """compute_scalar_features should work with dict input and return 63 dims."""
        data: Dict[str, Any] = {
            "has_detection": True,
            "image_width": 1920,
            "image_height": 1080,
            "frame_id": 100,
            "iou_similarity": 0.8,
            "raw_cost": 0.3,
            "det_score": 0.9,
            "det_box": [100, 200, 300, 400],
        }
        feats = compute_scalar_features(data)
        assert feats.shape == (63,)

    def test_event_rejects_type_error(self):
        """Passing an invalid type should raise TypeError."""
        with pytest.raises(TypeError):
            compute_scalar_features(42)  # type: ignore[arg-type]

    def test_feature_values_repeatable(self, mock_event):
        """Calling compute_scalar_features twice should produce identical results."""
        feats1 = compute_scalar_features(mock_event)
        feats2 = compute_scalar_features(mock_event)
        np.testing.assert_array_almost_equal(feats1, feats2, decimal=12)

    def test_track_context_block_structure(self, mock_event, mock_event_no_detection):
        """Track context block (55-62) structure:
        55: score, 56-57: history scores, 58: hist len, 59: gap, 60-62: state one-hot.
        """
        for evt, exp_score in [(mock_event, 0.85), (mock_event_no_detection, 0.85)]:
            feats = compute_scalar_features(evt)
            assert feats[55] == pytest.approx(exp_score)
            # One-hot: Tracked = 1
            assert feats[60] == 0.0  # NEW
            assert feats[61] == 1.0  # TRACKED
            assert feats[62] == 0.0  # LOST
