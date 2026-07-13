from __future__ import annotations

import numpy as np

from utils.etc import summarize_trackeval_results


def _sequence(scale: float) -> dict:
    return {
        "pedestrian": {
            "HOTA": {
                "HOTA": np.asarray([scale, scale + 0.2]),
                "DetA": np.asarray([scale + 0.1]),
                "AssA": np.asarray([scale + 0.3]),
            },
            "CLEAR": {"MOTA": scale + 0.4},
            "Identity": {"IDF1": scale + 0.5},
        }
    }


def test_trackeval_summary_is_json_safe_and_sequence_scoped():
    result = {
        "MotChallenge2DBox": {
            "tracker": {
                "MOT17-02-FRCNN": _sequence(0.1),
                "MOT17-11-FRCNN": _sequence(0.2),
                "COMBINED_SEQ": _sequence(0.3),
            }
        }
    }
    summary = summarize_trackeval_results(result, "tracker")
    assert summary["combined"]["HOTA"] == 0.4
    assert set(summary["per_sequence"]) == {
        "MOT17-02-FRCNN",
        "MOT17-11-FRCNN",
    }
    assert all(
        isinstance(value, float)
        for metrics in [summary["combined"], *summary["per_sequence"].values()]
        for value in metrics.values()
    )
