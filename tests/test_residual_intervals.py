"""Prediction-interval clamps: non-negative points and lower <= pred <= upper."""

from __future__ import annotations

import numpy as np

from app.models.residual_intervals import apply_prediction_intervals, load_residual_intervals
from tests.conftest import WEIGHTS_DIR


def test_negative_points_clamped_to_zero():
    intervals = {
        "global": {
            "1": {"lower_offset": -2.8, "upper_offset": 2.7},
            "2": {"lower_offset": -0.6, "upper_offset": 5.4},
            "3": {"lower_offset": -1.8, "upper_offset": 3.8},
            "4": {"lower_offset": -3.9, "upper_offset": 2.5},
        },
        "per_district": {},
    }
    preds, lower, upper = apply_prediction_intervals(
        np.array([1.5, -0.4, -0.6, -0.1]),
        intervals,
        rdhs_id=13,
    )
    assert preds == [1.5, 0.0, 0.0, 0.0]
    for pred, lo, hi in zip(preds, lower, upper):
        assert lo <= pred <= hi
        assert lo >= 0.0


def test_negative_upper_offset_does_not_invert_band():
    """Nuwara Eliya (rdhs_id 20) horizons 3–4 have negative upper_offset."""
    intervals = load_residual_intervals(WEIGHTS_DIR / "residual_intervals.json")
    preds, lower, upper = apply_prediction_intervals(
        np.array([34.6, 18.9, 24.7, 17.6]),
        intervals,
        rdhs_id=20,
    )
    assert preds == [34.6, 18.9, 24.7, 17.6]
    for pred, lo, hi in zip(preds, lower, upper):
        assert lo <= pred <= hi
        assert lo >= 0.0
