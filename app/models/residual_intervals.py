"""
Apply validation-set-derived prediction intervals.

These are a heuristic band: the empirical 10th/90th percentile of val-set
residuals (actual - predicted), computed **per district** so high-count
districts (e.g. Colombo/Gampaha) get a band that reflects their true error
scale instead of the fleet-wide mean. Districts without their own calibration
fall back to a global band. This is **not** a conformal prediction set or a
Bayesian credible interval. See app/weights/residual_intervals.json and the
dissertation limitations section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_STEPS = ("1", "2", "3", "4")

# Loaded structure:
#   {"global": {step: {lower_offset, upper_offset}},
#    "per_district": {rdhs_id: {step: {lower_offset, upper_offset}}}}
ResidualIntervals = dict[str, Any]


def _normalize_table(table: dict[str, Any]) -> dict[str, dict[str, float]]:
    for step in _STEPS:
        if step not in table:
            raise ValueError(f"residual_intervals table missing horizon key {step!r}")
        entry = table[step]
        if "lower_offset" not in entry or "upper_offset" not in entry:
            raise ValueError(f"horizon {step} missing lower_offset/upper_offset")
    return {
        step: {
            "lower_offset": float(table[step]["lower_offset"]),
            "upper_offset": float(table[step]["upper_offset"]),
        }
        for step in _STEPS
    }


def load_residual_intervals(path: Path) -> ResidualIntervals:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # New per-district format.
    if "per_district" in payload or "global" in payload:
        global_table = _normalize_table(payload["global"])
        per_district = {
            str(did): _normalize_table(table)
            for did, table in payload.get("per_district", {}).items()
        }
        return {"global": global_table, "per_district": per_district}

    # Legacy format: a single flat (or "intervals"-wrapped) global table.
    legacy = payload.get("intervals", payload)
    return {"global": _normalize_table(legacy), "per_district": {}}


def _table_for(intervals: ResidualIntervals, rdhs_id: int | None) -> dict[str, dict[str, float]]:
    per_district = intervals.get("per_district", {})
    if rdhs_id is not None and str(rdhs_id) in per_district:
        return per_district[str(rdhs_id)]
    return intervals["global"]


def apply_prediction_intervals(
    predictions: np.ndarray,
    intervals: ResidualIntervals,
    rdhs_id: int | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Return (predictions, lower_bounds, upper_bounds) rounded to 1 decimal.

    Uses the per-district band for ``rdhs_id`` when available, otherwise the
    global band.

    Predictions are clamped at 0 (case counts cannot be negative). After
    applying residual offsets, the band is forced to satisfy
    ``0 <= lower <= prediction <= upper`` even when a district's calibration
    table has a negative upper_offset (e.g. Nuwara Eliya horizons 3–4).
    """
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if preds.shape[0] != 4:
        raise ValueError(f"expected 4 horizon predictions, got {preds.shape[0]}")

    table = _table_for(intervals, rdhs_id)

    rounded_pred: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for h in range(4):
        step = str(h + 1)
        raw = max(0.0, float(preds[h]))
        lo = max(0.0, raw + table[step]["lower_offset"])
        hi = raw + table[step]["upper_offset"]
        if lo > raw:
            lo = raw
        if hi < raw:
            hi = raw
        pred_r = round(raw, 1)
        lo_r = round(lo, 1)
        hi_r = round(hi, 1)
        lo_r = min(lo_r, pred_r)
        hi_r = max(hi_r, pred_r)
        rounded_pred.append(pred_r)
        lower.append(lo_r)
        upper.append(hi_r)
    return rounded_pred, lower, upper
