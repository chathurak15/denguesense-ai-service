"""
Empirical coverage of the prediction band on the 2024 validation set:
global (old) band vs per-district (new) band.

Coverage = fraction of actual weekly cases that fall within
[pred + lower_offset, pred + upper_offset]. With p10/p90 offsets the nominal
target is 80%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import LSTM_MODEL_PATH, RESIDUAL_INTERVALS_PATH, TARGET_SCALER_PATH  # noqa: E402
from app.features.feature_engineering import (  # noqa: E402
    FORECAST_HORIZON,
    LOOKBACK,
    RDHS_NAMES,
    SEQ_COLS,
    STATIC_COLS,
)
from app.models.lstm_forecaster import load_lstm_model  # noqa: E402
from app.models.residual_intervals import (  # noqa: E402
    _table_for,
    load_residual_intervals,
)

VAL_CSV = PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed" / "scaled" / "val.csv"


def main() -> None:
    val = pd.read_csv(VAL_CSV, parse_dates=["week_start_date", "week_end_date"])
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    intervals = load_residual_intervals(RESIDUAL_INTERVALS_PATH)

    X_seq, X_static, X_dist, y_scaled = [], [], [], []
    for _, g in val.groupby("rdhs"):
        g = g.sort_values(["year", "week_no"]).reset_index(drop=True)
        seq, stat = g[SEQ_COLS].values, g[STATIC_COLS].values
        did, ys = g["rdhs_id"].values, g["week_cases_scaled"].values
        for i in range(len(g) - LOOKBACK - FORECAST_HORIZON + 1):
            X_seq.append(seq[i : i + LOOKBACK])
            X_static.append(stat[i + LOOKBACK - 1])
            X_dist.append(did[i + LOOKBACK - 1])
            y_scaled.append(ys[i + LOOKBACK : i + LOOKBACK + FORECAST_HORIZON])

    X_seq = np.asarray(X_seq, np.float32)
    X_static = np.asarray(X_static, np.float32)
    X_dist = np.asarray(X_dist, np.int32).reshape(-1, 1)
    y_scaled = np.asarray(y_scaled, np.float32)

    model = load_lstm_model(LSTM_MODEL_PATH)
    pred_scaled = model.predict(
        {"sequence_input": X_seq, "rdhs_id": X_dist, "static_features": X_static}, verbose=0
    )

    pred = np.zeros_like(pred_scaled, np.float64)
    y = np.zeros_like(y_scaled, np.float64)
    for h in range(FORECAST_HORIZON):
        pred[:, h] = target_scaler.inverse_transform(pred_scaled[:, h].reshape(-1, 1)).ravel()
        y[:, h] = target_scaler.inverse_transform(y_scaled[:, h].reshape(-1, 1)).ravel()

    dist = X_dist.ravel()

    def covered(band_kind: str) -> np.ndarray:
        hit = np.zeros_like(y, dtype=bool)
        for n in range(len(y)):
            rid = int(dist[n])
            table = intervals["global"] if band_kind == "global" else _table_for(intervals, rid)
            for h in range(FORECAST_HORIZON):
                off = table[str(h + 1)]
                lo = max(0.0, pred[n, h] + off["lower_offset"])
                hi = pred[n, h] + off["upper_offset"]
                hit[n, h] = lo <= y[n, h] <= max(hi, lo)
        return hit

    hit_g = covered("global")
    hit_d = covered("per_district")

    print("Overall coverage (target 80%):")
    for h in range(FORECAST_HORIZON):
        print(f"  Week {h + 1}:  global={hit_g[:, h].mean()*100:5.1f}%   "
              f"per-district={hit_d[:, h].mean()*100:5.1f}%")

    print("\nWeek-1 coverage for high-count districts:")
    for name in ("Colombo", "Gampaha", "Kalutara", "Kandy", "Jaffna"):
        rid = RDHS_NAMES.index(name)
        m = dist == rid
        print(f"  {name:<10} global={hit_g[m,0].mean()*100:5.1f}%   "
              f"per-district={hit_d[m,0].mean()*100:5.1f}%   (n={m.sum()})")


if __name__ == "__main__":
    main()
