"""
End-to-end check that the /forecast serving pipeline reproduces the notebook's
validation-set accuracy, and quantifies the week_no bug.

Windows are built inside the 2024 validation split exactly like
notebooks/06 + 07. For each window we predict three ways:

  A. ground-truth : sequence taken straight from scaled val.csv (what the model
     was evaluated on in notebook 07).
  B. serving+fix  : app.features.build_model_inputs on 8 raw weeks, week_no
     supplied (the fixed serving path).
  C. serving-bug  : same, but week_no dropped so the old isocalendar() fallback
     runs (approximates the previous behaviour).

MAE per horizon is reported for each, against the real (inverse-transformed)
weekly case counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import LSTM_MODEL_PATH  # noqa: E402
from app.features.feature_engineering import (  # noqa: E402
    FORECAST_HORIZON,
    LOOKBACK,
    SEQ_COLS,
    STATIC_COLS,
    build_model_inputs,
    load_temp_zscore_baselines,
)
from app.models.lstm_forecaster import load_lstm_model  # noqa: E402

SIBLING = PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed"
FEAT_CSV = SIBLING / "dengue_2011_2026_feature_engineering_&_encoding.csv"
VAL_CSV = SIBLING / "scaled" / "val.csv"
WEIGHTS = PROJECT_ROOT / "app" / "weights"

RAW_COLS = [
    "week_start_date", "week_end_date", "temp_mean", "temp_max", "temp_min",
    "rainfall_mm", "humidity_pct", "week_cases", "week_no",
]


def mae_report(name, pred_real, y_real):
    print(f"\n{name}")
    for h in range(FORECAST_HORIZON):
        mae = float(np.mean(np.abs(pred_real[:, h] - y_real[:, h])))
        print(f"   Week {h + 1}: MAE = {mae:6.2f} cases")


def main() -> None:
    feat = pd.read_csv(FEAT_CSV, parse_dates=["week_start_date", "week_end_date"])
    val = pd.read_csv(VAL_CSV, parse_dates=["week_start_date", "week_end_date"])
    feature_scaler = joblib.load(WEIGHTS / "feature_scaler.pkl")
    target_scaler = joblib.load(WEIGHTS / "target_scaler.pkl")
    baselines = load_temp_zscore_baselines(WEIGHTS / "temp_zscore_baselines.json")
    feat = feat.sort_values(["rdhs", "week_start_date"]).reset_index(drop=True)

    A_seq, B_seq, C_seq, static_l, rdhs_l, y_real_l, districts_l = [], [], [], [], [], [], []

    for district in sorted(val["rdhs"].unique()):
        vg = val[val["rdhs"] == district].sort_values(["year", "week_no"]).reset_index(drop=True)
        fg = feat[feat["rdhs"] == district].sort_values("week_start_date").reset_index(drop=True)
        fg_by_date = {d: k for k, d in enumerate(fg["week_start_date"])}

        row0 = fg.iloc[0]
        zdry, zint, zwet = (
            float(row0["zone_dry_zone"]),
            float(row0["zone_intermediate_zone"]),
            float(row0["zone_wet_zone"]),
        )
        pop_density = float(row0["population_density"])
        rdhs_id = int(row0["rdhs_id"])

        for i in range(len(vg) - LOOKBACK - FORECAST_HORIZON + 1):
            lb = vg.iloc[i : i + LOOKBACK]
            tgt = vg.iloc[i + LOOKBACK : i + LOOKBACK + FORECAST_HORIZON]

            # ground truth (A): straight from scaled val.csv
            A_seq.append(lb[SEQ_COLS].to_numpy(dtype=np.float32))
            static_l.append(lb.iloc[-1][STATIC_COLS].to_numpy(dtype=np.float32))
            rdhs_l.append(rdhs_id)
            y_real_l.append(tgt["week_cases"].to_numpy(dtype=np.float64))

            # serving: 8 raw weeks ending at the last lookback week
            p = fg_by_date[lb.iloc[-1]["week_start_date"]]
            hist = fg.iloc[p - 7 : p + 1][RAW_COLS].copy()

            b = build_model_inputs(
                hist, rdhs_id=rdhs_id, district_name=district,
                zone_dry_zone=zdry, zone_intermediate_zone=zint, zone_wet_zone=zwet,
                population_density=pop_density, feature_scaler=feature_scaler,
                target_scaler=target_scaler, baselines=baselines,
            )
            B_seq.append(b.sequence_input[0])

            # OLD production behaviour: Spring Boot sends Monday-Sunday weeks
            # (raw Saturday + 2 days) and the old code did isocalendar(week_start).
            hist_old = hist.copy()
            monday = hist_old["week_start_date"] + pd.Timedelta(days=2)
            hist_old["week_start_date"] = monday
            hist_old["week_end_date"] = monday + pd.Timedelta(days=6)
            hist_old["week_no"] = monday.dt.isocalendar().week.astype(int).to_numpy()
            c = build_model_inputs(
                hist_old, rdhs_id=rdhs_id, district_name=district,
                zone_dry_zone=zdry, zone_intermediate_zone=zint, zone_wet_zone=zwet,
                population_density=pop_density, feature_scaler=feature_scaler,
                target_scaler=target_scaler, baselines=baselines,
            )
            C_seq.append(c.sequence_input[0])
            districts_l.append(district)

    A_seq = np.asarray(A_seq, np.float32)
    B_seq = np.asarray(B_seq, np.float32)
    C_seq = np.asarray(C_seq, np.float32)
    static = np.asarray(static_l, np.float32)
    rdhs = np.asarray(rdhs_l, np.int32).reshape(-1, 1)
    y_real = np.asarray(y_real_l, np.float64)
    print(f"windows: {len(y_real)}")

    model = load_lstm_model(LSTM_MODEL_PATH)

    def predict_real(seq):
        scaled = model.predict(
            {"sequence_input": seq, "rdhs_id": rdhs, "static_features": static}, verbose=0
        )
        out = np.zeros_like(scaled, dtype=np.float64)
        for h in range(FORECAST_HORIZON):
            out[:, h] = target_scaler.inverse_transform(scaled[:, h].reshape(-1, 1)).ravel()
        return out

    predA, predB, predC = predict_real(A_seq), predict_real(B_seq), predict_real(C_seq)
    mae_report("A. ground-truth val.csv (notebook 07 path)", predA, y_real)
    mae_report("B. serving pipeline WITH week_no fix", predB, y_real)
    mae_report("C. serving pipeline OLD isocalendar-on-Monday (buggy production)", predC, y_real)

    # Per-district week-1 comparison, buggy vs fixed
    dist = np.asarray(districts_l)
    print("\nPer-district Week-1 MAE (fixed vs buggy), top by |delta|:")
    rows = []
    for d in sorted(set(districts_l)):
        m = dist == d
        mae_fix = float(np.mean(np.abs(predB[m, 0] - y_real[m, 0])))
        mae_bug = float(np.mean(np.abs(predC[m, 0] - y_real[m, 0])))
        rows.append((mae_bug - mae_fix, d, mae_fix, mae_bug))
    print(f"   {'district':<14}{'fixed':>8}{'buggy':>8}{'delta':>8}")
    for delta, d, mf, mb in sorted(rows, key=lambda r: abs(r[0]), reverse=True)[:10]:
        print(f"   {d:<14}{mf:>8.2f}{mb:>8.2f}{delta:>+8.2f}")


if __name__ == "__main__":
    main()
