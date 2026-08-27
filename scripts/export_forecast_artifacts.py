"""
Export serving artifacts that the training notebooks never saved:

  - app/weights/temp_zscore_baselines.json
      Per-(rdhs, season) mean/std of temp_mean, fit on training years
      (<= 2023) exactly as in notebooks/05_split_and_scaling.ipynb.
  - app/weights/residual_intervals.json
      Per-horizon empirical 10th/90th percentile of (actual - predicted)
      residuals on the **validation** set (2024), not test.
  - app/weights/model_metadata.json
      model_version read by POST /forecast (not hardcoded in the route).

Usage (from denguesense-ai-service root):
    python scripts/export_forecast_artifacts.py
    python scripts/export_forecast_artifacts.py --skip-residuals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.features.feature_engineering import (  # noqa: E402
    CLIMATE_ZONE_MAP,
    FORECAST_HORIZON,
    LOOKBACK,
    RDHS_NAMES,
    SEQ_COLS,
    STATIC_COLS,
    get_season,
)

SIBLING_DATA = PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed"
FEAT_CSV = SIBLING_DATA / "dengue_2011_2026_feature_engineering_&_encoding.csv"
VAL_CSV = SIBLING_DATA / "scaled" / "val.csv"
WEIGHTS_DIR = PROJECT_ROOT / "app" / "weights"

# Empirical interval configuration.
PERCENTILE_LOW = 10
PERCENTILE_HIGH = 90
# Districts with fewer validation windows than this fall back to the global band.
MIN_DISTRICT_WINDOWS = 20


def export_temp_zscore_baselines() -> dict:
    if not FEAT_CSV.exists():
        raise FileNotFoundError(f"Feature-engineered CSV not found: {FEAT_CSV}")

    df = pd.read_csv(FEAT_CSV, parse_dates=["week_start_date", "week_end_date"])
    df["climate_zone"] = df["rdhs"].map(CLIMATE_ZONE_MAP)
    df["season"] = [
        get_season(int(week), zone)
        for week, zone in zip(df["week_no"], df["climate_zone"])
    ]

    train = df[df["year"] <= 2023]
    grouped = train.groupby(["rdhs", "season"])["temp_mean"]
    means = grouped.mean()
    stds = grouped.std()

    payload: dict[str, dict[str, float]] = {}
    for (rdhs, season), mean in means.items():
        std = float(stds[(rdhs, season)])
        payload[f"{rdhs}|{season}"] = {"mean": float(mean), "std": std}

    out = WEIGHTS_DIR / "temp_zscore_baselines.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({len(payload)} groups)")
    return payload


def export_model_metadata() -> None:
    out = WEIGHTS_DIR / "model_metadata.json"
    payload = {
        "model_version": "lstm-v1",
        "weights_file": "_28_loss_0.0039_weights.keras",
        "lookback_weeks": LOOKBACK,
        "forecast_horizon_weeks": FORECAST_HORIZON,
    }
    if not out.exists():
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(f"Kept existing {out}")


def _make_val_sequences(val: pd.DataFrame):
    """Same windowing as notebooks/06_model_training_lstm.ipynb, validation split only."""
    X_seq, X_static, X_district, y = [], [], [], []
    for _, group in val.groupby("rdhs"):
        group = group.sort_values(["year", "week_no"]).reset_index(drop=True)
        seq_raw = group[SEQ_COLS].values
        static_raw = group[STATIC_COLS].values
        district_raw = group["rdhs_id"].values
        y_raw = group["week_cases_scaled"].values
        for i in range(len(group) - LOOKBACK - FORECAST_HORIZON + 1):
            X_seq.append(seq_raw[i : i + LOOKBACK])
            X_static.append(static_raw[i + LOOKBACK - 1])
            X_district.append(district_raw[i + LOOKBACK - 1])
            y.append(y_raw[i + LOOKBACK : i + LOOKBACK + FORECAST_HORIZON])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_static, dtype=np.float32),
        np.array(X_district, dtype=np.int32).reshape(-1, 1),
        np.array(y, dtype=np.float32),
    )


def export_residual_intervals() -> None:
    import joblib

    from app.config import LSTM_MODEL_PATH, TARGET_SCALER_PATH
    from app.models.lstm_forecaster import load_lstm_model

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Scaled val CSV not found: {VAL_CSV}")

    val = pd.read_csv(VAL_CSV, parse_dates=["week_start_date", "week_end_date"])
    X_seq, X_static, X_district, y_val = _make_val_sequences(val)
    print(f"Val windows: {X_seq.shape[0]}")

    model = load_lstm_model(LSTM_MODEL_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    scaled_pred = model.predict(
        {
            "sequence_input": X_seq,
            "rdhs_id": X_district,
            "static_features": X_static,
        },
        verbose=0,
    )

    pred_real = np.zeros_like(scaled_pred)
    y_real = np.zeros_like(y_val)
    for h in range(FORECAST_HORIZON):
        pred_real[:, h] = target_scaler.inverse_transform(
            scaled_pred[:, h].reshape(-1, 1)
        ).ravel()
        y_real[:, h] = target_scaler.inverse_transform(
            y_val[:, h].reshape(-1, 1)
        ).ravel()

    residuals = y_real - pred_real  # actual - predicted
    district_ids = X_district.ravel()

    def _offsets(res: np.ndarray) -> dict:
        """Per-horizon empirical percentile offsets for a residual matrix."""
        out: dict[str, dict[str, float]] = {}
        for h in range(FORECAST_HORIZON):
            r = res[:, h]
            out[str(h + 1)] = {
                "lower_offset": round(float(np.percentile(r, PERCENTILE_LOW)), 4),
                "upper_offset": round(float(np.percentile(r, PERCENTILE_HIGH)), 4),
                "std": round(float(np.std(r, ddof=1)), 4) if len(r) > 1 else 0.0,
                "mean": round(float(np.mean(r)), 4),
            }
        return out

    # Global (fallback) offsets.
    global_intervals = _offsets(residuals)
    print("Global offsets:")
    for h in range(FORECAST_HORIZON):
        e = global_intervals[str(h + 1)]
        print(f"  horizon {h + 1}: p{PERCENTILE_LOW}={e['lower_offset']:.2f}  "
              f"p{PERCENTILE_HIGH}={e['upper_offset']:.2f}  std={e['std']:.2f}")

    # Per-district offsets — this is what makes big districts (Colombo/Gampaha)
    # get a band that reflects their true error scale instead of the fleet mean.
    per_district: dict[str, dict] = {}
    print("\nPer-district offsets (horizon-1 shown):")
    for did in sorted(np.unique(district_ids)):
        mask = district_ids == did
        n = int(mask.sum())
        name = RDHS_NAMES[int(did)] if 0 <= int(did) < len(RDHS_NAMES) else str(did)
        if n < MIN_DISTRICT_WINDOWS:
            print(f"  [{did:>2}] {name:<14} n={n:<3} -> too few windows, uses global")
            continue
        offs = _offsets(residuals[mask])
        for h in range(FORECAST_HORIZON):
            offs[str(h + 1)]["n_windows"] = n
        per_district[str(int(did))] = offs
        h1 = offs["1"]
        print(f"  [{did:>2}] {name:<14} n={n:<3} "
              f"h1=[{h1['lower_offset']:+.1f}, {h1['upper_offset']:+.1f}]")

    payload = {
        "method": "empirical_percentile_per_district",
        "percentile_low": PERCENTILE_LOW,
        "percentile_high": PERCENTILE_HIGH,
        "source": "validation_set_year_2024",
        "n_windows": int(X_seq.shape[0]),
        "min_district_windows": MIN_DISTRICT_WINDOWS,
        "note": (
            "Per-district empirical percentile band from validation-set residuals "
            "(actual - predicted). Districts with fewer than min_district_windows "
            "fall back to the global band. This is a heuristic interval, NOT a "
            "conformal prediction set or a Bayesian credible interval."
        ),
        "global": {
            k: {"lower_offset": v["lower_offset"], "upper_offset": v["upper_offset"]}
            for k, v in global_intervals.items()
        },
        "per_district": {
            did: {
                k: {"lower_offset": v["lower_offset"], "upper_offset": v["upper_offset"]}
                for k, v in offs.items()
            }
            for did, offs in per_district.items()
        },
        "diagnostics": {"global": global_intervals, "per_district": per_district},
    }
    out = WEIGHTS_DIR / "residual_intervals.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}  (global + {len(per_district)} districts)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-residuals",
        action="store_true",
        help="Only export baselines + metadata (no Keras).",
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    export_model_metadata()
    export_temp_zscore_baselines()
    if not args.skip_residuals:
        export_residual_intervals()


if __name__ == "__main__":
    main()
