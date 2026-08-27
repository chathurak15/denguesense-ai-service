"""Feature-engineering unit tests + 8-week sufficiency (zero NaNs)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.features.feature_engineering import (
    LOOKBACK,
    MIN_HISTORY_WEEKS,
    SEQ_COLS,
    build_model_inputs,
    engineer_unscaled_features,
)
from tests.conftest import SIBLING_DATA, golden_history_rows

FEAT_CSV = SIBLING_DATA / "dengue_2011_2026_feature_engineering_&_encoding.csv"
SCALED_TEST_CSV = SIBLING_DATA / "scaled" / "test.csv"


def _history_frame_from_golden() -> pd.DataFrame:
    rows = []
    for raw in golden_history_rows():
        rows.append(
            {
                "week_start_date": date.fromisoformat(raw["week_start_date"]),
                "week_end_date": date.fromisoformat(raw["week_end_date"]),
                "temp_mean": raw["temp_mean"],
                "temp_max": raw["temp_max"],
                "temp_min": raw["temp_min"],
                "rainfall_mm": raw["rainfall_mm"],
                "humidity_pct": raw["humidity_pct"],
                "week_cases": raw["week_cases"],
                "week_no": raw["week_no"],
            }
        )
    return pd.DataFrame(rows)


def test_eight_weeks_is_sufficient_zero_nans(feature_scaler, target_scaler, baselines):
    inputs = build_model_inputs(
        _history_frame_from_golden(),
        rdhs_id=4,
        district_name="Colombo",
        zone_dry_zone=0.0,
        zone_intermediate_zone=0.0,
        zone_wet_zone=1.0,
        population_density=3392.0,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        baselines=baselines,
    )
    assert inputs.sequence_input.shape == (1, LOOKBACK, len(SEQ_COLS))
    assert inputs.rdhs_id.shape == (1, 1)
    assert inputs.static_features.shape == (1, 4)
    assert np.isfinite(inputs.sequence_input).all()
    assert np.isfinite(inputs.static_features).all()
    assert inputs.rdhs_id[0, 0] == 4
    # Zone one-hots pass through unscaled; density is scaled into [0, 1] (or slightly outside if OOD).
    assert list(inputs.static_features[0, :3]) == pytest.approx([0.0, 0.0, 1.0])


def test_unscaled_lags_and_rolling_on_golden_window(baselines):
    engineered = engineer_unscaled_features(
        _history_frame_from_golden(),
        district_name="Colombo",
        climate_zone="wet_zone",
        baselines=baselines,
    )
    lookback = engineered.iloc[-LOOKBACK:]
    assert lookback[SEQ_COLS[:-1]].notna().all().all()  # all but week_cases_scaled (not yet)

    # rain_roll4w_sum = current + lag1 + lag2 + lag3
    recon = (
        lookback["rainfall_mm"]
        + lookback["rain_lag_1w"]
        + lookback["rain_lag_2w"]
        + lookback["rain_lag_3w"]
    )
    np.testing.assert_allclose(lookback["rain_roll4w_sum"], recon, rtol=0, atol=1e-9)

    first_lookback = engineered.iloc[-LOOKBACK]
    # lag_4w of the earliest lookback row is history[0]
    raw = _history_frame_from_golden().sort_values("week_start_date").reset_index(drop=True)
    assert first_lookback["temp_lag_4w"] == pytest.approx(raw.loc[0, "temp_mean"])
    assert first_lookback["cases_lag_4w"] == pytest.approx(raw.loc[0, "week_cases"])


@pytest.mark.skipif(not FEAT_CSV.exists() or not SCALED_TEST_CSV.exists(), reason="training CSVs not present")
def test_matches_scaled_test_csv_colombo_2026(feature_scaler, target_scaler, baselines):
    feat = pd.read_csv(FEAT_CSV, parse_dates=["week_start_date", "week_end_date"])
    scaled = pd.read_csv(SCALED_TEST_CSV, parse_dates=["week_start_date", "week_end_date"])

    col = feat[(feat["rdhs"] == "Colombo") & (feat["year"] == 2026)].sort_values(
        "week_start_date"
    )
    assert len(col) >= MIN_HISTORY_WEEKS

    window = col.iloc[:MIN_HISTORY_WEEKS].copy()
    history = window[
        [
            "week_start_date",
            "week_end_date",
            "temp_mean",
            "temp_max",
            "temp_min",
            "rainfall_mm",
            "humidity_pct",
            "week_cases",
            "week_no",
        ]
    ].copy()
    last = window.iloc[-1]
    inputs = build_model_inputs(
        history,
        rdhs_id=int(last["rdhs_id"]),
        district_name="Colombo",
        zone_dry_zone=float(last["zone_dry_zone"]),
        zone_intermediate_zone=float(last["zone_intermediate_zone"]),
        zone_wet_zone=float(last["zone_wet_zone"]),
        population_density=float(last["population_density"]),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        baselines=baselines,
    )

    dates = window.iloc[-LOOKBACK:]["week_start_date"].dt.normalize()
    expected = (
        scaled[(scaled["rdhs"] == "Colombo") & (scaled["week_start_date"].isin(dates))]
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    assert len(expected) == LOOKBACK
    np.testing.assert_allclose(
        inputs.sequence_input[0],
        expected[SEQ_COLS].to_numpy(dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        inputs.static_features[0, 3],
        expected.iloc[-1]["population_density"],
        rtol=1e-5,
        atol=1e-6,
    )
