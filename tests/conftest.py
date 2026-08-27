"""Shared fixtures and the Section-2 golden request body."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import (
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    TEMP_ZSCORE_BASELINES_PATH,
    load_lstm_model_version,
)
from app.features.feature_engineering import load_temp_zscore_baselines
from app.models.residual_intervals import load_residual_intervals
from app.routers.forecast import router as forecast_router

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "app" / "weights"
SIBLING_DATA = PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed"

# Prompt Section 2: 8 consecutive Mon–Sun weeks ending the day before 2026-01-12.
GOLDEN_TARGET = date(2026, 1, 12)
GOLDEN_HISTORY_STARTS = [date(2025, 11, 17) + timedelta(days=7 * i) for i in range(8)]


def golden_history_rows() -> list[dict]:
    rows = []
    cumulative = 400
    cases = [12, 14, 15, 16, 18, 17, 19, 18]
    weather = [
        (27.4, 31.2, 24.1, 42.6, 81.3),
        (27.6, 31.0, 24.3, 38.1, 80.8),
        (27.1, 30.8, 23.9, 55.0, 82.1),
        (26.9, 30.5, 23.7, 61.2, 83.0),
        (27.0, 30.9, 24.0, 48.4, 81.7),
        (27.2, 31.1, 24.2, 33.5, 80.4),
        (27.5, 31.3, 24.4, 29.0, 79.9),
        (27.3, 31.0, 24.1, 36.8, 80.6),
    ]
    for start, week_cases, (tmean, tmax, tmin, rain, humid) in zip(
        GOLDEN_HISTORY_STARTS, cases, weather
    ):
        cumulative += week_cases
        rows.append(
            {
                "week_start_date": start.isoformat(),
                "week_end_date": (start + timedelta(days=6)).isoformat(),
                "week_no": start.isocalendar()[1],
                "temp_mean": tmean,
                "temp_max": tmax,
                "temp_min": tmin,
                "rainfall_mm": rain,
                "humidity_pct": humid,
                "week_cases": week_cases,
                "cumulative_cases": cumulative,
            }
        )
    return rows


def golden_payload(**overrides) -> dict:
    body = {
        "rdhs_id": 4,
        "district_name": "Colombo",
        "target_week_start": GOLDEN_TARGET.isoformat(),
        "static_features": {
            "zone_dry_zone": 0.0,
            "zone_intermediate_zone": 0.0,
            "zone_wet_zone": 1.0,
            "population_density": 3392.0,
        },
        "history": golden_history_rows(),
    }
    body.update(overrides)
    return body


class FakeLSTM:
    def predict(self, inputs, verbose=0):
        seq = inputs["sequence_input"]
        assert seq.shape[0] == 1 and seq.shape[1] == 4
        return np.array([[0.015, 0.016, 0.017, 0.014]], dtype=np.float32)


@pytest.fixture(scope="session")
def feature_scaler():
    if not FEATURE_SCALER_PATH.exists():
        pytest.skip(f"missing {FEATURE_SCALER_PATH}")
    return joblib.load(FEATURE_SCALER_PATH)


@pytest.fixture(scope="session")
def target_scaler():
    if not TARGET_SCALER_PATH.exists():
        pytest.skip(f"missing {TARGET_SCALER_PATH}")
    return joblib.load(TARGET_SCALER_PATH)


@pytest.fixture(scope="session")
def baselines():
    if not TEMP_ZSCORE_BASELINES_PATH.exists():
        pytest.skip(f"missing {TEMP_ZSCORE_BASELINES_PATH}")
    return load_temp_zscore_baselines(TEMP_ZSCORE_BASELINES_PATH)


@pytest.fixture
def forecast_client(feature_scaler, target_scaler, baselines):
    intervals_path = WEIGHTS_DIR / "residual_intervals.json"
    if not intervals_path.exists():
        pytest.skip(f"missing {intervals_path}")

    app = FastAPI()
    app.include_router(forecast_router)
    app.state.lstm_model = FakeLSTM()
    app.state.feature_scaler = feature_scaler
    app.state.target_scaler = target_scaler
    app.state.temp_zscore_baselines = baselines
    app.state.residual_intervals = load_residual_intervals(intervals_path)
    app.state.lstm_model_version = (
        load_lstm_model_version() if (WEIGHTS_DIR / "model_metadata.json").exists() else "lstm-v1"
    )
    with TestClient(app) as client:
        yield client
