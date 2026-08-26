"""POST /forecast — district-level 4-week dengue case forecast endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.forecast_schema import ForecastRequest, ForecastResponse, WeekForecast
from app.models.lstm_forecaster import run_forecast

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request_body: ForecastRequest, request: Request) -> ForecastResponse:
    lstm_model = request.app.state.lstm_model
    target_scaler = request.app.state.target_scaler
    latest_windows = request.app.state.latest_windows

    district_id = request_body.districtId

    # ── Look up precomputed window ────────────────────────────────────────────
    window = latest_windows.get(district_id)
    if window is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No precomputed forecast window found for districtId={district_id}. "
                "Valid ids are 0-25 (see endpoint description). "
                "If you recently added data, re-run scripts/build_forecast_windows.py."
            ),
        )

    # ── Run LSTM inference ────────────────────────────────────────────────────
    real_cases = run_forecast(lstm_model, target_scaler, window)

    forecast_weeks = [
        WeekForecast(weekAhead=week, predictedCases=round(cases, 2))
        for week, cases in enumerate(real_cases, start=1)
    ]

    logger.info(
        "forecast: districtId=%d predictions=%s",
        district_id,
        [w.predictedCases for w in forecast_weeks],
    )
    return ForecastResponse(districtId=district_id, forecast=forecast_weeks)
