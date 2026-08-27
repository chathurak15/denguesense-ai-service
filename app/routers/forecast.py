"""POST /forecast — district-level 4-week dengue case forecast endpoint."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.features.feature_engineering import RAW_HISTORY_COLS, build_model_inputs
from app.models.lstm_forecaster import run_forecast
from app.models.residual_intervals import apply_prediction_intervals
from app.schemas.forecast_schema import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _history_to_frame(request_body: ForecastRequest) -> pd.DataFrame:
    rows = [
        {
            "week_start_date": row.week_start_date,
            "week_end_date": row.week_end_date,
            "temp_mean": row.temp_mean,
            "temp_max": row.temp_max,
            "temp_min": row.temp_min,
            "rainfall_mm": row.rainfall_mm,
            "humidity_pct": row.humidity_pct,
            "week_cases": row.week_cases,
            "week_no": row.week_no,
        }
        for row in request_body.history
    ]
    return pd.DataFrame(rows, columns=list(RAW_HISTORY_COLS))


def _warn_if_cumulative_not_non_decreasing(request_body: ForecastRequest) -> None:
    prev = request_body.history[0].cumulative_cases
    for i, row in enumerate(request_body.history[1:], start=1):
        if row.cumulative_cases < prev:
            logger.warning(
                "forecast: cumulative_cases decreased at history[%d] "
                "(%d -> %d) rdhs_id=%d — possible upstream data correction",
                i,
                prev,
                row.cumulative_cases,
                request_body.rdhs_id,
            )
            return
        prev = row.cumulative_cases


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request_body: ForecastRequest, request: Request) -> ForecastResponse:
    """Synchronous route: Keras predict is blocking CPU work.

    FastAPI runs `def` routes in a threadpool so the event loop is not stalled.
    """
    _warn_if_cumulative_not_non_decreasing(request_body)

    static = request_body.static_features
    try:
        inputs = build_model_inputs(
            _history_to_frame(request_body),
            rdhs_id=request_body.rdhs_id,
            district_name=request_body.district_name,
            zone_dry_zone=static.zone_dry_zone,
            zone_intermediate_zone=static.zone_intermediate_zone,
            zone_wet_zone=static.zone_wet_zone,
            population_density=static.population_density,
            feature_scaler=request.app.state.feature_scaler,
            target_scaler=request.app.state.target_scaler,
            baselines=request.app.state.temp_zscore_baselines,
        )
        real_cases = run_forecast(
            request.app.state.lstm_model,
            request.app.state.target_scaler,
            inputs.sequence_input,
            inputs.rdhs_id,
            inputs.static_features,
        )
        predictions, lower, upper = apply_prediction_intervals(
            real_cases,
            request.app.state.residual_intervals,
            rdhs_id=request_body.rdhs_id,
        )
    except ValueError as exc:
        logger.exception("forecast: feature/inference error rdhs_id=%s", request_body.rdhs_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    except Exception:
        logger.exception("forecast: unexpected error rdhs_id=%s", request_body.rdhs_id)
        raise HTTPException(status_code=500, detail="Internal server error")

    logger.info(
        "forecast: rdhs_id=%d district=%s predictions=%s",
        request_body.rdhs_id,
        request_body.district_name,
        predictions,
    )
    return ForecastResponse(
        predictions=predictions,
        lower_bounds=lower,
        upper_bounds=upper,
        model_version=request.app.state.lstm_model_version,
    )
