"""Section 3 request-validation tests — each rule violated in isolation."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.schemas.forecast_schema import ForecastRequest
from tests.conftest import GOLDEN_TARGET, golden_payload


def _error_text(exc: ValidationError) -> str:
    return " | ".join(err["msg"] for err in exc.errors())


def test_valid_golden_request_parses():
    req = ForecastRequest.model_validate(golden_payload())
    assert req.rdhs_id == 4
    assert len(req.history) == 8
    assert req.target_week_start == GOLDEN_TARGET


def test_history_must_have_exactly_eight_entries():
    payload = golden_payload()
    payload["history"] = payload["history"][:7]
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "history" in _error_text(exc.value).lower() or any(
        "history" in e["loc"] for e in exc.value.errors()
    )

    payload = golden_payload()
    extra = deepcopy(payload["history"][-1])
    extra["week_start_date"] = (
        date.fromisoformat(extra["week_end_date"]) + timedelta(days=1)
    ).isoformat()
    extra["week_end_date"] = (
        date.fromisoformat(extra["week_start_date"]) + timedelta(days=6)
    ).isoformat()
    payload["history"].append(extra)
    with pytest.raises(ValidationError):
        ForecastRequest.model_validate(payload)


def test_history_not_monday_sunday():
    payload = golden_payload()
    payload["history"][0]["week_start_date"] = "2025-11-18"  # Tuesday
    payload["history"][0]["week_end_date"] = "2025-11-24"
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "Monday" in _error_text(exc.value)


def test_history_gap():
    payload = golden_payload()
    # Skip a week between index 3 and 4 by shifting 4..7 forward 7 days,
    # but keep target — also breaks target link; isolate the gap message
    # by also moving target if needed. Here we only bump week 4 start.
    payload["history"][4]["week_start_date"] = "2025-12-22"
    payload["history"][4]["week_end_date"] = "2025-12-28"
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "consecutive" in _error_text(exc.value)


def test_history_unsorted():
    payload = golden_payload()
    payload["history"][0], payload["history"][1] = (
        payload["history"][1],
        payload["history"][0],
    )
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "sorted" in _error_text(exc.value)


def test_last_week_must_end_day_before_target():
    payload = golden_payload()
    payload["target_week_start"] = "2026-01-19"
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "target_week_start" in _error_text(exc.value)


def test_zone_all_zeros():
    payload = golden_payload()
    payload["static_features"]["zone_wet_zone"] = 0.0
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "exactly one" in _error_text(exc.value)


def test_zone_two_hot():
    payload = golden_payload()
    payload["static_features"]["zone_dry_zone"] = 1.0
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "exactly one" in _error_text(exc.value)


def test_zone_non_binary():
    payload = golden_payload()
    payload["static_features"]["zone_wet_zone"] = 0.5
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "0.0 or 1.0" in _error_text(exc.value)


def test_rdhs_id_out_of_range():
    payload = golden_payload()
    payload["rdhs_id"] = 26
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert any("rdhs_id" in e["loc"] for e in exc.value.errors())

    payload["rdhs_id"] = -1
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert any("rdhs_id" in e["loc"] for e in exc.value.errors())


def test_negative_weather_and_density():
    payload = golden_payload()
    payload["history"][0]["rainfall_mm"] = -1.0
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "rainfall_mm" in _error_text(exc.value)

    payload = golden_payload()
    payload["static_features"]["population_density"] = -10.0
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "population_density" in _error_text(exc.value)


def test_nan_and_inf_rejected():
    payload = golden_payload()
    payload["history"][2]["temp_mean"] = float("nan")
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "temp_mean" in _error_text(exc.value)

    payload = golden_payload()
    payload["history"][2]["humidity_pct"] = float("inf")
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert "humidity_pct" in _error_text(exc.value)


def test_negative_week_cases():
    payload = golden_payload()
    payload["history"][0]["week_cases"] = -3
    with pytest.raises(ValidationError) as exc:
        ForecastRequest.model_validate(payload)
    assert any("week_cases" in e["loc"] for e in exc.value.errors())


def test_http_422_identifies_field(forecast_client):
    payload = golden_payload()
    payload["rdhs_id"] = 99
    response = forecast_client.post("/forecast", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    locs = [tuple(item.get("loc", ())) for item in body["detail"]]
    assert any("rdhs_id" in loc for loc in locs)


def test_http_422_history_length(forecast_client):
    payload = golden_payload()
    payload["history"] = payload["history"][:3]
    response = forecast_client.post("/forecast", json=payload)
    assert response.status_code == 422
    assert any(
        "history" in item.get("loc", []) for item in response.json()["detail"]
    )
