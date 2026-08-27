"""Golden POST /forecast contract test (shape, version, sanity — not exact numbers)."""

from __future__ import annotations

from tests.conftest import golden_payload


def test_golden_forecast_response_shape(forecast_client):
    response = forecast_client.post("/forecast", json=golden_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body.keys()) == {
        "predictions",
        "lower_bounds",
        "upper_bounds",
        "model_version",
    }
    assert body["model_version"]
    assert isinstance(body["model_version"], str)

    for key in ("predictions", "lower_bounds", "upper_bounds"):
        assert isinstance(body[key], list)
        assert len(body[key]) == 4
        assert all(isinstance(v, (int, float)) for v in body[key])
        assert all(v >= 0 for v in body[key])

    for pred, lo, hi in zip(
        body["predictions"], body["lower_bounds"], body["upper_bounds"]
    ):
        assert lo <= pred <= hi
        # 1 decimal place
        assert pred == round(pred, 1)
        assert lo == round(lo, 1)
        assert hi == round(hi, 1)


def test_cumulative_decrease_still_succeeds(forecast_client):
    payload = golden_payload()
    payload["history"][3]["cumulative_cases"] = 1  # not non-decreasing
    response = forecast_client.post("/forecast", json=payload)
    assert response.status_code == 200, response.text
