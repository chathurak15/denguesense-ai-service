# DengueSense AI Service

FastAPI microservice for the **DengueSense LK** platform. It serves two models loaded once at startup:

- **POST /classify** — MobileNetV3Large CNN that scores a breeding-site photo as `HIGH_RISK`, `LOW_RISK`, or `INVALID`
- **POST /forecast** — LSTM that predicts dengue case counts for the next 4 weeks for one RDHS district, with empirical prediction intervals

`GET /health` is the liveness/readiness probe used by Spring Boot and deployment checks.

Interactive docs are available at `/docs` once the service is running.

## Requirements

- Python 3.10 or 3.11+ (scikit-learn 1.8.x needs 3.11+; 3.10 installs 1.7.2)
- TensorFlow 2.16+
- Model weights and scalers under `app/weights/` (see [Model artifacts](#model-artifacts))

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m app.main
```

Or:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

The API listens on **port 5000**. Startup fails if any required weight or scaler file is missing.

## Endpoints

### `GET /health`

Returns `"ok"` when the CNN, LSTM, scalers, temperature baselines, and residual intervals are all loaded; otherwise `"degraded"`.

```json
{
  "status": "ok",
  "models": {
    "cnn_breeding_site_classifier": "loaded",
    "lstm_dengue_forecaster": "loaded",
    "feature_scaler": "loaded",
    "target_scaler": "loaded",
    "temp_zscore_baselines": "loaded",
    "residual_intervals": "loaded"
  }
}
```

### `POST /classify`

Downloads an image from a public URL and classifies it.

| Label | Sigmoid probability |
|---|---|
| `HIGH_RISK` | ≥ 0.7 |
| `LOW_RISK` | ≥ 0.3 and &lt; 0.7 |
| `INVALID` | &lt; 0.3 |

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"imageUrl": "https://example.com/breeding-site.jpg"}'
```

```json
{
  "riskLabel": "HIGH_RISK",
  "confidenceScore": 0.9123,
  "modelVersion": "mobilenetv3-breeding-site-v1.0.0"
}
```

`confidenceScore` is the raw sigmoid probability (0–1). Override the version string with `MODEL_VERSION` if needed.

### `POST /forecast`

Returns four weekly case predictions plus lower/upper bounds for one district.

Request rules (enforced by the schema):

- `rdhs_id` is 0–25
- `history` is **exactly 8** consecutive Monday–Sunday weeks
- `target_week_start` is the Monday immediately after the last history week
- Exactly one climate-zone flag is `1.0`
- Weather values and `population_density` are finite and non-negative

The first 4 history weeks supply lags; the last 4 form the LSTM lookback window.

```bash
curl -X POST http://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "rdhs_id": 4,
  "district_name": "Colombo",
  "target_week_start": "2026-01-12",
  "static_features": {
    "zone_dry_zone": 0.0,
    "zone_intermediate_zone": 0.0,
    "zone_wet_zone": 1.0,
    "population_density": 3392.0
  },
  "history": [
    {"week_start_date": "2025-11-17", "week_end_date": "2025-11-23", "week_no": 47, "temp_mean": 27.4, "temp_max": 31.2, "temp_min": 24.1, "rainfall_mm": 42.6, "humidity_pct": 81.3, "week_cases": 12, "cumulative_cases": 412},
    {"week_start_date": "2025-11-24", "week_end_date": "2025-11-30", "week_no": 48, "temp_mean": 27.6, "temp_max": 31.0, "temp_min": 24.3, "rainfall_mm": 38.1, "humidity_pct": 80.8, "week_cases": 14, "cumulative_cases": 426},
    {"week_start_date": "2025-12-01", "week_end_date": "2025-12-07", "week_no": 49, "temp_mean": 27.1, "temp_max": 30.8, "temp_min": 23.9, "rainfall_mm": 55.0, "humidity_pct": 82.1, "week_cases": 15, "cumulative_cases": 441},
    {"week_start_date": "2025-12-08", "week_end_date": "2025-12-14", "week_no": 50, "temp_mean": 26.9, "temp_max": 30.5, "temp_min": 23.7, "rainfall_mm": 61.2, "humidity_pct": 83.0, "week_cases": 16, "cumulative_cases": 457},
    {"week_start_date": "2025-12-15", "week_end_date": "2025-12-21", "week_no": 51, "temp_mean": 27.0, "temp_max": 30.9, "temp_min": 24.0, "rainfall_mm": 48.4, "humidity_pct": 81.7, "week_cases": 18, "cumulative_cases": 475},
    {"week_start_date": "2025-12-22", "week_end_date": "2025-12-28", "week_no": 52, "temp_mean": 27.2, "temp_max": 31.1, "temp_min": 24.2, "rainfall_mm": 33.5, "humidity_pct": 80.4, "week_cases": 17, "cumulative_cases": 492},
    {"week_start_date": "2025-12-29", "week_end_date": "2026-01-04", "week_no": 1,  "temp_mean": 27.5, "temp_max": 31.3, "temp_min": 24.4, "rainfall_mm": 29.0, "humidity_pct": 79.9, "week_cases": 19, "cumulative_cases": 511},
    {"week_start_date": "2026-01-05", "week_end_date": "2026-01-11", "week_no": 2,  "temp_mean": 27.3, "temp_max": 31.0, "temp_min": 24.1, "rainfall_mm": 36.8, "humidity_pct": 80.6, "week_cases": 18, "cumulative_cases": 529}
  ]
}
EOF
```

```json
{
  "predictions": [18.2, 19.1, 20.4, 17.8],
  "lower_bounds": [10.1, 9.4, 8.8, 7.2],
  "upper_bounds": [28.5, 31.0, 34.2, 30.1],
  "model_version": "lstm-v1"
}
```

Prediction intervals are the empirical 10th/90th percentile of validation residuals (per district, with a global fallback). They are a heuristic band, not a conformal or Bayesian interval.

`model_version` comes from `LSTM_MODEL_VERSION` if set, otherwise `app/weights/model_metadata.json`.

## Model artifacts

Place these files in `app/weights/` before starting the service:

| File | Role |
|---|---|
| `mobilenetv3_breeding_site_classifier_v1.keras` | CNN classifier |
| `_28_loss_0.0039_weights.keras` | LSTM forecaster |
| `feature_scaler.pkl` | Sequence / static feature scaler |
| `target_scaler.pkl` | Weekly case count scaler |
| `temp_zscore_baselines.json` | Per-(district, season) temperature mean/std |
| `residual_intervals.json` | Per-horizon prediction-interval offsets |
| `model_metadata.json` | LSTM version and lookback/horizon metadata |

JSON artifacts can be regenerated from the sibling `dengue_sense_lk` training data:

```bash
python scripts/export_forecast_artifacts.py
python scripts/export_forecast_artifacts.py --skip-residuals
```

## Tests

```bash
pytest
```

Coverage includes request validation, feature engineering (8-week sufficiency), residual-interval clamps, and a golden `/forecast` contract test.

## Other scripts

These expect processed CSVs from the sibling `dengue_sense_lk` repo (`../dengue_sense_lk/data/processed`).

| Script | Purpose |
|---|---|
| `scripts/export_forecast_artifacts.py` | Export baselines, residual intervals, and model metadata |
| `scripts/parity_check.py` | Compare serving features to scaled `val.csv` |
| `scripts/end_to_end_validation.py` | Compare serving MAE to notebook validation |
| `scripts/coverage_check.py` | Empirical coverage of prediction bands |
| `scripts/build_forecast_windows.py` | Precompute latest lookback windows (offline helper) |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_VERSION` | `mobilenetv3-breeding-site-v1.0.0` | CNN version returned by `/classify` |
| `LSTM_MODEL_VERSION` | from `model_metadata.json` | LSTM version returned by `/forecast` |

## Project layout

```
app/
  main.py                 # FastAPI app and model loading
  config.py               # Paths and version helpers
  routers/                # /classify, /forecast, /health
  schemas/                # Request/response models
  features/               # Train-aligned LSTM feature engineering
  models/                 # CNN, LSTM, residual-interval helpers
  weights/                # Keras weights, scalers, JSON artifacts
scripts/                  # Artifact export and validation
tests/
```
