from __future__ import annotations

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import (
    CNN_MODEL_PATH,
    FEATURE_SCALER_PATH,
    LSTM_MODEL_PATH,
    RESIDUAL_INTERVALS_PATH,
    TARGET_SCALER_PATH,
    TEMP_ZSCORE_BASELINES_PATH,
    load_lstm_model_version,
)
from app.features.feature_engineering import load_temp_zscore_baselines
from app.models.cnn_classifier import load_cnn_model
from app.models.lstm_forecaster import load_lstm_model
from app.models.residual_intervals import load_residual_intervals
from app.routers import classify, forecast, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== DengueSense AI Service startup ===")

    app.state.cnn_model = load_cnn_model(CNN_MODEL_PATH)
    app.state.lstm_model = load_lstm_model(LSTM_MODEL_PATH)

    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(f"target_scaler.pkl not found at {TARGET_SCALER_PATH}.")
    logger.info("Loading target scaler from %s …", TARGET_SCALER_PATH)
    app.state.target_scaler = joblib.load(TARGET_SCALER_PATH)

    if not FEATURE_SCALER_PATH.exists():
        raise FileNotFoundError(f"feature_scaler.pkl not found at {FEATURE_SCALER_PATH}.")
    logger.info("Loading feature scaler from %s …", FEATURE_SCALER_PATH)
    app.state.feature_scaler = joblib.load(FEATURE_SCALER_PATH)

    if not TEMP_ZSCORE_BASELINES_PATH.exists():
        raise FileNotFoundError(
            f"temp_zscore_baselines.json not found at {TEMP_ZSCORE_BASELINES_PATH}. "
            "Run  python scripts/export_forecast_artifacts.py --skip-residuals"
        )
    logger.info("Loading temp_zscore baselines from %s …", TEMP_ZSCORE_BASELINES_PATH)
    app.state.temp_zscore_baselines = load_temp_zscore_baselines(TEMP_ZSCORE_BASELINES_PATH)

    if not RESIDUAL_INTERVALS_PATH.exists():
        raise FileNotFoundError(
            f"residual_intervals.json not found at {RESIDUAL_INTERVALS_PATH}. "
            "Run  python scripts/export_forecast_artifacts.py"
        )
    logger.info("Loading residual intervals from %s …", RESIDUAL_INTERVALS_PATH)
    app.state.residual_intervals = load_residual_intervals(RESIDUAL_INTERVALS_PATH)

    app.state.lstm_model_version = load_lstm_model_version()
    logger.info("LSTM model_version=%s", app.state.lstm_model_version)

    logger.info("=== All models loaded. Service is ready. ===")
    yield
    logger.info("=== DengueSense AI Service shutdown ===")


app = FastAPI(
    title="DengueSense AI Service",
    description=(
        "Microservice for the DengueSense LK platform. "
        "Exposes two ML endpoints:\n\n"
        "- **POST /classify** — CNN breeding-site risk classifier (MobileNetV3Large)\n"
        "- **POST /forecast** — LSTM 4-week dengue case forecaster (per district)\n\n"
        "Both models are loaded once at startup. See **/health** to confirm readiness."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(classify.router, tags=["Breeding Site Classification"])
app.include_router(forecast.router, tags=["Dengue Case Forecast"])
app.include_router(health.router, tags=["Health"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
