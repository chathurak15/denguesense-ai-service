from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

from app.config import (
    CNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    TARGET_SCALER_PATH,
    LATEST_WINDOWS_PATH,
)
from app.models.cnn_classifier import load_cnn_model
from app.models.lstm_forecaster import load_lstm_model
from app.routers import classify, forecast, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


#Lifespan — load once, fail fast
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== DengueSense AI Service startup ===")

    #CNN classifier
    app.state.cnn_model = load_cnn_model(CNN_MODEL_PATH)

    # LSTM forecaster
    app.state.lstm_model = load_lstm_model(LSTM_MODEL_PATH)

    #Target scaler
    if not TARGET_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"target_scaler.pkl not found at {TARGET_SCALER_PATH}."
        )
    logger.info("Loading target scaler from %s …", TARGET_SCALER_PATH)
    app.state.target_scaler = joblib.load(TARGET_SCALER_PATH)
    logger.info("Target scaler loaded ✓")

    # Precomputed forecast windows
    if not LATEST_WINDOWS_PATH.exists():
        raise FileNotFoundError(
            f"latest_windows.pkl not found at {LATEST_WINDOWS_PATH}. "
            "Run  python scripts/build_forecast_windows.py  first."
        )
    logger.info("Loading forecast windows from %s …", LATEST_WINDOWS_PATH)
    app.state.latest_windows = joblib.load(LATEST_WINDOWS_PATH)
    logger.info(
        "Forecast windows loaded ✓  (%d districts)", len(app.state.latest_windows)
    )

    logger.info("=== All models loaded. Service is ready. ===")
    yield
    #Shutdown (nothing to teardown for Keras/joblib)
    logger.info("=== DengueSense AI Service shutdown ===")


#FastAPI app

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

# Routers

app.include_router(classify.router, tags=["Breeding Site Classification"])
app.include_router(forecast.router, tags=["Dengue Case Forecast"])
app.include_router(health.router, tags=["Health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
