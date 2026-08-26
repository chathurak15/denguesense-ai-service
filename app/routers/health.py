"""GET /health — liveness/readiness probe for Spring Boot and deployment checks."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    cnn_ok = request.app.state.cnn_model is not None
    lstm_ok = request.app.state.lstm_model is not None
    windows_ok = request.app.state.latest_windows is not None

    return HealthResponse(
        status="ok",
        models={
            "cnn_breeding_site_classifier": "loaded" if cnn_ok else "missing",
            "lstm_dengue_forecaster": "loaded" if lstm_ok else "missing",
            "forecast_windows": "loaded" if windows_ok else "missing",
        },
    )
