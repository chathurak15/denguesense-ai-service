"""GET /health — liveness/readiness probe for Spring Boot and deployment checks."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    flags = {
        "cnn_breeding_site_classifier": getattr(state, "cnn_model", None) is not None,
        "lstm_dengue_forecaster": getattr(state, "lstm_model", None) is not None,
        "feature_scaler": getattr(state, "feature_scaler", None) is not None,
        "target_scaler": getattr(state, "target_scaler", None) is not None,
        "temp_zscore_baselines": getattr(state, "temp_zscore_baselines", None) is not None,
        "residual_intervals": getattr(state, "residual_intervals", None) is not None,
    }
    ready = all(flags.values())
    return HealthResponse(
        status="ok" if ready else "degraded",
        models={name: "loaded" if ok else "missing" for name, ok in flags.items()},
    )
