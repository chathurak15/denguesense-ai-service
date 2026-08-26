"""Pydantic schemas for POST /classify."""

from pydantic import BaseModel, HttpUrl


class ClassifyRequest(BaseModel):
    imageUrl: str  # Cloudinary or any public HTTPS URL


class ClassifyResponse(BaseModel):
    riskLabel: str          # "HIGH_RISK" | "LOW_RISK" | "INVALID"
    confidenceScore: float  # raw sigmoid probability, 0.0 – 1.0
    modelVersion: str       # version of the model used
