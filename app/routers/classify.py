"""POST /classify — breeding-site image classification endpoint."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.schemas.classify_schema import ClassifyRequest, ClassifyResponse
from app.models.cnn_classifier import classify_image
from app.config import MODEL_VERSION

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request_body: ClassifyRequest, request: Request) -> ClassifyResponse:
    cnn_model = request.app.state.cnn_model

    #Download image
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(request_body.imageUrl)
            response.raise_for_status()
            image_bytes = response.content
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Image URL returned HTTP {exc.response.status_code}: {request_body.imageUrl}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not reach image URL: {exc}",
        )

    #Run CNN inference
    try:
        risk_label, confidence_score = classify_image(cnn_model, image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data at URL: {exc}",
        )

    logger.info(
        "classify: url=%s label=%s prob=%.4f",
        request_body.imageUrl,
        risk_label,
        confidence_score,
    )
    return ClassifyResponse(
        riskLabel=risk_label,
        confidenceScore=round(confidence_score, 4),
        modelVersion=MODEL_VERSION,
    )
