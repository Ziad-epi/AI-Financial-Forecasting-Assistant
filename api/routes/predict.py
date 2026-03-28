from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.schemas.request import PredictionRequest
from api.schemas.response import ErrorResponse, PredictionResponse
from api.services import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


def _to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    features = _to_dict(payload.features)
    logger.info("Predict request: %s", features)
    try:
        prediction = model_service.predict(features)
    except KeyError as exc:
        logger.warning("Missing feature: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing feature: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    logger.info("Predict response: %s", prediction)
    return PredictionResponse(prediction=prediction)
