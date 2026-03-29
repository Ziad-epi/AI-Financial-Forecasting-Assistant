from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.schemas.request import DecisionRequest
from api.schemas.response import DecisionResponse, ErrorResponse
from api.services import model_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decision"])


def _to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post(
    "/decision",
    response_model=DecisionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def decision(payload: DecisionRequest) -> DecisionResponse:
    features = _to_dict(payload.features)
    logger.info("Decision request: %s", features)
    try:
        result = model_service.decision(features)
    except (KeyError, ValueError) as exc:
        logger.warning("Invalid features: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Decision failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision failed: {exc}",
        ) from exc

    return DecisionResponse(**result)
