from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from api.schemas.request import AskRequest
from api.schemas.response import AskResponse, ErrorResponse
from api.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def ask(payload: AskRequest) -> AskResponse:
    data = _to_dict(payload)
    logger.info("Ask request: %s", data)
    try:
        answer = rag_service.ask(payload.question)
    except ValueError as exc:
        logger.warning("Invalid question: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("RAG failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG failed: {exc}",
        ) from exc

    logger.info("Ask response: %s", answer)
    return AskResponse(answer=answer)
