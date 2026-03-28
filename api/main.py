from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes.ask import router as ask_router
from api.routes.predict import router as predict_router
from api.schemas.response import ErrorResponse, HealthResponse, ModelInfoResponse
from api.services import model_service, rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="Financial Forecasting & RAG API", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    logger.info("Starting API")
    model_service.load()
    rag_service.init()


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    payload = ErrorResponse(error="Invalid request", details=exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=_to_dict(payload))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    info = model_service.info()
    return ModelInfoResponse(**asdict(info))


app.include_router(predict_router)
app.include_router(ask_router)


def _to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
