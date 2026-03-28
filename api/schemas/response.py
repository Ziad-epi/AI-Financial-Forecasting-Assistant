from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Forecasted value")


class AskResponse(BaseModel):
    answer: str = Field(..., description="LLM answer")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")


class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Model class name")
    model_path: str = Field(..., description="Path to model artifact")
    features: Optional[List[str]] = Field(None, description="Feature names if available")
    n_features: Optional[int] = Field(None, description="Number of features")
    loaded_at: Optional[str] = Field(None, description="ISO timestamp when model loaded")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[object] = Field(None, description="Additional error details")
