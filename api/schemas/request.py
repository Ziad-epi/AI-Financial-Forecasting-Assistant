from __future__ import annotations

from pydantic import BaseModel, Field


class Features(BaseModel):
    lag_1: float = Field(..., description="Lag 1 value")
    lag_2: float = Field(..., description="Lag 2 value")
    lag_3: float = Field(..., description="Lag 3 value")
    MA7: float = Field(..., description="7-day moving average")
    MA30: float = Field(..., description="30-day moving average")
    volatility: float = Field(..., description="Rolling volatility")


class PredictionRequest(BaseModel):
    features: Features


class DecisionRequest(BaseModel):
    features: Features


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
