from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Feature values required by the trained model."""

    return_1: float = Field(
        ...,
        description="Percentage return of the current candle.",
        examples=[0.0012],
    )
    ma_gap: float = Field(
        ...,
        description="Gap between the current price and moving average.",
        examples=[-0.0021],
    )
    volatility_10: float = Field(
        ...,
        ge=0,
        description="Rolling 10-candle return volatility.",
        examples=[0.0045],
    )
    candle_body_ratio: float = Field(
        ...,
        ge=0,
        description="Candle body size relative to candle range.",
        examples=[0.62],
    )
    rsi_14: float = Field(
        ...,
        ge=0,
        le=100,
        description="Fourteen-period Relative Strength Index.",
        examples=[54.3],
    )


class PredictionResponse(BaseModel):
    """Model prediction returned by the API."""

    predicted_class: int
    direction: str
    probability_up: float
    probability_down: float
    confidence: float
    threshold: float


class HealthResponse(BaseModel):
    """Application health response."""

    status: str
    model_loaded: bool
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Information about the currently loaded model."""

    model_name: str
    model_path: str
    features: list[str]
    threshold: float
