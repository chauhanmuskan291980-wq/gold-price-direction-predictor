from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.model_service import ModelService


model_service = ModelService()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Load all trained models once when the API starts.

    The loaded models are reused for every prediction request.
    """

    try:
        model_service.load()
        app.state.model_service = model_service

    except FileNotFoundError:
        # Allow the API to start even when model artifacts
        # are unavailable. The health endpoint will report
        # the degraded state.
        app.state.model_service = model_service

    yield


app = FastAPI(
    title="Gold Price Direction Predictor API",
    description=(
        "Predict the direction of the next hourly Gold candle "
        "using Logistic Regression, Random Forest, and "
        "Gradient Boosting classifiers."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["General"],
)
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": "Gold Price Direction Predictor API",
        "version": "2.0.0",
        "documentation": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health() -> HealthResponse:
    """Return API and model health information."""

    return HealthResponse(
        status=(
            "healthy"
            if model_service.is_loaded
            else "degraded"
        ),
        model_loaded=model_service.is_loaded,
        loaded_model_count=model_service.loaded_model_count,
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    """Return information about all trained models."""

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction models are unavailable. "
                "Train the models before requesting model information."
            ),
        )

    return ModelInfoResponse(
        **model_service.model_info()
    )


@app.post(
    "/predict/compare",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def compare_predictions(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Return predictions from all trained models
    and calculate the majority-vote result.
    """

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction models are unavailable. "
                "Train the models before starting the API."
            ),
        )

    try:
        result = model_service.predict_all(request)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return PredictionResponse(**result)