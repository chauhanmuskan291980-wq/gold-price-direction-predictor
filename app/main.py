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
    Load the trained model once when the API starts.

    The loaded model is reused for every prediction request.
    """
    del app

    try:
        model_service.load()
    except FileNotFoundError:
        # The health endpoint will expose that the model
        # is unavailable instead of preventing API startup.
        pass

    yield


app = FastAPI(
    title="Gold Price Direction Predictor API",
    description=(
        "Predict the direction of the next hourly XAUUSD "
        "candle using a trained Logistic Regression pipeline."
    ),
    version="1.0.0",
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
        status=("healthy" if model_service.is_loaded else "degraded"),
        model_loaded=model_service.is_loaded,
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    """Return information about the trained model."""
    return ModelInfoResponse(**model_service.model_info())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict the next hourly Gold candle direction."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction model is unavailable. "
                "Train the model before starting the API."
            ),
        )

    try:
        result = model_service.predict(request)
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

    return PredictionResponse(
        predicted_class=int(result["predicted_class"]),
        direction=str(result["direction"]),
        probability_up=float(result["probability_up"]),
        probability_down=float(result["probability_down"]),
        confidence=float(result["confidence"]),
        threshold=float(result["threshold"]),
    )
