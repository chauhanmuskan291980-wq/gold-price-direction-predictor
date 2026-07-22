from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Load all trained models once when the application starts.

    The same model service instance is reused for every request.
    """

    model_service = ModelService()

    try:
        model_service.load()

    except FileNotFoundError:
        # Allow the API to start in a degraded state.
        # Endpoints that require models will return HTTP 503.
        pass

    app.state.model_service = model_service

    yield

    # Optional cleanup can be added here later.


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

app.include_router(router)