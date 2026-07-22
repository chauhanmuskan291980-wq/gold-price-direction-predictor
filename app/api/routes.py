from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import (
    HealthResponse,
    LatestPredictionMetadata,
    LatestPredictionResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.latest_prediction_service import (
    GOLD_TICKER,
    LatestPredictionError,
    build_latest_feature_row,
    current_utc_timestamp,
    serialize_features,
)
from app.services.model_service import ModelService

router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """Return the model service stored in application state."""

    model_service = getattr(
        request.app.state,
        "model_service",
        None,
    )

    if not isinstance(model_service, ModelService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model service is unavailable.",
        )

    return model_service


@router.get(
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


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health(request: Request) -> HealthResponse:
    """Return API and model health information."""

    model_service = get_model_service(request)

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


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info(request: Request) -> ModelInfoResponse:
    """Return information about all trained models."""

    model_service = get_model_service(request)

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


@router.post(
    "/predict/compare",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def compare_predictions(
    prediction_request: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """
    Return predictions from all trained models and calculate
    the ensemble majority-vote result.
    """

    model_service = get_model_service(request)

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction models are unavailable. "
                "Train the models before starting the API."
            ),
        )

    try:
        result = model_service.predict_all(
            prediction_request
        )

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


@router.get(
    "/predict/latest",
    response_model=LatestPredictionResponse,
    tags=["Prediction"],
)
def predict_latest(
    request: Request,
) -> LatestPredictionResponse:
    """
    Download the latest Gold market data, generate features,
    and return predictions from all trained models.
    """

    model_service = get_model_service(request)

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The prediction models are unavailable. "
                "Train the models before requesting a prediction."
            ),
        )

    try:
        market_timestamp, feature_row = (
            build_latest_feature_row()
        )

        feature_values = serialize_features(
            feature_row
        )

        prediction_request = PredictionRequest(
            **feature_values
        )

        predictions = model_service.predict_all(
            prediction_request
        )

        return LatestPredictionResponse(
            metadata=LatestPredictionMetadata(
                ticker=GOLD_TICKER,
                market_timestamp=market_timestamp.isoformat(),
                generated_at=current_utc_timestamp(),
            ),
            features=feature_values,
            predictions=predictions,
        )

    except LatestPredictionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

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