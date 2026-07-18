from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the current health status of the service."""
    return {
        "status": "healthy",
        "service": "gold-price-direction-predictor",
    }