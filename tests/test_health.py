from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    """Health endpoint should report API and model status."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] in {
        "healthy",
        "degraded",
    }
    assert isinstance(
        data["model_loaded"],
        bool,
    )
    assert isinstance(
        data["timestamp"],
        str,
    )

    # Confirm that the returned timestamp is valid ISO-8601.
    datetime.fromisoformat(
        data["timestamp"].replace(
            "Z",
            "+00:00",
        )
    )
