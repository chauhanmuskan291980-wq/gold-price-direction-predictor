from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint() -> None:
    """Root endpoint should return API information."""
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Gold Price Direction Predictor API",
        "documentation": "/docs",
    }


def test_model_info_endpoint() -> None:
    """Model information endpoint should return metadata."""
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "LogisticRegression"
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0
    assert data["threshold"] == 0.5


def test_prediction_endpoint() -> None:
    """Prediction endpoint should return valid probabilities."""
    request_body = {
        "return_1": 0.0012,
        "ma_gap": -0.0021,
        "volatility_10": 0.0045,
        "candle_body_ratio": 0.62,
        "rsi_14": 54.3,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=request_body,
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] in {0, 1}
    assert data["direction"] in {
        "up",
        "down_or_flat",
    }

    assert 0 <= data["probability_up"] <= 1
    assert 0 <= data["probability_down"] <= 1
    assert 0 <= data["confidence"] <= 1
    assert data["threshold"] == 0.5

    probability_total = (
        data["probability_up"]
        + data["probability_down"]
    )

    assert abs(probability_total - 1.0) < 1e-6


def test_prediction_rejects_invalid_rsi() -> None:
    """Prediction endpoint should reject an invalid RSI."""
    request_body = {
        "return_1": 0.0012,
        "ma_gap": -0.0021,
        "volatility_10": 0.0045,
        "candle_body_ratio": 0.62,
        "rsi_14": 150,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=request_body,
        )

    assert response.status_code == 422


def test_prediction_rejects_missing_feature() -> None:
    """Prediction endpoint should reject incomplete input."""
    request_body = {
        "return_1": 0.0012,
        "ma_gap": -0.0021,
        "volatility_10": 0.0045,
        "rsi_14": 54.3,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=request_body,
        )

    assert response.status_code == 422