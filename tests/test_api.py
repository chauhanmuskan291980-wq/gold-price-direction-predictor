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
        "version": "2.0.0",
        "documentation": "/docs",
    }


def test_prediction_endpoint() -> None:
    """Comparison endpoint should return model predictions."""
    with TestClient(app) as client:
        payload = {
        "return_1": 0.001,
        "ma_gap": 0.002,
        "volatility_10": 0.0015,
        "candle_body_ratio": 0.5,
        "rsi_14": 55.0,
        }

        response = client.post(
        "/predict/compare",
        json=payload,
      )

        assert response.status_code == 200

        data = response.json()

        assert isinstance(data, dict)


def test_prediction_rejects_invalid_rsi() -> None:
    """Prediction endpoint should reject an invalid RSI."""
    payload = {
        "return_1": 0.001,
        "ma_gap": 0.002,
        "volatility_10": 0.0015,
        "candle_body_ratio": 0.5,
        # rsi_14 is intentionally missing
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict/compare",
            json=payload,
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
            "/predict/compare",
            json=request_body,
        )

    assert response.status_code == 422