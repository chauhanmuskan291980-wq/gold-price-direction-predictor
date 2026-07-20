from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.schemas import PredictionRequest


FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]


class ModelService:
    def __init__(self) -> None:
        self.model_directory = Path("artifacts/models")
        self.metrics_path = Path(
            "artifacts/metrics/model_metrics.json"
        )

        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict[str, float]] = {}

    @property
    def is_loaded(self) -> bool:
        """Return True when at least one model is loaded."""
        return bool(self.models)

    @property
    def loaded_model_count(self) -> int:
        """Return the number of loaded models."""
        return len(self.models)

    def load(self) -> None:
        """Load all trained model artifacts and evaluation metrics."""
        expected_models = {
            "logistic_regression": (
                self.model_directory
                / "logistic_regression.joblib"
            ),
            "random_forest": (
                self.model_directory
                / "random_forest.joblib"
            ),
            "gradient_boosting": (
                self.model_directory
                / "gradient_boosting.joblib"
            ),
        }

        missing_files = [
            str(model_path)
            for model_path in expected_models.values()
            if not model_path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "The following model files are missing: "
                + ", ".join(missing_files)
            )

        self.models = {
            model_name: joblib.load(model_path)
            for model_name, model_path
            in expected_models.items()
        }

        if self.metrics_path.exists():
            with self.metrics_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                self.metrics = json.load(file)
        else:
            self.metrics = {}

    def _request_to_dataframe(
        self,
        request: PredictionRequest,
    ) -> pd.DataFrame:
        """Convert validated request data into model input."""
        data = request.model_dump()

        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in data
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing_features)
            )

        features = pd.DataFrame(
            [
                {
                    column: float(data[column])
                    for column in FEATURE_COLUMNS
                }
            ]
        )

        return features

    @staticmethod
    def _predict_single_model(
        model_name: str,
        model: Any,
        features: pd.DataFrame,
        metrics: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Generate prediction details for one model."""
        predicted_class = int(
            model.predict(features)[0]
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

            probability_down = float(probabilities[0])
            probability_up = float(probabilities[1])
        else:
            probability_up = float(predicted_class)
            probability_down = float(1 - predicted_class)

        confidence = max(
            probability_up,
            probability_down,
        )

        return {
            "model_name": model_name,
            "predicted_class": predicted_class,
            "direction": (
                "UP"
                if predicted_class == 1
                else "DOWN"
            ),
            "probability_up": round(
                probability_up,
                4,
            ),
            "probability_down": round(
                probability_down,
                4,
            ),
            "confidence": round(
                confidence,
                4,
            ),
            "threshold": 0.5,
            "historical_metrics": metrics.get(
                model_name
            ),
        }

    @staticmethod
    def _build_ensemble(
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create majority-vote ensemble output."""
        up_votes = sum(
            prediction["predicted_class"] == 1
            for prediction in predictions
        )

        down_votes = len(predictions) - up_votes

        predicted_class = (
            1
            if up_votes > down_votes
            else 0
        )

        return {
            "method": "majority_vote",
            "predicted_class": predicted_class,
            "direction": (
                "UP"
                if predicted_class == 1
                else "DOWN"
            ),
            "up_votes": up_votes,
            "down_votes": down_votes,
        }

    def _recommended_model(self) -> dict[str, Any]:
        """Select the model with the best balanced accuracy."""
        if not self.metrics:
            return {
                "name": next(iter(self.models)),
                "selection_metric": "balanced_accuracy",
                "score": 0.0,
            }

        eligible_models = {
            model_name: values
            for model_name, values in self.metrics.items()
            if model_name in self.models
            and "balanced_accuracy" in values
        }

        if not eligible_models:
            return {
                "name": next(iter(self.models)),
                "selection_metric": "balanced_accuracy",
                "score": 0.0,
            }

        best_model_name = max(
            eligible_models,
            key=lambda name: eligible_models[name][
                "balanced_accuracy"
            ],
        )

        return {
            "name": best_model_name,
            "selection_metric": "balanced_accuracy",
            "score": float(
                eligible_models[best_model_name][
                    "balanced_accuracy"
                ]
            ),
        }

    def predict_all(
        self,
        request: PredictionRequest,
    ) -> dict[str, Any]:
        """Run prediction using every loaded model."""
        if not self.is_loaded:
            raise RuntimeError(
                "No trained prediction models are loaded."
            )

        features = self._request_to_dataframe(request)

        predictions = [
            self._predict_single_model(
                model_name=model_name,
                model=model,
                features=features,
                metrics=self.metrics,
            )
            for model_name, model in self.models.items()
        ]

        return {
            "predictions": predictions,
            "ensemble": self._build_ensemble(
                predictions
            ),
            "recommended_model": (
                self._recommended_model()
            ),
        }

    def model_info(self) -> dict[str, Any]:
        """Return loaded model and feature information."""
        models = []

        for model_name in self.models:
            models.append(
                {
                    "name": model_name,
                    "artifact_path": str(
                        self.model_directory
                        / f"{model_name}.joblib"
                    ),
                    "loaded": True,
                    "metrics": self.metrics.get(
                        model_name
                    ),
                }
            )

        return {
            "loaded_model_count": (
                self.loaded_model_count
            ),
            "feature_columns": FEATURE_COLUMNS,
            "models": models,
            "recommended_model": (
                self._recommended_model()
                if self.models
                else None
            ),

        }
    
    
    
    