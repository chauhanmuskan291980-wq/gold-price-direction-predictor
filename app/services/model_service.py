from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import PredictionRequest
from src.features.build_features import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("artifacts/gold_direction_pipeline.joblib")
DEFAULT_THRESHOLD = 0.50


class ModelService:
    """Load the model artifact and generate predictions."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Prediction threshold must be between 0 and 1.")

        self.model_path = model_path
        self.threshold = threshold
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the trained pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        self._model = joblib.load(self.model_path)

    def predict(
        self,
        request: PredictionRequest,
    ) -> dict[str, float | int | str]:
        """Generate a direction prediction."""
        if self._model is None:
            raise RuntimeError("The prediction model has not been loaded.")

        feature_values = request.model_dump()

        missing_features = [
            feature for feature in FEATURE_COLUMNS if feature not in feature_values
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: " + ", ".join(missing_features)
            )

        features = pd.DataFrame(
            [{feature: feature_values[feature] for feature in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )

        probability_values = self._model.predict_proba(features)[0]

        probability_down = float(probability_values[0])
        probability_up = float(probability_values[1])

        predicted_class = int(probability_up >= self.threshold)

        direction = "up" if predicted_class == 1 else "down_or_flat"

        confidence = float(np.max(probability_values))

        return {
            "predicted_class": predicted_class,
            "direction": direction,
            "probability_up": probability_up,
            "probability_down": probability_down,
            "confidence": confidence,
            "threshold": self.threshold,
        }

    def model_info(self) -> dict[str, Any]:
        """Return metadata about the active model."""
        return {
            "model_name": "LogisticRegression",
            "model_path": str(self.model_path),
            "features": FEATURE_COLUMNS,
            "threshold": self.threshold,
        }
