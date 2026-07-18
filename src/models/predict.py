from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.build_features import FEATURE_COLUMNS


@dataclass(frozen=True)
class PredictionResult:
    """Structured model prediction result."""

    predicted_class: int
    direction: str
    probability_up: float
    probability_down: float
    confidence: float
    threshold: float


def load_model(model_path: Path) -> Pipeline:
    """Load the trained machine-learning pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError("Loaded model is not a scikit-learn Pipeline.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support probability predictions.")

    return model


def validate_feature_data(data: pd.DataFrame) -> None:
    """Validate the feature data used for prediction."""
    missing_columns = [
        feature for feature in FEATURE_COLUMNS if feature not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Prediction data is missing features: {missing_columns}")

    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Prediction data contains missing feature values.")

    non_numeric_columns = [
        feature
        for feature in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(data[feature])
    ]

    if non_numeric_columns:
        raise ValueError(
            "Prediction features must be numeric. "
            f"Invalid columns: {non_numeric_columns}"
        )


def predict_direction(
    model: Pipeline,
    features: pd.DataFrame,
    threshold: float = 0.50,
) -> PredictionResult:
    """Predict the next-hour Gold price direction."""
    if len(features) != 1:
        raise ValueError("Exactly one feature row is required for prediction.")

    if not 0 < threshold < 1:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    validate_feature_data(features)

    model_input = features[FEATURE_COLUMNS].copy()

    probabilities = model.predict_proba(model_input)[0]

    probability_down = float(probabilities[0])
    probability_up = float(probabilities[1])

    predicted_class = int(probability_up >= threshold)

    direction = "up" if predicted_class == 1 else "down_or_flat"

    confidence = probability_up if predicted_class == 1 else probability_down

    return PredictionResult(
        predicted_class=predicted_class,
        direction=direction,
        probability_up=probability_up,
        probability_down=probability_down,
        confidence=confidence,
        threshold=threshold,
    )


def load_latest_feature_row(
    feature_data_path: Path,
) -> tuple[pd.DataFrame, str | None]:
    """Load the latest available engineered-feature row."""
    if not feature_data_path.exists():
        raise FileNotFoundError(f"Feature dataset was not found: {feature_data_path}")

    data = pd.read_csv(feature_data_path)

    if data.empty:
        raise ValueError("The feature dataset is empty.")

    validate_feature_data(data)

    timestamp: str | None = None

    if "timestamp" in data.columns:
        timestamp = str(data.iloc[-1]["timestamp"])

    latest_features = data.iloc[[-1]][FEATURE_COLUMNS].copy()

    return latest_features, timestamp


def result_to_dictionary(
    result: PredictionResult,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Convert a prediction result to a serializable dictionary."""
    output: dict[str, Any] = asdict(result)

    if timestamp is not None:
        output["timestamp"] = timestamp

    return output


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict next-hour Gold price direction using the latest feature row."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/gold_direction_pipeline.joblib"),
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/gold_features.csv"),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:
    """Run a prediction using the latest feature row."""
    args = parse_arguments()

    model = load_model(args.model)

    latest_features, timestamp = load_latest_feature_row(args.features)

    result = predict_direction(
        model=model,
        features=latest_features,
        threshold=args.threshold,
    )

    output = result_to_dictionary(
        result,
        timestamp,
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
