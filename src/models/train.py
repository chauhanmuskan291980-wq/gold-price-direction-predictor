from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS

TARGET_COLUMN = "target"
TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""

    input_path: Path = Path("data/processed/gold_features.csv")
    model_output_path: Path = Path("artifacts/gold_direction_pipeline.joblib")
    metadata_output_path: Path = Path("artifacts/training_metadata.json")
    train_ratio: float = 0.80
    classification_threshold: float = 0.50
    random_state: int = 42
    max_iterations: int = 1_000


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train and test datasets."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_timestamps: pd.Series
    test_timestamps: pd.Series


def load_feature_data(input_path: Path) -> pd.DataFrame:
    """Load the feature-engineered Gold dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature dataset was not found: {input_path}")

    data = pd.read_csv(input_path)

    if data.empty:
        raise ValueError("The feature dataset is empty.")

    return data


def validate_training_data(data: pd.DataFrame) -> None:
    """Validate columns, feature values, and target classes."""
    required_columns = [
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {missing_columns}"
        )

    if data[required_columns].isna().any().any():
        raise ValueError("Training data contains missing values.")

    invalid_targets = set(data[TARGET_COLUMN].unique()) - {0, 1}

    if invalid_targets:
        raise ValueError(
            "Target column must contain only 0 and 1. "
            f"Invalid values: {sorted(invalid_targets)}"
        )

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data must contain both target classes.")

    if len(data) < 50:
        raise ValueError("At least 50 rows are required for model training.")


def prepare_training_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Parse timestamps and arrange data chronologically."""
    prepared = data.copy()

    prepared[TIMESTAMP_COLUMN] = pd.to_datetime(
        prepared[TIMESTAMP_COLUMN],
        utc=True,
        errors="coerce",
    )

    if prepared[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Training data contains invalid timestamps.")

    prepared = prepared.sort_values(TIMESTAMP_COLUMN)

    prepared = prepared.drop_duplicates(
        subset=[TIMESTAMP_COLUMN],
        keep="last",
    )

    prepared = prepared.reset_index(drop=True)

    validate_training_data(prepared)

    return prepared


def chronological_train_test_split(
    data: pd.DataFrame,
    train_ratio: float,
) -> DatasetSplit:
    """
    Split time-series data without shuffling.

    Earlier observations are used for training.
    Later observations are reserved for testing.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(data) * train_ratio)

    if split_index <= 1 or split_index >= len(data):
        raise ValueError("The train-test split produced an empty dataset.")

    train_data = data.iloc[: split_index - 1].copy()
    test_data = data.iloc[split_index:].copy()

    x_train = train_data[FEATURE_COLUMNS].copy()
    x_test = test_data[FEATURE_COLUMNS].copy()

    y_train = train_data[TARGET_COLUMN].astype(int)
    y_test = test_data[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training split must contain both target classes.")

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_timestamps=train_data[TIMESTAMP_COLUMN].copy(),
        test_timestamps=test_data[TIMESTAMP_COLUMN].copy(),
    )


def build_model_pipeline(
    random_state: int,
    max_iterations: int,
) -> Pipeline:
    """
    Build preprocessing and classification steps.

    StandardScaler is fitted only on training data because
    it is inside the scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=max_iterations,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    split: DatasetSplit,
    classification_threshold: float,
) -> dict[str, Any]:
    """Evaluate predictions on unseen chronological data."""
    if not 0 < classification_threshold < 1:
        raise ValueError("classification_threshold must be between 0 and 1.")

    positive_probabilities = model.predict_proba(split.x_test)[:, 1]

    predictions = (positive_probabilities >= classification_threshold).astype(int)

    report = classification_report(
        split.y_test,
        predictions,
        labels=[0, 1],
        target_names=["down_or_flat", "up"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        split.y_test,
        predictions,
        labels=[0, 1],
    )

    metrics: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                split.y_test,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                split.y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                split.y_test,
                positive_probabilities,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    return metrics


def save_model(
    model: Pipeline,
    output_path: Path,
) -> None:
    """Save the complete fitted model pipeline."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )


def save_metadata(
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """Save training metadata and evaluation results."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def print_evaluation(
    metrics: dict[str, Any],
) -> None:
    """Print readable model evaluation results."""
    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 score:          {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")

    matrix = metrics["confusion_matrix"]

    print("\nConfusion matrix")
    print("----------------")
    print("                 Predicted 0  Predicted 1")
    print(f"Actual 0         {matrix[0][0]:11d}  {matrix[0][1]:11d}")
    print(f"Actual 1         {matrix[1][0]:11d}  {matrix[1][1]:11d}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a Logistic Regression model for next-hour Gold price direction."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/gold_features.csv"),
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/gold_direction_pipeline.joblib"),
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("artifacts/training_metadata.json"),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete model-training pipeline."""
    args = parse_arguments()

    config = TrainingConfig(
        input_path=args.input,
        model_output_path=args.model_output,
        metadata_output_path=(args.metadata_output),
        train_ratio=args.train_ratio,
        classification_threshold=args.threshold,
    )

    print(f"Loading feature data from: {config.input_path}")

    raw_data = load_feature_data(config.input_path)

    prepared_data = prepare_training_data(raw_data)

    split = chronological_train_test_split(
        prepared_data,
        config.train_ratio,
    )

    print(f"Total rows: {len(prepared_data):,}")
    print(f"Training rows: {len(split.x_train):,}")
    print(f"Testing rows: {len(split.x_test):,}")

    print(
        "Training period: "
        f"{split.train_timestamps.min()} to "
        f"{split.train_timestamps.max()}"
    )

    print(
        "Testing period: "
        f"{split.test_timestamps.min()} to "
        f"{split.test_timestamps.max()}"
    )

    model = build_model_pipeline(
        random_state=config.random_state,
        max_iterations=config.max_iterations,
    )

    print("\nTraining Logistic Regression...")

    model.fit(
        split.x_train,
        split.y_train,
    )

    metrics = evaluate_model(
        model,
        split,
        config.classification_threshold,
    )

    print_evaluation(metrics)

    metadata: dict[str, Any] = {
        "model_name": "LogisticRegression",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "train_ratio": config.train_ratio,
        "classification_threshold": (config.classification_threshold),
        "total_rows": len(prepared_data),
        "training_rows": len(split.x_train),
        "testing_rows": len(split.x_test),
        "training_period": {
            "start": (split.train_timestamps.min().isoformat()),
            "end": (split.train_timestamps.max().isoformat()),
        },
        "testing_period": {
            "start": (split.test_timestamps.min().isoformat()),
            "end": (split.test_timestamps.max().isoformat()),
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "metrics": metrics,
    }

    save_model(
        model,
        config.model_output_path,
    )

    save_metadata(
        metadata,
        config.metadata_output_path,
    )

    print(f"\nModel saved to: {config.model_output_path}")

    print(f"Metadata saved to: {config.metadata_output_path}")


if __name__ == "__main__":
    main()
