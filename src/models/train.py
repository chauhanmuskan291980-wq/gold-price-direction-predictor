from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.model_factory import build_models

FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]

TARGET_COLUMN = "target"

ARTIFACT_DIR = Path("artifacts/models")


def load_training_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return data.dropna(
        subset=required_columns
    ).copy()


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.80,
    purge_rows: int = 1,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split time-series data chronologically.

    Rows immediately before the test set are excluded from training
    to reduce the risk of look-ahead leakage at the split boundary.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if purge_rows < 0:
        raise ValueError(
            "purge_rows must be zero or greater."
        )

    split_index = int(len(data) * train_ratio)
    train_end_index = split_index - purge_rows

    if train_end_index <= 0:
        raise ValueError(
            "Not enough training rows after applying purge_rows."
        )

    if split_index >= len(data):
        raise ValueError(
            "The testing dataset cannot be empty."
        )

    train_data = data.iloc[:train_end_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURE_COLUMNS].copy()
    y_train = train_data[TARGET_COLUMN].copy()

    X_test = test_data[FEATURE_COLUMNS].copy()
    y_test = test_data[TARGET_COLUMN].copy()

    return X_train, X_test, y_train, y_test

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    models = build_models()
    trained_models: dict[str, Any] = {}

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        model_path = (
            ARTIFACT_DIR / f"{model_name}.joblib"
        )

        joblib.dump(model, model_path)

        trained_models[model_name] = model

        print(f"Saved model to: {model_path}")

    return trained_models


def main() -> None:
    data = load_training_data(
        "data/processed/gold_features.csv"
    )

    X_train, X_test, y_train, _y_test = (
        chronological_split(data)
    )

    train_all_models(
        X_train=X_train,
        y_train=y_train,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")


if __name__ == "__main__":
    main()