from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.build_features import FEATURE_COLUMNS
from src.models.train import (
    build_model_pipeline,
    chronological_train_test_split,
    evaluate_model,
    load_feature_data,
    prepare_training_data,
    save_metadata,
    save_model,
)


def create_training_data(
    rows: int = 100,
) -> pd.DataFrame:
    """Create deterministic sample feature data."""
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "return_1": [
                ((index % 7) - 3) / 1_000
                for index in range(rows)
            ],
            "ma_gap": [
                ((index % 9) - 4) / 1_000
                for index in range(rows)
            ],
            "volatility_10": [
                0.001 + (index % 5) / 10_000
                for index in range(rows)
            ],
            "candle_body_ratio": [
                0.2 + (index % 6) / 10
                for index in range(rows)
            ],
            "rsi_14": [
                35.0 + (index % 30)
                for index in range(rows)
            ],
            "target": [
                index % 2
                for index in range(rows)
            ],
        }
    )


def test_prepare_training_data_sorts_timestamps() -> None:
    """Training data should be chronological."""
    data = create_training_data().iloc[::-1]

    result = prepare_training_data(data)

    assert result["timestamp"].is_monotonic_increasing


def test_chronological_split_preserves_order() -> None:
    """Training observations must occur before test data."""
    data = prepare_training_data(
        create_training_data()
    )

    split = chronological_train_test_split(
        data,
        train_ratio=0.80,
    )

    assert len(split.x_train) == 80
    assert len(split.x_test) == 20

    assert (
        split.train_timestamps.max()
        < split.test_timestamps.min()
    )


def test_split_contains_expected_features() -> None:
    """Model input should contain only feature columns."""
    data = prepare_training_data(
        create_training_data()
    )

    split = chronological_train_test_split(
        data,
        train_ratio=0.80,
    )

    assert split.x_train.columns.tolist() == (
        FEATURE_COLUMNS
    )

    assert split.x_test.columns.tolist() == (
        FEATURE_COLUMNS
    )


def test_invalid_train_ratio_is_rejected() -> None:
    """Invalid split ratios should raise an error."""
    data = prepare_training_data(
        create_training_data()
    )

    with pytest.raises(
        ValueError,
        match="train_ratio",
    ):
        chronological_train_test_split(
            data,
            train_ratio=1.0,
        )


def test_build_model_returns_pipeline() -> None:
    """Training model should include a complete pipeline."""
    model = build_model_pipeline(
        random_state=42,
        max_iterations=1_000,
    )

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_training_and_evaluation() -> None:
    """Model should train and produce valid metrics."""
    data = prepare_training_data(
        create_training_data()
    )

    split = chronological_train_test_split(
        data,
        train_ratio=0.80,
    )

    model = build_model_pipeline(
        random_state=42,
        max_iterations=1_000,
    )

    model.fit(
        split.x_train,
        split.y_train,
    )

    metrics = evaluate_model(
        model,
        split,
        classification_threshold=0.50,
    )

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1

    assert len(metrics["confusion_matrix"]) == 2


def test_save_model(
    tmp_path: Path,
) -> None:
    """A fitted model should be serialized."""
    data = prepare_training_data(
        create_training_data()
    )

    split = chronological_train_test_split(
        data,
        train_ratio=0.80,
    )

    model = build_model_pipeline(
        random_state=42,
        max_iterations=1_000,
    )

    model.fit(
        split.x_train,
        split.y_train,
    )

    output_path = tmp_path / "model.joblib"

    save_model(
        model,
        output_path,
    )

    assert output_path.exists()


def test_save_metadata(
    tmp_path: Path,
) -> None:
    """Training metadata should be written as JSON."""
    output_path = tmp_path / "metadata.json"

    metadata = {
        "model_name": "LogisticRegression",
        "accuracy": 0.55,
    }

    save_metadata(
        metadata,
        output_path,
    )

    assert output_path.exists()


def test_load_missing_feature_file() -> None:
    """A missing feature dataset should be rejected."""
    missing_path = Path(
        "data/processed/missing-features.csv"
    )

    with pytest.raises(FileNotFoundError):
        load_feature_data(missing_path)