from pathlib import Path

import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from src.features.build_features import FEATURE_COLUMNS
from src.models.model_factory import build_models


def create_training_data(
    rows: int = 100,
) -> pd.DataFrame:
    """Create deterministic sample training data."""
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


def split_training_data(
    data: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create a chronological train-test split for tests."""
    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    prepared_data = (
        data.sort_values("timestamp")
        .dropna(
            subset=[
                *FEATURE_COLUMNS,
                "target",
            ]
        )
        .reset_index(drop=True)
    )

    split_index = int(
        len(prepared_data) * train_ratio
    )

    x_train = prepared_data.iloc[
        :split_index
    ][FEATURE_COLUMNS]

    x_test = prepared_data.iloc[
        split_index:
    ][FEATURE_COLUMNS]

    y_train = prepared_data.iloc[
        :split_index
    ]["target"]

    y_test = prepared_data.iloc[
        split_index:
    ]["target"]

    return x_train, x_test, y_train, y_test


def test_training_data_can_be_sorted_by_timestamp() -> None:
    """Training data should be sortable chronologically."""
    data = create_training_data().iloc[::-1]

    sorted_data = (
        data.sort_values("timestamp")
        .reset_index(drop=True)
    )

    assert sorted_data[
        "timestamp"
    ].is_monotonic_increasing


def test_chronological_split_preserves_order() -> None:
    """Training rows must occur before test rows."""
    data = create_training_data()

    x_train, x_test, _, _ = split_training_data(
        data,
        train_ratio=0.80,
    )

    assert len(x_train) == 80
    assert len(x_test) == 20

    train_end_timestamp = data.iloc[
        x_train.index[-1]
    ]["timestamp"]

    test_start_timestamp = data.iloc[
        x_test.index[0]
    ]["timestamp"]

    assert (
        train_end_timestamp
        < test_start_timestamp
    )


def test_split_contains_expected_features() -> None:
    """Model input should contain feature columns only."""
    data = create_training_data()

    x_train, x_test, _, _ = split_training_data(
        data,
        train_ratio=0.80,
    )

    assert (
        x_train.columns.tolist()
        == FEATURE_COLUMNS
    )

    assert (
        x_test.columns.tolist()
        == FEATURE_COLUMNS
    )


def test_invalid_train_ratio_is_rejected() -> None:
    """Invalid split ratios should raise an error."""
    data = create_training_data()

    with pytest.raises(
        ValueError,
        match="train_ratio",
    ):
        split_training_data(
            data,
            train_ratio=1.0,
        )


def test_build_models_returns_expected_models() -> None:
    """Factory should return all configured models."""
    models = build_models()

    assert isinstance(models, dict)

    assert set(models.keys()) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


def test_logistic_regression_is_pipeline() -> None:
    """Logistic regression should use a pipeline."""
    models = build_models()

    assert isinstance(
        models["logistic_regression"],
        Pipeline,
    )


from sklearn.base import is_classifier


def test_all_models_are_classifiers() -> None:
    """Every configured model should be recognized as a classifier."""
    models = build_models()

    for model in models.values():
        assert is_classifier(model)


def test_all_models_can_be_trained() -> None:
    """Every configured model should fit successfully."""
    data = create_training_data()

    x_train, x_test, y_train, _ = (
        split_training_data(data)
    )

    models = build_models()

    for model in models.values():
        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(x_test)

        assert len(predictions) == len(x_test)

        assert set(predictions).issubset(
            {0, 1}
        )


def test_all_models_return_probabilities() -> None:
    """Every model should return class probabilities."""
    data = create_training_data()

    x_train, x_test, y_train, _ = (
        split_training_data(data)
    )

    models = build_models()

    for model in models.values():
        model.fit(
            x_train,
            y_train,
        )

        probabilities = model.predict_proba(
            x_test
        )

        assert probabilities.shape == (
            len(x_test),
            2,
        )

        assert (
            probabilities >= 0
        ).all()

        assert (
            probabilities <= 1
        ).all()

