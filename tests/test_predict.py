from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS
from src.models.predict import (
    load_latest_feature_row,
    load_model,
    predict_direction,
    result_to_dictionary,
)
from src.models.train import (
    build_model_pipeline,
)


def create_feature_data(
    rows: int = 100,
) -> pd.DataFrame:
    """Create deterministic feature data for tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
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


def create_fitted_model():
    """Create a fitted model pipeline for prediction tests."""
    data = create_feature_data()

    model = build_model_pipeline(
        random_state=42,
        max_iterations=1_000,
    )

    model.fit(
        data[FEATURE_COLUMNS],
        data["target"],
    )

    return model


def test_predict_direction_returns_valid_result() -> None:
    """Prediction should contain probabilities and direction."""
    model = create_fitted_model()
    data = create_feature_data()

    feature_row = data.iloc[[-1]][
        FEATURE_COLUMNS
    ]

    result = predict_direction(
        model=model,
        features=feature_row,
        threshold=0.50,
    )

    assert result.predicted_class in {0, 1}
    assert result.direction in {
        "up",
        "down_or_flat",
    }

    assert 0 <= result.probability_up <= 1
    assert 0 <= result.probability_down <= 1
    assert 0 <= result.confidence <= 1

    assert (
        result.probability_up
        + result.probability_down
    ) == pytest.approx(1.0)


def test_prediction_requires_one_row() -> None:
    """Prediction should reject multiple feature rows."""
    model = create_fitted_model()
    data = create_feature_data()

    feature_rows = data.iloc[:2][
        FEATURE_COLUMNS
    ]

    with pytest.raises(
        ValueError,
        match="Exactly one feature row",
    ):
        predict_direction(
            model=model,
            features=feature_rows,
        )


def test_prediction_rejects_missing_feature() -> None:
    """Prediction should reject incomplete feature input."""
    model = create_fitted_model()
    data = create_feature_data()

    feature_row = data.iloc[[-1]][
        FEATURE_COLUMNS
    ].drop(columns=["rsi_14"])

    with pytest.raises(
        ValueError,
        match="missing features",
    ):
        predict_direction(
            model=model,
            features=feature_row,
        )


def test_invalid_threshold_is_rejected() -> None:
    """Threshold must be between zero and one."""
    model = create_fitted_model()
    data = create_feature_data()

    feature_row = data.iloc[[-1]][
        FEATURE_COLUMNS
    ]

    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        predict_direction(
            model=model,
            features=feature_row,
            threshold=1.0,
        )


def test_load_latest_feature_row(
    tmp_path: Path,
) -> None:
    """Latest feature row should be loaded from CSV."""
    data = create_feature_data()

    output_path = tmp_path / "features.csv"

    data.to_csv(
        output_path,
        index=False,
    )

    latest_features, timestamp = (
        load_latest_feature_row(output_path)
    )

    assert len(latest_features) == 1

    assert latest_features.columns.tolist() == (
        FEATURE_COLUMNS
    )

    assert timestamp is not None


def test_result_to_dictionary() -> None:
    """Prediction result should be serializable."""
    model = create_fitted_model()
    data = create_feature_data()

    feature_row = data.iloc[[-1]][
        FEATURE_COLUMNS
    ]

    result = predict_direction(
        model=model,
        features=feature_row,
    )

    output = result_to_dictionary(
        result,
        timestamp="2026-01-01T00:00:00Z",
    )

    assert output["direction"] in {
        "up",
        "down_or_flat",
    }

    assert "timestamp" in output


def test_load_missing_model() -> None:
    """Missing model artifacts should raise an error."""
    missing_path = Path(
        "artifacts/missing-model.joblib"
    )

    with pytest.raises(FileNotFoundError):
        load_model(missing_path)