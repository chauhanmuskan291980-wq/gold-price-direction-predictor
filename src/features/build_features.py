from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target",
]


FEATURE_COLUMNS = [
    "return_1",
    "ma_gap",
    "volatility_10",
    "candle_body_ratio",
    "rsi_14",
]


def load_processed_data(input_path: Path) -> pd.DataFrame:
    """Load the processed Gold dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Processed data file was not found: {input_path}")

    data = pd.read_csv(input_path)

    if data.empty:
        raise ValueError("The processed dataset is empty.")

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Validate that all required columns are present."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate the Relative Strength Index."""
    price_change = close.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_loss = losses.rolling(
        window=period,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.fillna(50)

    return rsi


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create technical features for model training."""
    validate_columns(data)

    featured = data.copy()

    featured["timestamp"] = pd.to_datetime(
        featured["timestamp"],
        utc=True,
        errors="coerce",
    )

    featured = featured.sort_values("timestamp")
    featured = featured.reset_index(drop=True)

    # Percentage change from the previous hourly close.
    featured["return_1"] = featured["close"].pct_change()

    # Difference between short and long moving averages.
    short_moving_average = featured["close"].rolling(window=5).mean()

    long_moving_average = featured["close"].rolling(window=20).mean()

    featured["ma_gap"] = (
        short_moving_average - long_moving_average
    ) / long_moving_average

    # Standard deviation of recent hourly returns.
    featured["volatility_10"] = featured["return_1"].rolling(window=10).std()

    # Size of candle body relative to full candle range.
    candle_range = featured["high"] - featured["low"]

    candle_body = (featured["close"] - featured["open"]).abs()

    featured["candle_body_ratio"] = np.where(
        candle_range > 0,
        candle_body / candle_range,
        0.0,
    )

    featured["rsi_14"] = calculate_rsi(
        featured["close"],
        period=14,
    )

    featured = featured.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    featured = featured.dropna(
        subset=FEATURE_COLUMNS + ["target"],
    )

    featured = featured.reset_index(drop=True)

    if featured.empty:
        raise ValueError("No valid rows remained after feature engineering.")

    return featured


def save_feature_data(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the feature dataset to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )


def print_feature_summary(data: pd.DataFrame) -> None:
    """Print information about engineered features."""
    print("\nFeature engineering summary")
    print("---------------------------")
    print(f"Rows: {len(data):,}")
    print(f"Start: {data['timestamp'].min()}")
    print(f"End: {data['timestamp'].max()}")

    print("\nFeatures created:")

    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create machine-learning features from processed Gold hourly data."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/gold_processed.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gold_features.csv"),
    )

    return parser.parse_args()


def main() -> None:
    """Run the feature-engineering pipeline."""
    args = parse_arguments()

    print(f"Loading processed data from: {args.input}")

    processed_data = load_processed_data(args.input)

    print(f"Input rows: {len(processed_data):,}")

    feature_data = build_features(processed_data)

    save_feature_data(
        feature_data,
        args.output,
    )

    print_feature_summary(feature_data)

    print(f"\nFeature data saved to: {args.output}")


if __name__ == "__main__":
    main()
