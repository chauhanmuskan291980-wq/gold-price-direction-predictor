from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load raw hourly Gold data from a CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data file was not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    if data.empty:
        raise ValueError("The raw dataset is empty.")

    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    """Ensure the dataset contains all required OHLCV columns."""
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )


def clean_ohlc_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate hourly OHLCV market data.

    The function:
    - validates required columns
    - parses UTC timestamps
    - converts OHLCV columns to numeric values
    - removes missing values
    - removes duplicate timestamps
    - removes invalid prices
    - sorts candles chronologically
    """
    validate_required_columns(data)

    cleaned = data[REQUIRED_COLUMNS].copy()

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=REQUIRED_COLUMNS,
    )

    cleaned = cleaned.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    positive_price_mask = (
        cleaned[price_columns] > 0
    ).all(axis=1)

    non_negative_volume_mask = (
        cleaned["volume"] >= 0
    )

    valid_high_mask = (
        cleaned["high"]
        >= cleaned[["open", "low", "close"]].max(axis=1)
    )

    valid_low_mask = (
        cleaned["low"]
        <= cleaned[["open", "high", "close"]].min(axis=1)
    )

    cleaned = cleaned[
        positive_price_mask
        & non_negative_volume_mask
        & valid_high_mask
        & valid_low_mask
    ]

    cleaned = cleaned.sort_values("timestamp")
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(
            "No valid rows remained after cleaning."
        )

    return cleaned


def create_direction_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the next-hour price direction target.

    target = 1:
        the next hourly close is greater than the current close.

    target = 0:
        the next hourly close is equal to or lower than the current close.
    """
    if "close" not in data.columns:
        raise ValueError(
            "The dataset must contain a close column."
        )

    if len(data) < 2:
        raise ValueError(
            "At least two rows are required to create the target."
        )

    result = data.copy()

    result["next_close"] = result["close"].shift(-1)

    result = result.dropna(
        subset=["next_close"],
    )

    result["target"] = (
        result["next_close"] > result["close"]
    ).astype("int8")

    result = result.drop(
        columns=["next_close"],
    )

    result = result.reset_index(drop=True)

    return result


def save_processed_data(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the processed dataset to a CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )


def print_dataset_summary(data: pd.DataFrame) -> None:
    """Print useful information about the processed dataset."""
    class_counts = data["target"].value_counts().sort_index()

    down_count = int(class_counts.get(0, 0))
    up_count = int(class_counts.get(1, 0))

    print("\nProcessed dataset summary")
    print("-------------------------")
    print(f"Rows: {len(data):,}")
    print(f"Start: {data['timestamp'].min()}")
    print(f"End: {data['timestamp'].max()}")
    print(f"Down or unchanged candles: {down_count:,}")
    print(f"Up candles: {up_count:,}")

    if len(data) > 0:
        up_percentage = (
            up_count / len(data)
        ) * 100

        print(
            f"Up-candle percentage: "
            f"{up_percentage:.2f}%"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean hourly Gold OHLCV data and create "
            "the next-hour direction target."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/gold_hourly.csv"),
        help="Path to the raw hourly Gold CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/gold_processed.csv"
        ),
        help="Path where processed data will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete preprocessing pipeline."""
    args = parse_arguments()

    print(f"Loading raw data from: {args.input}")

    raw_data = load_raw_data(args.input)

    print(f"Raw rows: {len(raw_data):,}")

    cleaned_data = clean_ohlc_data(raw_data)

    print(f"Valid rows after cleaning: {len(cleaned_data):,}")

    processed_data = create_direction_target(
        cleaned_data
    )

    save_processed_data(
        processed_data,
        args.output,
    )

    print_dataset_summary(processed_data)

    print(f"\nProcessed data saved to: {args.output}")


if __name__ == "__main__":
    main()