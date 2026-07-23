import pandas as pd

from src.models.train import (
    chronological_split,
    load_training_data,
)


def test_chronological_split_prevents_lookahead() -> None:
    data = load_training_data(
        "data/processed/gold_features.csv"
    )

    X_train, X_test, _, _ = chronological_split(
        data
    )

    train_end = pd.to_datetime(
        data.loc[X_train.index, "timestamp"],
        utc=True,
    ).max()

    test_start = pd.to_datetime(
        data.loc[X_test.index, "timestamp"],
        utc=True,
    ).min()

    assert train_end < test_start


def test_chronological_split_purges_boundary_row() -> None:
    data = load_training_data(
        "data/processed/gold_features.csv"
    )

    X_train, X_test, _, _ = chronological_split(
        data
    )

    last_train_index = int(X_train.index.max())
    first_test_index = int(X_test.index.min())

    purged_rows = (
        first_test_index
        - last_train_index
        - 1
    )

    assert purged_rows == 1