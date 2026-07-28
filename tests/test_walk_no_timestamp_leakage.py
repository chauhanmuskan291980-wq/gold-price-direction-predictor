from pathlib import Path

import pandas as pd

from src.evaluation.walk_forward_split import generate_walk_forward_folds
from src.models.walk_forward import prepare_fold_data

DATA_PATH = Path("data/processed/gold_features.csv")


def test_actual_fold_data_has_no_timestamp_leakage() -> None:
    """No test timestamp may appear in its fold's training data."""

    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    folds = generate_walk_forward_folds(
        total_rows=len(data),
        train_window=1500,
        test_window=250,
        step_size=250,
        purge_gap=1,
    )

    assert folds, "Expected at least one walk-forward fold."

    for fold in folds:
        # Correct return order:
        # X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = prepare_fold_data(
            data=data,
            fold=fold,
        )

        train_timestamps = X_train.index
        test_timestamps = X_test.index

        # No actual test timestamp may exist in training.
        overlap = train_timestamps.intersection(test_timestamps)

        assert overlap.empty, (
            f"Timestamp leakage detected in fold {fold}: "
            f"{len(overlap)} overlapping timestamps."
        )

        # Every training row must occur before every testing row.
        assert train_timestamps.max() < test_timestamps.min(), (
            f"Future-data leakage detected in fold {fold}: "
            f"last training timestamp={train_timestamps.max()}, "
            f"first testing timestamp={test_timestamps.min()}."
        )

        # Features and labels must remain aligned.
        assert X_train.index.equals(y_train.index)
        assert X_test.index.equals(y_test.index)