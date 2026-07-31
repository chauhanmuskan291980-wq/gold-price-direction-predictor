from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation.permutation_null import (
    permute_target,
    summarize_permutation_null,
)
from src.evaluation.walk_forward_split import (
    generate_walk_forward_folds,
)
from src.models.train import (
    TARGET_COLUMN,
    load_training_data,
)
from src.models.walk_forward_grid import (
    CONFIG_PATH,
    load_grid_config,
)


@pytest.fixture(scope="session")
def training_data() -> pd.DataFrame:
    """
    Load the processed training dataset once
    for all permutation tests.
    """

    return load_training_data(
        Path(
            "data/processed/gold_features.csv"
        )
    )


@pytest.fixture(scope="session")
def grid_config() -> dict[str, Any]:
    """
    Load the approved walk-forward grid
    configuration once.
    """

    return load_grid_config(
        CONFIG_PATH
    )


def test_permute_target_preserves_structure(
    training_data: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(42)

    permuted = permute_target(
        data=training_data,
        rng=rng,
    )

    assert permuted.index.equals(
        training_data.index
    )

    assert len(permuted) == len(
        training_data
    )

    non_target_columns = [
        column
        for column in training_data.columns
        if column != TARGET_COLUMN
    ]

    pd.testing.assert_frame_equal(
        permuted[non_target_columns],
        training_data[
            non_target_columns
        ],
    )


def test_permute_target_preserves_class_counts(
    training_data: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(42)

    permuted = permute_target(
        data=training_data,
        rng=rng,
    )

    original_counts = (
        training_data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    permuted_counts = (
        permuted[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    pd.testing.assert_series_equal(
        original_counts,
        permuted_counts,
    )


def test_permutation_is_reproducible(
    training_data: pd.DataFrame,
) -> None:
    first = permute_target(
        data=training_data,
        rng=np.random.default_rng(42),
    )

    second = permute_target(
        data=training_data,
        rng=np.random.default_rng(42),
    )

    pd.testing.assert_series_equal(
        first[TARGET_COLUMN],
        second[TARGET_COLUMN],
    )


def test_null_summary_calculates_p_value() -> None:
    null_results = pd.DataFrame(
        {
            "maximum_median_strategy_return": [
                0.01,
                0.02,
                0.03,
                0.04,
            ]
        }
    )

    summary = summarize_permutation_null(
        real_best_return=0.025,
        null_results=null_results,
    )

    assert summary[
        "exceedance_count"
    ] == 2

    assert summary[
        "empirical_p_value"
    ] == pytest.approx(3 / 5)



def test_permuted_data_uses_same_fold_indices(
    training_data: pd.DataFrame,
) -> None:
    permuted_data = permute_target(
        data=training_data,
        rng=np.random.default_rng(42),
    )

    real_folds = list(
        generate_walk_forward_folds(
            len(training_data),
            train_window=1000,
            test_window=250,
            step_size=250,
            purge_gap=1,
        )
    )

    permuted_folds = list(
        generate_walk_forward_folds(
            len(permuted_data),
            train_window=1000,
            test_window=250,
            step_size=250,
            purge_gap=1,
        )
    )

    assert real_folds == permuted_folds