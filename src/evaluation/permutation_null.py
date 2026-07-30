from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.train import TARGET_COLUMN


def permute_target(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Return a copy with only target labels permuted.

    Features, timestamps, returns, indexes, and class
    balance remain unchanged.
    """

    permuted_data = data.copy()

    original_target = data[
        TARGET_COLUMN
    ].to_numpy()

    permuted_data[TARGET_COLUMN] = (
        rng.permutation(original_target)
    )

    return permuted_data


def summarize_permutation_null(
    real_best_return: float,
    null_results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compare the real best-of-nine result with the
    permuted best-of-nine null distribution.
    """

    metric_column = (
        "maximum_median_strategy_return"
    )

    if metric_column not in null_results:
        raise ValueError(
            f"Missing column: {metric_column}"
        )

    null_returns = null_results[
        metric_column
    ].to_numpy(
        dtype=float
    )

    if len(null_returns) == 0:
        raise ValueError(
            "The permutation null contains no results."
        )

    q1 = float(
        np.quantile(
            null_returns,
            0.25,
        )
    )

    q3 = float(
        np.quantile(
            null_returns,
            0.75,
        )
    )

    exceedance_count = int(
        np.sum(
            null_returns
            >= real_best_return
        )
    )

    empirical_p_value = float(
        (
            exceedance_count + 1
        )
        / (
            len(null_returns) + 1
        )
    )

    real_percentile = float(
        np.mean(
            null_returns
            <= real_best_return
        )
        * 100.0
    )

    return {
        "real_best_median_strategy_return": float(
            real_best_return
        ),
        "null_median": float(
            np.median(null_returns)
        ),
        "null_q1": q1,
        "null_q3": q3,
        "null_iqr": q3 - q1,
        "null_95th_percentile": float(
            np.quantile(
                null_returns,
                0.95,
            )
        ),
        "null_maximum": float(
            np.max(null_returns)
        ),
        "exceedance_count": exceedance_count,
        "real_result_percentile": (
            real_percentile
        ),
        "empirical_p_value": (
            empirical_p_value
        ),
    }