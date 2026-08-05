from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from gate.metrics import prepare_return_values

DEFAULT_NULL_ITERATIONS = 5_000
DEFAULT_NULL_SEED = 42
DEFAULT_NULL_PERCENTILE = 95.0

NULL_METHOD = "sign_permutation_max_statistic"
STATISTIC_NAME = "mean_trade_return"


@dataclass(frozen=True)
class NullSummary:
    """Summarize selection-adjusted permutation-null evidence."""

    iterations: int
    seed: int
    configs_tried: int
    sample_size: int
    statistic_name: str
    null_method: str
    observed_statistic: float
    unadjusted_p_value: float
    selection_adjusted_p_value: float
    unadjusted_null_percentile: float
    adjusted_null_percentile: float
    percentile_level: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the null summary into a JSON-compatible dictionary."""

        return {
            "iterations": self.iterations,
            "seed": self.seed,
            "configs_tried": self.configs_tried,
            "sample_size": self.sample_size,
            "statistic_name": self.statistic_name,
            "null_method": self.null_method,
            "observed_statistic": self.observed_statistic,
            "unadjusted_p_value": self.unadjusted_p_value,
            "selection_adjusted_p_value": (
                self.selection_adjusted_p_value
            ),
            "unadjusted_null_percentile": (
                self.unadjusted_null_percentile
            ),
            "adjusted_null_percentile": (
                self.adjusted_null_percentile
            ),
            "percentile_level": self.percentile_level,
        }


def generate_candidate_expectancies(
    magnitudes: NDArray[np.float64],
    *,
    configs_tried: int,
    random_generator: Generator,
) -> NDArray[np.float64]:
    """Generate null expectancies for simulated strategy configs."""

    sample_size = int(magnitudes.size)

    random_bits = random_generator.integers(
        low=0,
        high=2,
        size=(
            configs_tried,
            sample_size,
        ),
        dtype=np.int8,
    )

    signs = random_bits.astype(
        np.float64
    )

    signs = (
        signs * 2.0
        - 1.0
    )

    candidate_expectancies = np.mean(
        signs * magnitudes,
        axis=1,
        dtype=np.float64,
    )

    return cast(
        NDArray[np.float64],
        candidate_expectancies,
    )


def calculate_upper_tail_p_value(
    null_values: NDArray[np.float64],
    *,
    observed_statistic: float,
) -> float:
    """Calculate a corrected upper-tail permutation p-value."""

    if null_values.size == 0:
        raise ValueError(
            "Null values cannot be empty."
        )

    if not np.isfinite(null_values).all():
        raise ValueError(
            "Null values must contain only finite values."
        )

    if not math.isfinite(observed_statistic):
        raise ValueError(
            "observed_statistic must be finite."
        )

    exceedance_count = int(
        np.count_nonzero(
            null_values >= observed_statistic
        )
    )

    return (
        exceedance_count + 1
    ) / (
        null_values.size + 1
    )


def calculate_nearest_rank_percentile(
    values: NDArray[np.float64],
    *,
    percentile: float,
) -> float:
    """Calculate a percentile using the nearest-rank method."""

    if values.size == 0:
        raise ValueError(
            "Percentile values cannot be empty."
        )

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            "percentile must be between zero and 100."
        )

    sorted_values = np.sort(values)

    rank = math.ceil(
        (percentile / 100.0)
        * sorted_values.size
    )

    index = min(
        max(rank - 1, 0),
        sorted_values.size - 1,
    )

    return float(
        sorted_values[index]
    )


def run_selection_adjusted_null(
    trade_returns: pd.Series,
    *,
    configs_tried: int,
    iterations: int = DEFAULT_NULL_ITERATIONS,
    seed: int = DEFAULT_NULL_SEED,
    percentile_level: float = DEFAULT_NULL_PERCENTILE,
) -> NullSummary:
    """Run a sign-permutation max-statistic null test."""

    if configs_tried <= 0:
        raise ValueError(
            "configs_tried must be greater than zero."
        )

    if iterations <= 0:
        raise ValueError(
            "iterations must be greater than zero."
        )

    if seed < 0:
        raise ValueError(
            "seed must be zero or greater."
        )

    if not 50.0 < percentile_level < 100.0:
        raise ValueError(
            "percentile_level must be greater than 50 "
            "and less than 100."
        )

    return_values = prepare_return_values(
        trade_returns
    )

    magnitudes = np.abs(
        return_values
    )

    sample_size = int(
        return_values.size
    )

    observed_statistic = float(
        np.mean(return_values)
    )

    unadjusted_null_values: NDArray[np.float64] = (
        np.empty(
            iterations,
            dtype=np.float64,
        )
    )

    adjusted_null_values: NDArray[np.float64] = (
        np.empty(
            iterations,
            dtype=np.float64,
        )
    )

    random_generator = np.random.default_rng(
        seed
    )

    for iteration in range(iterations):
        candidate_expectancies = (
            generate_candidate_expectancies(
                magnitudes,
                configs_tried=configs_tried,
                random_generator=random_generator,
            )
        )

        unadjusted_null_values[iteration] = (
            candidate_expectancies[0]
        )

        adjusted_null_values[iteration] = float(
            np.max(candidate_expectancies)
        )

    unadjusted_p_value = (
        calculate_upper_tail_p_value(
            unadjusted_null_values,
            observed_statistic=observed_statistic,
        )
    )

    selection_adjusted_p_value = (
        calculate_upper_tail_p_value(
            adjusted_null_values,
            observed_statistic=observed_statistic,
        )
    )

    unadjusted_null_percentile = (
        calculate_nearest_rank_percentile(
            unadjusted_null_values,
            percentile=percentile_level,
        )
    )

    adjusted_null_percentile = (
        calculate_nearest_rank_percentile(
            adjusted_null_values,
            percentile=percentile_level,
        )
    )

    return NullSummary(
        iterations=iterations,
        seed=seed,
        configs_tried=configs_tried,
        sample_size=sample_size,
        statistic_name=STATISTIC_NAME,
        null_method=NULL_METHOD,
        observed_statistic=observed_statistic,
        unadjusted_p_value=unadjusted_p_value,
        selection_adjusted_p_value=(
            selection_adjusted_p_value
        ),
        unadjusted_null_percentile=(
            unadjusted_null_percentile
        ),
        adjusted_null_percentile=(
            adjusted_null_percentile
        ),
        percentile_level=percentile_level,
    )