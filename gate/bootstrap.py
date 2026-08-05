from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gate.metrics import (
    calculate_profit_factor,
    prepare_return_values,
)

DEFAULT_BOOTSTRAP_ITERATIONS = 5_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_LOWER_PERCENTILE = 5.0


@dataclass(frozen=True)
class BootstrapSummary:
    """Summarize trade-level bootstrap evidence."""

    iterations: int
    seed: int
    sample_size: int
    lower_percentile: float
    observed_expectancy: float
    observed_profit_factor: float
    expectancy_lower_bound: float
    profit_factor_lower_bound: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the bootstrap summary into a JSON-safe dictionary."""

        return {
            "iterations": self.iterations,
            "seed": self.seed,
            "sample_size": self.sample_size,
            "lower_percentile": self.lower_percentile,
            "observed_expectancy": self.observed_expectancy,
            "observed_profit_factor": json_safe_number(
                self.observed_profit_factor
            ),
            "expectancy_lower_bound": (
                self.expectancy_lower_bound
            ),
            "profit_factor_lower_bound": json_safe_number(
                self.profit_factor_lower_bound
            ),
        }


def json_safe_number(
    value: float,
) -> float | str:
    """Convert infinite values into JSON-safe text."""

    if math.isinf(value):
        if value > 0:
            return "Infinity"

        return "-Infinity"

    return value


def calculate_empirical_percentile(
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

    return float(sorted_values[index])


def run_bootstrap(
    trade_returns: pd.Series,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
) -> BootstrapSummary:
    """Estimate lower bounds by resampling completed trades."""

    if iterations <= 0:
        raise ValueError(
            "iterations must be greater than zero."
        )

    if seed < 0:
        raise ValueError(
            "seed must be zero or greater."
        )

    if not 0.0 < lower_percentile < 50.0:
        raise ValueError(
            "lower_percentile must be greater than zero "
            "and less than 50."
        )

    return_values = prepare_return_values(
        trade_returns
    )

    sample_size = int(return_values.size)

    observed_expectancy = float(
        np.mean(return_values)
    )

    observed_profit_factor = (
        calculate_profit_factor(return_values)
    )

    bootstrap_expectancies: NDArray[np.float64] = (
        np.empty(
            iterations,
            dtype=np.float64,
        )
    )

    bootstrap_profit_factors: NDArray[np.float64] = (
        np.empty(
            iterations,
            dtype=np.float64,
        )
    )

    random_generator = np.random.default_rng(
        seed
    )

    for iteration in range(iterations):
        sampled_indices = random_generator.integers(
            low=0,
            high=sample_size,
            size=sample_size,
        )

        sampled_returns = cast(
            NDArray[np.float64],
            return_values[sampled_indices],
        )

        bootstrap_expectancies[iteration] = (
            float(
                np.mean(sampled_returns)
            )
        )

        bootstrap_profit_factors[iteration] = (
            calculate_profit_factor(
                sampled_returns
            )
        )

    expectancy_lower_bound = (
        calculate_empirical_percentile(
            bootstrap_expectancies,
            percentile=lower_percentile,
        )
    )

    profit_factor_lower_bound = (
        calculate_empirical_percentile(
            bootstrap_profit_factors,
            percentile=lower_percentile,
        )
    )

    return BootstrapSummary(
        iterations=iterations,
        seed=seed,
        sample_size=sample_size,
        lower_percentile=lower_percentile,
        observed_expectancy=observed_expectancy,
        observed_profit_factor=observed_profit_factor,
        expectancy_lower_bound=expectancy_lower_bound,
        profit_factor_lower_bound=(
            profit_factor_lower_bound
        ),
    )