from __future__ import annotations

import numpy as np
import pandas as pd

from gate.schemas import WalkForwardSummary


def generate_window_expectancies(
    returns: pd.Series,
    *,
    window_size: int,
    step_size: int,
) -> tuple[float, ...]:
    """Calculate expectancy for each complete chronological window."""

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be greater than zero."
        )

    numeric_returns = pd.to_numeric(
        returns,
        errors="raise",
    ).astype(float)

    numeric_returns = numeric_returns.reset_index(
        drop=True
    )

    if numeric_returns.empty:
        raise ValueError(
            "Returns cannot be empty."
        )

    values = numeric_returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Returns must contain only finite values."
        )

    if len(values) < window_size:
        raise ValueError(
            "There is not enough data to create "
            "one complete evaluation window."
        )

    window_expectancies: list[float] = []

    final_start = (
        len(values)
        - window_size
        + 1
    )

    for start in range(
        0,
        final_start,
        step_size,
    ):
        end = start + window_size

        window_returns = values[start:end]

        window_expectancies.append(
            float(np.mean(window_returns))
        )

    return tuple(window_expectancies)


def calculate_walk_forward_summary(
    returns: pd.Series,
    *,
    window_size: int,
    step_size: int,
) -> WalkForwardSummary:
    """Summarize strategy performance across chronological windows."""

    window_expectancies = (
        generate_window_expectancies(
            returns,
            window_size=window_size,
            step_size=step_size,
        )
    )

    values = np.asarray(
        window_expectancies,
        dtype=float,
    )

    first_quartile = float(
        np.percentile(values, 25)
    )

    third_quartile = float(
        np.percentile(values, 75)
    )

    return WalkForwardSummary(
        window_count=len(values),
        window_size=window_size,
        step_size=step_size,
        median_expectancy=float(
            np.median(values)
        ),
        iqr_expectancy=(
            third_quartile
            - first_quartile
        ),
        worst_window_expectancy=float(
            np.min(values)
        ),
        positive_window_rate=float(
            np.mean(values > 0)
        ),
        window_expectancies=tuple(
            float(value)
            for value in values
        ),
    )