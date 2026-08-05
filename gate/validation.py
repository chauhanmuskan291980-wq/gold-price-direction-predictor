from __future__ import annotations

from pathlib import Path

import pandas as pd

from gate.bootstrap import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_LOWER_PERCENTILE,
    run_bootstrap,
)
from gate.input_loader import load_strategy_csv
from gate.metrics import calculate_trade_metrics
from gate.normalization import normalize_strategy
from gate.nulls import (
    DEFAULT_NULL_ITERATIONS,
    DEFAULT_NULL_PERCENTILE,
    DEFAULT_NULL_SEED,
    run_selection_adjusted_null,
)
from gate.schemas import (
    TemporaryValidationReport,
    Verdict,
    WalkForwardSummary,
)
from gate.windows import calculate_walk_forward_summary

DEFAULT_WINDOW_SIZE = 50
DEFAULT_STEP_SIZE = 50


def validate(
    trades_path: str | Path,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    configs_tried: int | None = None,
    null_iterations: int = DEFAULT_NULL_ITERATIONS,
    null_seed: int = DEFAULT_NULL_SEED,
    null_percentile_level: float = DEFAULT_NULL_PERCENTILE,
) -> TemporaryValidationReport:
    """Load, normalize, and partially validate a strategy history."""

    source_path = Path(trades_path).expanduser()

    dataframe, input_type = load_strategy_csv(
        source_path
    )

    normalized_strategy = normalize_strategy(
        dataframe,
        input_type,
    )

    metrics = calculate_trade_metrics(
        normalized_strategy.trade_returns
    )

    bootstrap = run_bootstrap(
        normalized_strategy.trade_returns,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        lower_percentile=bootstrap_lower_percentile,
    )

    walk_forward = build_walk_forward_summary(
        normalized_strategy.trade_returns,
        window_size=window_size,
        step_size=step_size,
    )

    null_test = None

    if configs_tried is not None:
        null_test = run_selection_adjusted_null(
            normalized_strategy.trade_returns,
            configs_tried=configs_tried,
            iterations=null_iterations,
            seed=null_seed,
            percentile_level=null_percentile_level,
        )

    if walk_forward is None and null_test is None:
        message = (
            "CSV loaded, normalized, and evaluated with "
            "observed metrics and bootstrap evidence. There "
            "are not enough completed trades for one full "
            "chronological window, and the permutation-null "
            "test was not run because configs_tried was not "
            "provided."
        )
    elif walk_forward is None:
        message = (
            "CSV loaded, normalized, and evaluated with "
            "observed metrics, bootstrap evidence, and a "
            "selection-adjusted permutation null. There are "
            "not enough completed trades for one full "
            "chronological evaluation window."
        )
    elif null_test is None:
        message = (
            "CSV loaded, normalized, and evaluated with "
            "chronological windows, observed metrics, and "
            "bootstrap evidence. The permutation-null test "
            "was not run because configs_tried was not "
            "provided."
        )
    else:
        message = (
            "CSV loaded, normalized, and evaluated with "
            "chronological windows, observed metrics, "
            "bootstrap evidence, and a selection-adjusted "
            "permutation null. The final verdict policy is "
            "not implemented yet."
        )

    return TemporaryValidationReport(
        source_path=source_path,
        input_type=input_type,
        row_count=len(dataframe),
        columns=tuple(dataframe.columns),
        trade_count=normalized_strategy.trade_count,
        return_unit=normalized_strategy.return_unit,
        walk_forward=walk_forward,
        metrics=metrics,
        bootstrap=bootstrap,
        null_test=null_test,
        verdict=Verdict.INSUFFICIENT_DATA,
        message=message,
    )


def build_walk_forward_summary(
    trade_returns: pd.Series,
    *,
    window_size: int,
    step_size: int,
) -> WalkForwardSummary | None:
    """Build window evidence when enough trades are available."""

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be greater than zero."
        )

    if len(trade_returns) < window_size:
        return None

    return calculate_walk_forward_summary(
        trade_returns,
        window_size=window_size,
        step_size=step_size,
    )