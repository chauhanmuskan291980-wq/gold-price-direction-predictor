from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class InputType(str, Enum):
    CLOSED_TRADES = "closed_trades"
    BAR_SERIES = "bar_series"


class Verdict(str, Enum):
    EDGE = "EDGE"
    NO_EDGE = "NO-EDGE"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


@dataclass(frozen=True)
class NormalizedStrategy:
    input_type: InputType
    strategy_returns: pd.Series
    trade_returns: pd.Series
    observation_times: pd.Series
    trade_count: int
    return_unit: str


@dataclass(frozen=True)
class WalkForwardSummary:
    window_count: int
    median_expectancy: float
    iqr_expectancy: float
    worst_window_expectancy: float
    positive_window_rate: float


@dataclass(frozen=True)
class BootstrapSummary:
    expectancy_lower_bound: float
    profit_factor_lower_bound: float


@dataclass(frozen=True)
class NullSummary:
    real_statistic: float
    null_median: float
    null_95th_percentile: float
    exceedance_count: int
    p_value: float


@dataclass(frozen=True)
class RiskSummary:
    maximum_losing_streak: int
    top_winner_concentration: float


@dataclass(frozen=True)
class ValidationReport:
    verdict: Verdict
    input_type: InputType
    trade_count: int
    configs_tried: int
    walk_forward: WalkForwardSummary | None
    bootstrap: BootstrapSummary | None
    permutation: NullSummary | None
    risk: RiskSummary | None
    checks: dict[str, bool]
    failed_checks: list[str]
    metadata: dict[str, Any]