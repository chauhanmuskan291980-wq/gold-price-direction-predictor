from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class TradeMetricsSummary:
    """Summarize the observed performance of completed trades."""

    trade_count: int
    winning_trade_count: int
    losing_trade_count: int
    breakeven_trade_count: int
    expectancy: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    win_rate: float
    max_losing_streak: int
    top_five_percent_winner_concentration: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the metrics into a JSON-compatible dictionary."""

        profit_factor: float | str

        if math.isinf(self.profit_factor):
            profit_factor = "Infinity"
        else:
            profit_factor = self.profit_factor

        return {
            "trade_count": self.trade_count,
            "winning_trade_count": self.winning_trade_count,
            "losing_trade_count": self.losing_trade_count,
            "breakeven_trade_count": self.breakeven_trade_count,
            "expectancy": self.expectancy,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": profit_factor,
            "win_rate": self.win_rate,
            "max_losing_streak": self.max_losing_streak,
            "top_five_percent_winner_concentration": (
                self.top_five_percent_winner_concentration
            ),
        }


def prepare_return_values(
    trade_returns: pd.Series,
) -> NDArray[np.float64]:
    """Convert completed-trade returns into a finite numeric array."""

    numeric_returns = pd.to_numeric(
        trade_returns,
        errors="raise",
    ).astype(float)

    values = cast(
        NDArray[np.float64],
        numeric_returns.to_numpy(
            dtype=np.float64,
        ),
    )

    if values.size == 0:
        raise ValueError(
            "Trade returns cannot be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Trade returns must contain only finite values."
        )

    return values


def calculate_profit_factor(
    return_values: NDArray[np.float64],
) -> float:
    """Calculate gross profit divided by absolute gross loss."""

    gross_profit = float(
        np.sum(return_values[return_values > 0])
    )

    gross_loss = float(
        np.abs(
            np.sum(return_values[return_values < 0])
        )
    )

    if gross_loss == 0.0:
        if gross_profit > 0.0:
            return math.inf

        return 0.0

    return gross_profit / gross_loss


def calculate_max_losing_streak(
    return_values: NDArray[np.float64],
) -> int:
    """Calculate the longest sequence of consecutive losing trades."""

    longest_streak = 0
    current_streak = 0

    for trade_return in return_values:
        if trade_return < 0:
            current_streak += 1
            longest_streak = max(
                longest_streak,
                current_streak,
            )
        else:
            current_streak = 0

    return longest_streak


def calculate_top_winner_concentration(
    return_values: NDArray[np.float64],
    *,
    winner_fraction: float = 0.05,
) -> float:
    """Calculate the share of profit from the largest winners."""

    if not 0.0 < winner_fraction <= 1.0:
        raise ValueError(
            "winner_fraction must be greater than zero "
            "and less than or equal to one."
        )

    winning_returns = return_values[
        return_values > 0
    ]

    if winning_returns.size == 0:
        return 0.0

    gross_profit = float(
        np.sum(winning_returns)
    )

    top_winner_count = max(
        1,
        math.ceil(
            winning_returns.size
            * winner_fraction
        ),
    )

    sorted_winners = np.sort(
        winning_returns
    )

    top_winners = sorted_winners[
        -top_winner_count:
    ]

    top_winner_profit = float(
        np.sum(top_winners)
    )

    return top_winner_profit / gross_profit


def calculate_trade_metrics(
    trade_returns: pd.Series,
) -> TradeMetricsSummary:
    """Calculate observed metrics for completed strategy trades."""

    return_values = prepare_return_values(
        trade_returns
    )

    winning_returns = return_values[
        return_values > 0
    ]

    losing_returns = return_values[
        return_values < 0
    ]

    breakeven_returns = return_values[
        return_values == 0
    ]

    trade_count = int(return_values.size)
    winning_trade_count = int(
        winning_returns.size
    )
    losing_trade_count = int(
        losing_returns.size
    )
    breakeven_trade_count = int(
        breakeven_returns.size
    )

    gross_profit = float(
        np.sum(winning_returns)
    )

    gross_loss = float(
        np.abs(
            np.sum(losing_returns)
        )
    )

    return TradeMetricsSummary(
        trade_count=trade_count,
        winning_trade_count=winning_trade_count,
        losing_trade_count=losing_trade_count,
        breakeven_trade_count=breakeven_trade_count,
        expectancy=float(
            np.mean(return_values)
        ),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=calculate_profit_factor(
            return_values
        ),
        win_rate=(
            winning_trade_count
            / trade_count
        ),
        max_losing_streak=(
            calculate_max_losing_streak(
                return_values
            )
        ),
        top_five_percent_winner_concentration=(
            calculate_top_winner_concentration(
                return_values,
                winner_fraction=0.05,
            )
        ),
    )