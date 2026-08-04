from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gate.schemas import InputType

VALID_TRADE_SIDES = frozenset(
    {
        "long",
        "short",
    }
)


@dataclass(frozen=True)
class NormalizedStrategy:
    """Standard representation used by the validation pipeline."""

    input_type: InputType
    observation_times: pd.Series
    strategy_returns: pd.Series
    trade_returns: pd.Series
    trade_count: int
    return_unit: str


def normalize_closed_trades(
    dataframe: pd.DataFrame,
) -> NormalizedStrategy:
    """Normalize a closed-trade CSV into ordered trade returns."""

    trades = dataframe.copy()

    trades["open_time"] = pd.to_datetime(
        trades["open_time"],
        utc=True,
        errors="raise",
    )

    trades["close_time"] = pd.to_datetime(
        trades["close_time"],
        utc=True,
        errors="raise",
    )

    if (
        trades["close_time"]
        <= trades["open_time"]
    ).any():
        raise ValueError(
            "Every close_time must be later than open_time."
        )

    trades["side"] = (
        trades["side"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_sides = (
        set(trades["side"])
        - VALID_TRADE_SIDES
    )

    if invalid_sides:
        raise ValueError(
            "Invalid trade side values: "
            f"{sorted(invalid_sides)}. "
            "Expected long or short."
        )

    if "return_R" not in trades.columns:
        trades["pnl"] = pd.to_numeric(
            trades["pnl"],
            errors="raise",
        )

        trades["risk"] = pd.to_numeric(
            trades["risk"],
            errors="raise",
        )

        if (trades["risk"] <= 0).any():
            raise ValueError(
                "Every risk value must be greater than zero."
            )

        trades["return_R"] = (
            trades["pnl"]
            / trades["risk"]
        )

    trades["return_R"] = pd.to_numeric(
        trades["return_R"],
        errors="raise",
    )

    return_values = trades[
        "return_R"
    ].to_numpy(dtype=float)

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Trade returns must contain only finite values."
        )

    trades = (
        trades.sort_values(
            "close_time",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    trade_returns = (
        trades["return_R"]
        .astype(float)
        .reset_index(drop=True)
    )

    observation_times = (
        trades["close_time"]
        .reset_index(drop=True)
    )

    return NormalizedStrategy(
        input_type=InputType.CLOSED_TRADES,
        observation_times=observation_times,
        strategy_returns=trade_returns.copy(),
        trade_returns=trade_returns,
        trade_count=len(trade_returns),
        return_unit="R",
    )


def derive_trade_returns(
    bars: pd.DataFrame,
) -> pd.Series:
    """Combine consecutive same-side positions into trade outcomes."""

    completed_trade_returns: list[float] = []
    active_side = 0
    active_returns: list[float] = []

    for row in bars.itertuples(index=False):
        position = float(row.position)
        strategy_return = float(row.strategy_return)

        current_side = int(np.sign(position))

        if current_side == 0:
            if active_returns:
                completed_trade_returns.append(
                    compound_returns(active_returns)
                )

                active_returns = []
                active_side = 0

            continue

        if active_side == 0:
            active_side = current_side
            active_returns = [strategy_return]
            continue

        if current_side == active_side:
            active_returns.append(strategy_return)
            continue

        completed_trade_returns.append(
            compound_returns(active_returns)
        )

        active_side = current_side
        active_returns = [strategy_return]

    if active_returns:
        completed_trade_returns.append(
            compound_returns(active_returns)
        )

    return pd.Series(
        completed_trade_returns,
        dtype=float,
        name="trade_return",
    )


def compound_returns(
    returns: list[float],
) -> float:
    """Compound a sequence of decimal strategy returns."""

    values = np.asarray(
        returns,
        dtype=float,
    )

    if (values <= -1.0).any():
        raise ValueError(
            "A bar strategy return cannot be less than "
            "or equal to -100% when compounding trades."
        )

    return float(
        np.prod(1.0 + values)
        - 1.0
    )


def normalize_bar_series(
    dataframe: pd.DataFrame,
) -> NormalizedStrategy:
    """Normalize position and forward-return bars."""

    bars = dataframe.copy()

    bars["timestamp"] = pd.to_datetime(
        bars["timestamp"],
        utc=True,
        errors="raise",
    )

    bars["position"] = pd.to_numeric(
        bars["position"],
        errors="raise",
    )

    bars["forward_return"] = pd.to_numeric(
        bars["forward_return"],
        errors="raise",
    )

    numeric_values = bars[
        [
            "position",
            "forward_return",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Positions and forward returns must be finite."
        )

    bars = (
        bars.sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if bars["timestamp"].duplicated().any():
        raise ValueError(
            "Bar timestamps must be unique."
        )

    bars["strategy_return"] = (
        bars["position"]
        * bars["forward_return"]
    )

    strategy_returns = (
        bars["strategy_return"]
        .astype(float)
        .reset_index(drop=True)
    )

    trade_returns = derive_trade_returns(bars)

    observation_times = (
        bars["timestamp"]
        .reset_index(drop=True)
    )

    return NormalizedStrategy(
        input_type=InputType.BAR_SERIES,
        observation_times=observation_times,
        strategy_returns=strategy_returns,
        trade_returns=trade_returns,
        trade_count=len(trade_returns),
        return_unit="decimal_return",
    )


def normalize_strategy(
    dataframe: pd.DataFrame,
    input_type: InputType,
) -> NormalizedStrategy:
    """Normalize either supported CSV structure."""

    if input_type == InputType.CLOSED_TRADES:
        return normalize_closed_trades(dataframe)

    if input_type == InputType.BAR_SERIES:
        return normalize_bar_series(dataframe)

    raise ValueError(
        f"Unsupported input type: {input_type}"
    )