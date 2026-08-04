from pathlib import Path

import pandas as pd

from gate.schemas import InputType

CLOSED_TRADE_RETURN_COLUMNS = frozenset(
    {
        "open_time",
        "close_time",
        "side",
        "return_R"
    }
)

CLOSED_TRADE_PNL_COLUMNS = frozenset(
    {
        "open_time"
        "close_time"
        "side"
        "pnl"
        "risk"
    }
)

BAR_SERIES_COLUMNS = frozenset(
    {
        "timestamp",
        "position",
        "forward_return"
    }
)


def detect_input_type(
        dataframe:pd.DataFrame,
) -> InputType:
    """Detect which supported strategy - history format is present."""

    columns = set(dataframe.columns)

    is_closed_trade_input = (
        CLOSED_TRADE_RETURN_COLUMNS <= columns
        or 
        CLOSED_TRADE_PNL_COLUMNS <= columns
    )

    is_bar_series_input = BAR_SERIES_COLUMNS <= columns

    if is_closed_trade_input and is_bar_series_input:
        raise ValueError(
            "The CSV is ambigous because it contains both"
            "closed-trade and bar-series columns."
        )

    if is_closed_trade_input:
        return InputType.CLOSED_TRADES

    if is_bar_series_input:
        return InputType.BAR_SERIES

    raise ValueError(
    "Unsupported CSV structure. Expected either "
    "open_time, close_time, side, return_R; "
    "open_time, close_time, side, pnl, risk; or "
    "timestamp, position, forward_return."
)


def load_strategy_csv(
        path: str| Path,
) -> tuple[pd.DataFrame , InputType]:
    """Load a strategy CSV and return its data and detedted input type."""

    csv_path = Path(path).expanduser()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Strategy CSV was not found : {csv_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"The supplied path is not a file:{csv_path}"
        )

    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"The supplied CSV is empty:{csv_path}"
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            f"The supplied CSV could not be parsed:{csv_path}"
        ) from error

    dataframe.columns = dataframe.columns.str.strip()

    if dataframe.empty:
        raise ValueError(
            "The supplied CSV contains column names but no data rows"
        )

    input_type = detect_input_type(dataframe)

    return dataframe , input_type