from pathlib import Path
import pandas as pd

from gate.schemas import InputType


CLOSED_TRADE_RETURN_COLUMNS = {
    "open_time",
    "close_time",
    "side",
    "return_R"
}

CLOSED_TRADE_PNL_COLUMNS = {
    "open_time",
    "close_time",
    "side",
    "pnl",
    "risk",
}

BAR_SERIES_COLUMNS = {
    "timestamp",
    "position",
    "forward_return"
}

def detect_input_type(
        dataframe : pd.DataFrame,
)-> InputType:
    columns = set(dataframe.columns)

    if CLOSED_TRADE_RETURN_COLUMNS <= columns:
        return InputType.CLOSED_TRADES

    if CLOSED_TRADE_PNL_COLUMNS <= columns:
        return InputType.CLOSED_TRADES

    if BAR_SERIES_COLUMNS <= columns:
        return InputType.BAR_SERIES

    raise ValueError(
        "Unsupported CSV structure , Expected either"
        "closed-trade columns or bar-series columns."
    )


def load_strategy_csv(
        path: str | Path,
) -> tuple[pd.DataFrame,InputType]:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Trade file not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        raise ValueError(
            "The supplied CSV contains no rows."
        )

    input_type = detect_input_type(dataframe)

    return dataframe , input_type