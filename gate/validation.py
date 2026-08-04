from pathlib import Path

from gate.input_loader import load_strategy_csv
from gate.schemas import TemporaryValidationReport, Verdict


def validate(
    trades_path: str | Path,
) -> TemporaryValidationReport:
    """
    Load a strategy CSV and produce the gate's temporary report.

    Statistical validation is intentionally not implemented at this stage.
    This function currently proves that the complete loading and reporting
    path works before more complex analysis is added.
    """

    source_path = Path(trades_path).expanduser()

    dataframe, input_type = load_strategy_csv(source_path)

    return TemporaryValidationReport(
        source_path=source_path,
        input_type=input_type,
        row_count=len(dataframe),
        columns=tuple(dataframe.columns),
        verdict=Verdict.INSUFFICIENT_DATA,
        message=(
            "CSV loaded and input format detected successfully. "
            "Statistical validation is not implemented yet."
        ),
    )