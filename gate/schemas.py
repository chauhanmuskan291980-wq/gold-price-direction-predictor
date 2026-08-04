from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class InputType(str, Enum):
    """CSV input formats supported by the validation gate."""

    CLOSED_TRADES = "closed_trades"
    BAR_SERIES = "bar_series"


class Verdict(str, Enum):
    """Possible final decisions returned by the gate."""

    EDGE = "EDGE"
    NO_EDGE = "NO-EDGE"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


@dataclass(frozen=True)
class WalkForwardSummary:
    """Summarize performance across chronological evaluation windows."""

    window_count: int
    window_size: int
    step_size: int
    median_expectancy: float
    iqr_expectancy: float
    worst_window_expectancy: float
    positive_window_rate: float
    window_expectancies: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary into a JSON-compatible dictionary."""

        return {
            "window_count": self.window_count,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "median_expectancy": self.median_expectancy,
            "iqr_expectancy": self.iqr_expectancy,
            "worst_window_expectancy": (
                self.worst_window_expectancy
            ),
            "positive_window_rate": (
                self.positive_window_rate
            ),
            "window_expectancies": list(
                self.window_expectancies
            ),
        }


@dataclass(frozen=True)
class TemporaryValidationReport:
    """Represent the current partial result produced by the gate."""

    source_path: Path
    input_type: InputType
    row_count: int
    columns: tuple[str, ...]
    trade_count: int
    return_unit: str
    walk_forward: WalkForwardSummary | None
    verdict: Verdict
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-compatible dictionary."""

        return {
            "source_path": str(self.source_path),
            "input_type": self.input_type.value,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "trade_count": self.trade_count,
            "return_unit": self.return_unit,
            "walk_forward": (
                self.walk_forward.to_dict()
                if self.walk_forward is not None
                else None
            ),
            "verdict": self.verdict.value,
            "message": self.message,
        }