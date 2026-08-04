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
class TemporaryValidationReport:
    """Represent the first temporary result produced by the gate."""

    source_path: Path
    input_type: InputType
    row_count: int
    columns: tuple[str, ...]
    verdict: Verdict
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-compatible dictionary."""

        return {
            "source_path": str(self.source_path),
            "input_type": self.input_type.value,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "verdict": self.verdict.value,
            "message": self.message,
        }