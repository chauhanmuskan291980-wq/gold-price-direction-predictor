from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gate.schemas import TemporaryValidationReport
from gate.validation import validate


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m gate",
        description=(
            "Validate a strategy trade history and produce "
            "an honest signal verdict."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a closed-trade or bar-series CSV file.",
    )

    validate_parser.add_argument(
        "--trades",
        type=Path,
        required=True,
        help="Path to the strategy trade-history CSV file.",
    )

    return parser


def format_temporary_report(
    report: TemporaryValidationReport,
) -> str:
    """Format the temporary report for terminal output."""

    formatted_columns = ", ".join(report.columns)

    return "\n".join(
        [
            "",
            "SIGNAL VALIDATION GATE",
            "======================",
            f"Source: {report.source_path}",
            f"Input type: {report.input_type.value}",
            f"Rows: {report.row_count}",
            f"Columns: {formatted_columns}",
            "",
            f"Verdict: {report.verdict.value}",
            f"Message: {report.message}",
            "",
        ]
    )


def run_validate_command(
    trades_path: Path,
) -> int:
    """Run validation for the supplied strategy CSV."""

    try:
        report = validate(trades_path)
    except (FileNotFoundError, ValueError) as error:
        print(
            f"Validation failed: {error}",
            file=sys.stderr,
        )
        return 2

    print(format_temporary_report(report))

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the Signal Validation Gate command-line interface."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "validate":
        return run_validate_command(
            trades_path=arguments.trades,
        )

    parser.error(
        f"Unsupported command: {arguments.command}"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())