from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gate.schemas import TemporaryValidationReport
from gate.validation import (
    DEFAULT_STEP_SIZE,
    DEFAULT_WINDOW_SIZE,
    validate,
)


def positive_integer(value: str) -> int:
    """Parse and validate a positive command-line integer."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "The value must be an integer."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "The value must be greater than zero."
        )

    return parsed_value


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

    validate_parser.add_argument(
        "--window-size",
        type=positive_integer,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "Number of completed trades in each chronological "
            f"window. Default: {DEFAULT_WINDOW_SIZE}."
        ),
    )

    validate_parser.add_argument(
        "--step-size",
        type=positive_integer,
        default=DEFAULT_STEP_SIZE,
        help=(
            "Number of completed trades between window starts. "
            f"Default: {DEFAULT_STEP_SIZE}."
        ),
    )

    return parser


def format_temporary_report(
    report: TemporaryValidationReport,
) -> str:
    """Format the current validation report for terminal output."""

    formatted_columns = ", ".join(report.columns)

    lines = [
        "",
        "SIGNAL VALIDATION GATE",
        "======================",
        f"Source: {report.source_path}",
        f"Input type: {report.input_type.value}",
        f"Rows: {report.row_count}",
        f"Completed trades: {report.trade_count}",
        f"Return unit: {report.return_unit}",
        f"Columns: {formatted_columns}",
        "",
        "WALK-FORWARD DISTRIBUTION",
        "-------------------------",
    ]

    if report.walk_forward is None:
        lines.append(
            "Unavailable: not enough completed trades "
            "for one full window."
        )
    else:
        walk_forward = report.walk_forward

        lines.extend(
            [
                f"Windows: {walk_forward.window_count}",
                f"Window size: {walk_forward.window_size}",
                f"Step size: {walk_forward.step_size}",
                (
                    "Median expectancy: "
                    f"{walk_forward.median_expectancy:.6f}"
                ),
                (
                    "Expectancy IQR: "
                    f"{walk_forward.iqr_expectancy:.6f}"
                ),
                (
                    "Worst-window expectancy: "
                    f"{walk_forward.worst_window_expectancy:.6f}"
                ),
                (
                    "Positive windows: "
                    f"{walk_forward.positive_window_rate:.2%}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            f"Verdict: {report.verdict.value}",
            f"Message: {report.message}",
            "",
        ]
    )

    return "\n".join(lines)


def run_validate_command(
    trades_path: Path,
    *,
    window_size: int,
    step_size: int,
) -> int:
    """Run validation for the supplied strategy CSV."""

    try:
        report = validate(
            trades_path,
            window_size=window_size,
            step_size=step_size,
        )
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
            window_size=arguments.window_size,
            step_size=arguments.step_size,
        )

    parser.error(
        f"Unsupported command: {arguments.command}"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())