from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from gate.nulls import (
    DEFAULT_NULL_ITERATIONS,
    DEFAULT_NULL_SEED,
)
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


def non_negative_integer(value: str) -> int:
    """Parse and validate a non-negative command-line integer."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "The value must be an integer."
        ) from error

    if parsed_value < 0:
        raise argparse.ArgumentTypeError(
            "The value must be zero or greater."
        )

    return parsed_value


def resolve_configs_tried(
    configs_tried: int | None,
) -> int:
    """Return configs tried, prompting when it was not supplied."""

    if configs_tried is not None:
        return configs_tried

    try:
        entered_value = input(
            "How many strategy configurations were tried "
            "before selecting this result? "
        )
    except EOFError as error:
        raise ValueError(
            "configs_tried is required. Supply it with "
            "--configs-tried when running non-interactively."
        ) from error

    try:
        return positive_integer(
            entered_value.strip()
        )
    except argparse.ArgumentTypeError as error:
        raise ValueError(
            "configs_tried must be a positive integer."
        ) from error


def format_number(value: float) -> str:
    """Format finite and infinite metric values."""

    if math.isinf(value):
        return (
            "Infinity"
            if value > 0
            else "-Infinity"
        )

    return f"{value:.6f}"


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

    validate_parser.add_argument(
        "--configs-tried",
        type=positive_integer,
        default=None,
        help=(
            "Number of strategy configurations tested before "
            "selecting this result. The CLI prompts when omitted."
        ),
    )

    validate_parser.add_argument(
        "--null-iterations",
        type=positive_integer,
        default=DEFAULT_NULL_ITERATIONS,
        help=(
            "Number of permutation-null simulations. "
            f"Default: {DEFAULT_NULL_ITERATIONS}."
        ),
    )

    validate_parser.add_argument(
        "--null-seed",
        type=non_negative_integer,
        default=DEFAULT_NULL_SEED,
        help=(
            "Random seed for permutation-null simulations. "
            f"Default: {DEFAULT_NULL_SEED}."
        ),
    )

    return parser


def format_temporary_report(
    report: TemporaryValidationReport,
) -> str:
    """Format the current validation report for terminal output."""

    formatted_columns = ", ".join(
        report.columns
    )

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
            "OBSERVED TRADE METRICS",
            "----------------------",
        ]
    )

    if report.metrics is None:
        lines.append("Unavailable.")
    else:
        metrics = report.metrics

        lines.extend(
            [
                f"Trades: {metrics.trade_count}",
                (
                    "Winning trades: "
                    f"{metrics.winning_trade_count}"
                ),
                (
                    "Losing trades: "
                    f"{metrics.losing_trade_count}"
                ),
                (
                    "Breakeven trades: "
                    f"{metrics.breakeven_trade_count}"
                ),
                (
                    "Expectancy: "
                    f"{format_number(metrics.expectancy)}"
                ),
                (
                    "Gross profit: "
                    f"{format_number(metrics.gross_profit)}"
                ),
                (
                    "Gross loss: "
                    f"{format_number(metrics.gross_loss)}"
                ),
                (
                    "Profit factor: "
                    f"{format_number(metrics.profit_factor)}"
                ),
                f"Win rate: {metrics.win_rate:.2%}",
                (
                    "Maximum losing streak: "
                    f"{metrics.max_losing_streak}"
                ),
                (
                    "Top-5% winner concentration: "
                    f"{metrics.top_five_percent_winner_concentration:.2%}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "BOOTSTRAP LOWER BOUNDS",
            "----------------------",
        ]
    )

    if report.bootstrap is None:
        lines.append("Unavailable.")
    else:
        bootstrap = report.bootstrap

        lines.extend(
            [
                f"Iterations: {bootstrap.iterations}",
                f"Seed: {bootstrap.seed}",
                f"Sample size: {bootstrap.sample_size}",
                (
                    "Lower percentile: "
                    f"{bootstrap.lower_percentile:.2f}%"
                ),
                (
                    "Observed expectancy: "
                    f"{format_number(bootstrap.observed_expectancy)}"
                ),
                (
                    "Observed profit factor: "
                    f"{format_number(bootstrap.observed_profit_factor)}"
                ),
                (
                    "Expectancy lower bound: "
                    f"{format_number(bootstrap.expectancy_lower_bound)}"
                ),
                (
                    "Profit-factor lower bound: "
                    f"{format_number(bootstrap.profit_factor_lower_bound)}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "SELECTION-ADJUSTED NULL",
            "-----------------------",
        ]
    )

    if report.null_test is None:
        lines.append(
            "Unavailable: configs_tried was not provided."
        )
    else:
        null_test = report.null_test

        lines.extend(
            [
                f"Method: {null_test.null_method}",
                f"Statistic: {null_test.statistic_name}",
                f"Configurations tried: {null_test.configs_tried}",
                f"Iterations: {null_test.iterations}",
                f"Seed: {null_test.seed}",
                (
                    "Observed statistic: "
                    f"{null_test.observed_statistic:.6f}"
                ),
                (
                    "Unadjusted p-value: "
                    f"{null_test.unadjusted_p_value:.6f}"
                ),
                (
                    "Selection-adjusted p-value: "
                    f"{null_test.selection_adjusted_p_value:.6f}"
                ),
                (
                    f"Unadjusted {null_test.percentile_level:.2f}% "
                    "null threshold: "
                    f"{null_test.unadjusted_null_percentile:.6f}"
                ),
                (
                    f"Adjusted {null_test.percentile_level:.2f}% "
                    "null threshold: "
                    f"{null_test.adjusted_null_percentile:.6f}"
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
    configs_tried: int | None,
    null_iterations: int,
    null_seed: int,
) -> int:
    """Run validation for the supplied strategy CSV."""

    try:
        resolved_configs_tried = (
            resolve_configs_tried(
                configs_tried
            )
        )

        report = validate(
            trades_path,
            window_size=window_size,
            step_size=step_size,
            configs_tried=resolved_configs_tried,
            null_iterations=null_iterations,
            null_seed=null_seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(
            f"Validation failed: {error}",
            file=sys.stderr,
        )
        return 2

    print(
        format_temporary_report(report)
    )

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
            configs_tried=arguments.configs_tried,
            null_iterations=arguments.null_iterations,
            null_seed=arguments.null_seed,
        )

    parser.error(
        f"Unsupported command: {arguments.command}"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())