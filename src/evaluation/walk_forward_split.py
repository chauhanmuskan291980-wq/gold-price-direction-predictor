from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardFold:
    """Index boundaries for one walk-forward fold."""

    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        """Return the training slice using exclusive end indexing."""
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        """Return the testing slice using exclusive end indexing."""
        return slice(self.test_start, self.test_end)


def generate_walk_forward_folds(
    total_rows: int,
    train_window: int,
    test_window: int,
    step_size: int,
    purge_gap: int = 1,
) -> Iterator[WalkForwardFold]:
    """
    Generate rolling-origin walk-forward folds.

    Each fold contains:

    - A fixed-size training window.
    - A purged gap after the training window.
    - A fixed-size test window.
    - A configurable forward step.

    End indexes are exclusive.
    """

    if total_rows <= 0:
        raise ValueError("total_rows must be greater than zero.")

    if train_window <= 0:
        raise ValueError("train_window must be greater than zero.")

    if test_window <= 0:
        raise ValueError("test_window must be greater than zero.")

    if step_size <= 0:
        raise ValueError("step_size must be greater than zero.")

    if purge_gap < 0:
        raise ValueError("purge_gap cannot be negative.")

    minimum_required_rows = (
        train_window
        + purge_gap
        + test_window
    )

    if total_rows < minimum_required_rows:
        raise ValueError(
            "The dataset does not contain enough rows for one "
            "walk-forward fold. "
            f"Required: {minimum_required_rows}, "
            f"available: {total_rows}."
        )

    fold_number = 1
    train_start = 0

    while True:
        train_end = train_start + train_window
        test_start = train_end + purge_gap
        test_end = test_start + test_window

        if test_end > total_rows:
            break

        yield WalkForwardFold(
            fold_number=fold_number,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        fold_number += 1
        train_start += step_size