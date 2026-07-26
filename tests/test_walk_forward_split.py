import pytest

from src.evaluation.walk_forward_split import (
    generate_walk_forward_folds,
)


def test_walk_forward_generates_expected_windows() -> None:
    folds = list(
        generate_walk_forward_folds(
            total_rows=1000,
            train_window=400,
            test_window=100,
            step_size=100,
            purge_gap=1,
        )
    )

    assert len(folds) > 1

    first_fold = folds[0]

    assert first_fold.train_start == 0
    assert first_fold.train_end == 400
    assert first_fold.test_start == 401
    assert first_fold.test_end == 501


def test_train_and_test_windows_never_overlap() -> None:
    folds = generate_walk_forward_folds(
        total_rows=1000,
        train_window=400,
        test_window=100,
        step_size=100,
        purge_gap=1,
    )

    for fold in folds:
        train_indexes = set(
            range(
                fold.train_start,
                fold.train_end,
            )
        )

        test_indexes = set(
            range(
                fold.test_start,
                fold.test_end,
            )
        )

        assert train_indexes.isdisjoint(test_indexes)


def test_training_window_never_contains_future_rows() -> None:
    folds = generate_walk_forward_folds(
        total_rows=1000,
        train_window=400,
        test_window=100,
        step_size=100,
        purge_gap=1,
    )

    for fold in folds:
        latest_training_index = fold.train_end - 1
        earliest_testing_index = fold.test_start

        assert latest_training_index < earliest_testing_index


def test_walk_forward_respects_purge_gap() -> None:
    purge_gap = 1

    folds = generate_walk_forward_folds(
        total_rows=1000,
        train_window=400,
        test_window=100,
        step_size=100,
        purge_gap=purge_gap,
    )

    for fold in folds:
        actual_gap = (
            fold.test_start
            - fold.train_end
        )

        assert actual_gap == purge_gap


def test_fold_does_not_exceed_dataset() -> None:
    total_rows = 1000

    folds = generate_walk_forward_folds(
        total_rows=total_rows,
        train_window=400,
        test_window=100,
        step_size=100,
        purge_gap=1,
    )

    for fold in folds:
        assert fold.train_end <= total_rows
        assert fold.test_end <= total_rows


@pytest.mark.parametrize(
    (
        "train_window",
        "test_window",
        "step_size",
        "purge_gap",
    ),
    [
        (0, 100, 100, 1),
        (400, 0, 100, 1),
        (400, 100, 0, 1),
        (400, 100, 100, -1),
    ],
)
def test_invalid_configuration_raises_error(
    train_window: int,
    test_window: int,
    step_size: int,
    purge_gap: int,
) -> None:
    with pytest.raises(ValueError):
        list(
            generate_walk_forward_folds(
                total_rows=1000,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                purge_gap=purge_gap,
            )
        )