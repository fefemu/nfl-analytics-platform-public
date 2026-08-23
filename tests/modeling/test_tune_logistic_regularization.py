"""
Tests for logistic-regression regularization tuning.
"""

import pandas as pd
import pytest

from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)
from src.modeling.tune_logistic_regularization import (
    run_regularization_tuning,
    select_best_per_feature_group,
    validate_regularization_grid,
)
from src.modeling.run_logistic_ablation import (
    FEATURE_GROUPS,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic train and validation data."""

    rows = []

    for index in range(20):
        split_name = (
            "train"
            if index < 16
            else "validation"
        )

        target = index % 2
        direction = 1.0 if target else -1.0

        row = {
            "game_id": f"game_{index}",
            "season": 2020,
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": split_name,
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        for feature_index, feature_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    1.0
                    + 0.01 * feature_index
                )
            )

        for (
            feature_index,
            feature_name,
        ) in enumerate(
            SCHEDULE_CONTEXT_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    0.10
                    + 0.01 * feature_index
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "regularization_grid",
    [
        (),
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.1, 0.1),
    ],
)
def test_validate_regularization_grid_rejects_invalid_grid(
    regularization_grid: tuple[float, ...],
) -> None:
    """Reject empty, invalid or duplicate C grids."""

    with pytest.raises(ValueError):
        validate_regularization_grid(
            regularization_grid
        )


def test_run_regularization_tuning_tests_every_combination() -> None:
    """Evaluate every feature-group and C combination."""

    data = create_development_frame()

    results = run_regularization_tuning(
        development_data=data,
        regularization_grid=(
            0.1,
            1.0,
        ),
    )

    assert len(results) == (
        len(FEATURE_GROUPS) * 2
    )
    assert set(results["regularization_c"]) == {
        0.1,
        1.0,
    }


def test_select_best_per_feature_group_returns_one_row_each() -> None:
    """Select one best result for every model."""

    results = pd.DataFrame(
        [
            {
                "model_name": "model_a",
                "feature_count": 1,
                "regularization_c": 1.0,
                "brier_score": 0.22,
                "log_loss": 0.64,
            },
            {
                "model_name": "model_a",
                "feature_count": 1,
                "regularization_c": 0.1,
                "brier_score": 0.21,
                "log_loss": 0.63,
            },
            {
                "model_name": "model_b",
                "feature_count": 2,
                "regularization_c": 1.0,
                "brier_score": 0.20,
                "log_loss": 0.62,
            },
        ]
    )

    best_results = select_best_per_feature_group(
        results
    )

    assert len(best_results) == 2

    model_a = best_results.loc[
        best_results["model_name"] == "model_a"
    ].iloc[0]

    assert model_a["regularization_c"] == pytest.approx(
        0.1
    )