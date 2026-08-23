"""
Tests for logistic expanding-window validation.
"""

import pandas as pd
import pytest

from src.modeling.run_logistic_time_cv import (
    aggregate_time_cv_results,
    run_logistic_time_cv,
    select_best_per_feature_group,
    validate_cv_configuration,
    select_best_model_fold_results,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic multi-season model data."""

    rows = []

    for season in range(2018, 2025):
        for game_index in range(4):
            target = game_index % 2
            direction = (
                1.0 if target else -1.0
            )

            row = {
                "game_id": (
                    f"{season}_{game_index}_A_B"
                ),
                "season": season,
                "game_date": pd.Timestamp(
                    f"{season}-09-01"
                )
                + pd.Timedelta(
                    days=game_index
                ),
                "split_name": (
                    "train"
                    if season <= 2022
                    else "validation"
                ),
                TARGET_COLUMN: target,
                "elo_home_win_probability": (
                    0.65 if target else 0.35
                ),
            }

            for (
                feature_index,
                feature_name,
            ) in enumerate(
                MODEL_FEATURE_COLUMNS
            ):
                row[feature_name] = (
                    direction
                    * (
                        1.0
                        + 0.01 * feature_index
                    )
                )

            rows.append(row)

    return pd.DataFrame(rows)


def test_run_logistic_time_cv_evaluates_every_combination() -> None:
    """Evaluate every model, C and season fold."""

    data = create_development_frame()

    feature_groups = {
        "elo_only": (
            "elo_rating_difference",
        ),
        "elo_qb": (
            "elo_rating_difference",
            "listed_qb_rating_difference",
        ),
    }

    fold_results = run_logistic_time_cv(
        development_data=data,
        feature_groups=feature_groups,
        regularization_grid=(
            0.1,
            1.0,
        ),
    )

    assert len(fold_results) == 12

    assert set(
        fold_results["validation_season"]
    ) == {
        2020,
        2021,
        2022,
    }


def test_aggregate_time_cv_uses_game_weights() -> None:
    """Weight fold metrics by validation game count."""

    fold_results = pd.DataFrame(
        [
            {
                "model_name": "model",
                "feature_count": 1,
                "regularization_c": 1.0,
                "validation_season": 2020,
                "game_count": 1,
                "accuracy": 1.0,
                "brier_score": 0.10,
                "log_loss": 0.30,
                "elo_accuracy": 0.0,
                "elo_brier_score": 0.30,
                "elo_log_loss": 0.70,
            },
            {
                "model_name": "model",
                "feature_count": 1,
                "regularization_c": 1.0,
                "validation_season": 2021,
                "game_count": 3,
                "accuracy": 0.0,
                "brier_score": 0.30,
                "log_loss": 0.70,
                "elo_accuracy": 1.0,
                "elo_brier_score": 0.10,
                "elo_log_loss": 0.30,
            },
        ]
    )

    results = aggregate_time_cv_results(
        fold_results
    )

    row = results.iloc[0]

    assert row["game_count"] == 4
    assert row["fold_count"] == 2
    assert row["accuracy"] == pytest.approx(
        0.25
    )
    assert row["brier_score"] == pytest.approx(
        0.25
    )
    assert row["log_loss"] == pytest.approx(
        0.60
    )


def test_select_best_per_feature_group_uses_brier_first() -> None:
    """Select one lowest-Brier result per model."""

    results = pd.DataFrame(
        [
            {
                "model_name": "model",
                "feature_count": 1,
                "regularization_c": 1.0,
                "brier_score": 0.22,
                "log_loss": 0.60,
            },
            {
                "model_name": "model",
                "feature_count": 1,
                "regularization_c": 0.1,
                "brier_score": 0.21,
                "log_loss": 0.65,
            },
        ]
    )

    best_results = select_best_per_feature_group(
        results
    )

    assert best_results.iloc[0][
        "regularization_c"
    ] == pytest.approx(0.1)


@pytest.mark.parametrize(
    (
        "feature_groups",
        "regularization_grid",
    ),
    [
        (
            {},
            (1.0,),
        ),
        (
            {
                "model": (),
            },
            (1.0,),
        ),
        (
            {
                "model": (
                    "elo_rating_difference",
                ),
            },
            (),
        ),
        (
            {
                "model": (
                    "elo_rating_difference",
                ),
            },
            (0.0,),
        ),
    ],
)
def test_validate_cv_configuration_rejects_invalid_values(
    feature_groups: dict[
        str,
        tuple[str, ...],
    ],
    regularization_grid: tuple[
        float, ...
    ],
) -> None:
    """Reject invalid CV configurations."""

    with pytest.raises(ValueError):
        validate_cv_configuration(
            feature_groups=feature_groups,
            regularization_grid=regularization_grid,
        )


def test_select_best_model_fold_results_returns_seasons() -> None:
    """Return fold rows for the best aggregate model."""

    fold_results = pd.DataFrame(
        [
            {
                "model_name": "best_model",
                "regularization_c": 0.1,
                "validation_season": 2020,
            },
            {
                "model_name": "best_model",
                "regularization_c": 0.1,
                "validation_season": 2021,
            },
            {
                "model_name": "other_model",
                "regularization_c": 1.0,
                "validation_season": 2020,
            },
        ]
    )

    aggregate_results = pd.DataFrame(
        [
            {
                "model_name": "best_model",
                "regularization_c": 0.1,
                "brier_score": 0.20,
            },
            {
                "model_name": "other_model",
                "regularization_c": 1.0,
                "brier_score": 0.21,
            },
        ]
    )

    selected_results = (
        select_best_model_fold_results(
            fold_results=fold_results,
            aggregate_results=aggregate_results,
        )
    )

    assert list(
        selected_results["validation_season"]
    ) == [
        2020,
        2021,
    ]

    assert set(
        selected_results["model_name"]
    ) == {
        "best_model",
    }