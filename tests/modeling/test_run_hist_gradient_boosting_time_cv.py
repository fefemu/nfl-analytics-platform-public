"""
Tests for histogram gradient boosting time-CV.
"""

import pandas as pd
import pytest

from src.modeling.run_hist_gradient_boosting_time_cv import (
    aggregate_boosting_time_cv_results,
    run_hist_gradient_boosting_time_cv,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic multi-season model data."""

    rows = []

    for season in range(2018, 2025):
        for game_index in range(24):
            target = game_index % 2

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

            direction = (
                1.0 if target else -1.0
            )

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


def test_run_boosting_time_cv_evaluates_groups_and_folds() -> None:
    """Evaluate two feature groups on three folds."""

    data = create_development_frame()

    fold_results = (
        run_hist_gradient_boosting_time_cv(
            development_data=data
        )
    )

    assert len(fold_results) == 24

    assert set(
        fold_results["configuration_name"]
    ) == {
        "very_conservative",
        "conservative",
        "moderate",
        "original_baseline",
    }

    assert set(
        fold_results["model_name"]
    ) == {
        "elo_plus_qb",
        "full_core",
    }

    assert set(
        fold_results["validation_season"]
    ) == {
        2020,
        2021,
        2022,
    }


def test_aggregate_boosting_results_uses_game_weights() -> None:
    """Weight season metrics by validation games."""

    fold_results = pd.DataFrame(
        [
            {
                "model_name": "model",
                "configuration_name": "test_config",
                "feature_count": 2,
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
                "configuration_name": "test_config",
                "feature_count": 2,
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

    results = aggregate_boosting_time_cv_results(
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


def test_aggregate_boosting_results_orders_by_brier() -> None:
    """Rank feature groups by Brier score first."""

    fold_results = pd.DataFrame(
        [
            {
                "model_name": "worse",
                "configuration_name": "worse_config",
                "feature_count": 2,
                "validation_season": 2020,
                "game_count": 10,
                "accuracy": 0.70,
                "brier_score": 0.22,
                "log_loss": 0.60,
                "elo_accuracy": 0.60,
                "elo_brier_score": 0.23,
                "elo_log_loss": 0.65,
            },
            {
                "model_name": "better",
                "configuration_name": "better_config",
                "feature_count": 17,
                "validation_season": 2020,
                "game_count": 10,
                "accuracy": 0.60,
                "brier_score": 0.20,
                "log_loss": 0.64,
                "elo_accuracy": 0.60,
                "elo_brier_score": 0.23,
                "elo_log_loss": 0.65,
            },
        ]
    )

    results = aggregate_boosting_time_cv_results(
        fold_results
    )

    assert results.iloc[0][
        "model_name"
    ] == "better"


def test_aggregate_boosting_results_rejects_empty_data() -> None:
    """Reject an empty fold-result table."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        aggregate_boosting_time_cv_results(
            pd.DataFrame()
        )