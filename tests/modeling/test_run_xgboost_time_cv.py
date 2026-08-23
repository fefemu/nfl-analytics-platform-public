"""
Tests for XGBoost expanding-window validation.
"""

import pandas as pd

from src.modeling.run_xgboost_time_cv import (
    run_xgboost_time_cv,
)
from src.modeling.train_xgboost import (
    XGBOOST_CONFIGURATIONS,
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


def test_run_xgboost_time_cv_evaluates_groups_and_folds() -> None:
    """Evaluate two feature groups on three folds."""

    data = create_development_frame()

    fold_results = run_xgboost_time_cv(
        development_data=data
    )

    assert len(fold_results) == 24

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

    assert set(
        fold_results["configuration_name"]
    ) == set(
        XGBOOST_CONFIGURATIONS
    )


def test_run_xgboost_time_cv_excludes_external_validation() -> None:
    """Use only internal training-period validation seasons."""

    data = create_development_frame()

    fold_results = run_xgboost_time_cv(
        development_data=data
    )

    assert 2023 not in set(
        fold_results["validation_season"]
    )

    assert 2024 not in set(
        fold_results["validation_season"]
    )