"""
NFL Analytics Platform
Time-Series Model Validation

Purpose:
    Create leakage-safe expanding-window season folds
    from the model training period.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import pandas as pd

from src.modeling.train_logistic_baseline import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


CV_VALIDATION_SEASONS = (
    2020,
    2021,
    2022,
)


@dataclass(frozen=True)
class ExpandingSeasonFold:
    """Store one expanding-window validation fold."""

    validation_season: int
    development_data: pd.DataFrame


def create_expanding_season_fold(
    development_data: pd.DataFrame,
    validation_season: int,
) -> ExpandingSeasonFold:
    """Create one fold using earlier seasons for training."""

    required_columns = {
        "game_id",
        "season",
        "game_date",
        "split_name",
    }

    missing_columns = sorted(
        required_columns - set(development_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Development data is missing columns: "
            + ", ".join(missing_columns)
        )

    training_pool = development_data.loc[
        development_data["split_name"] == TRAIN_SPLIT
    ].copy()

    if training_pool.empty:
        raise ValueError(
            "Development data contains no training games."
        )

    fold_data = training_pool.loc[
        training_pool["season"] <= validation_season
    ].copy()

    fold_data["split_name"] = VALIDATION_SPLIT

    fold_data.loc[
        fold_data["season"] < validation_season,
        "split_name",
    ] = TRAIN_SPLIT

    train_data = fold_data.loc[
        fold_data["split_name"] == TRAIN_SPLIT
    ]

    validation_data = fold_data.loc[
        fold_data["split_name"] == VALIDATION_SPLIT
    ]

    if train_data.empty:
        raise ValueError(
            "Expanding fold has no earlier training seasons "
            f"for validation season {validation_season}."
        )

    if validation_data.empty:
        raise ValueError(
            "Expanding fold has no validation games for "
            f"season {validation_season}."
        )

    latest_train_date = train_data[
        "game_date"
    ].max()

    earliest_validation_date = validation_data[
        "game_date"
    ].min()

    if latest_train_date >= earliest_validation_date:
        raise ValueError(
            "Expanding fold chronology is invalid for "
            f"validation season {validation_season}."
        )

    fold_data = fold_data.sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(drop=True)

    return ExpandingSeasonFold(
        validation_season=validation_season,
        development_data=fold_data,
    )


def create_expanding_season_folds(
    development_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = CV_VALIDATION_SEASONS,
) -> tuple[ExpandingSeasonFold, ...]:
    """Create all configured expanding-window folds."""

    if not validation_seasons:
        raise ValueError(
            "At least one CV validation season is required."
        )

    if len(validation_seasons) != len(
        set(validation_seasons)
    ):
        raise ValueError(
            "CV validation seasons must be unique."
        )

    if tuple(sorted(validation_seasons)) != (
        validation_seasons
    ):
        raise ValueError(
            "CV validation seasons must be chronological."
        )

    return tuple(
        create_expanding_season_fold(
            development_data=development_data,
            validation_season=validation_season,
        )
        for validation_season in validation_seasons
    )