"""
Tests for expanding-window time-series validation.
"""

import pandas as pd
import pytest

from src.modeling.time_series_validation import (
    create_expanding_season_fold,
    create_expanding_season_folds,
)


def create_development_frame() -> pd.DataFrame:
    """Create train and external-validation seasons."""

    rows = []

    for season in range(2018, 2025):
        rows.append(
            {
                "game_id": f"{season}_01_A_B",
                "season": season,
                "game_date": pd.Timestamp(
                    f"{season}-09-01"
                ),
                "split_name": (
                    "train"
                    if season <= 2022
                    else "validation"
                ),
            }
        )

    return pd.DataFrame(rows)


def test_create_expanding_fold_uses_only_earlier_train_seasons() -> None:
    """Build an expanding train window and one validation season."""

    data = create_development_frame()

    fold = create_expanding_season_fold(
        development_data=data,
        validation_season=2021,
    )

    fold_data = fold.development_data

    train_seasons = set(
        fold_data.loc[
            fold_data["split_name"] == "train",
            "season",
        ]
    )

    validation_seasons = set(
        fold_data.loc[
            fold_data["split_name"] == "validation",
            "season",
        ]
    )

    assert train_seasons == {
        2018,
        2019,
        2020,
    }

    assert validation_seasons == {
        2021,
    }


def test_create_expanding_fold_excludes_external_validation() -> None:
    """Exclude 2023 and 2024 from internal CV folds."""

    data = create_development_frame()

    fold = create_expanding_season_fold(
        development_data=data,
        validation_season=2022,
    )

    assert set(
        fold.development_data["season"]
    ) == {
        2018,
        2019,
        2020,
        2021,
        2022,
    }


def test_create_default_folds_uses_configured_seasons() -> None:
    """Create the 2020, 2021 and 2022 folds."""

    data = create_development_frame()

    folds = create_expanding_season_folds(data)

    assert tuple(
        fold.validation_season
        for fold in folds
    ) == (
        2020,
        2021,
        2022,
    )


def test_create_expanding_fold_requires_prior_season() -> None:
    """Reject a fold without earlier training data."""

    data = create_development_frame()

    with pytest.raises(
        ValueError,
        match="no earlier training seasons",
    ):
        create_expanding_season_fold(
            development_data=data,
            validation_season=2018,
        )


@pytest.mark.parametrize(
    "validation_seasons",
    [
        (),
        (2020, 2020),
        (2021, 2020),
    ],
)
def test_create_expanding_folds_rejects_invalid_seasons(
    validation_seasons: tuple[int, ...],
) -> None:
    """Reject empty, duplicate or unordered fold seasons."""

    data = create_development_frame()

    with pytest.raises(ValueError):
        create_expanding_season_folds(
            development_data=data,
            validation_seasons=validation_seasons,
        )