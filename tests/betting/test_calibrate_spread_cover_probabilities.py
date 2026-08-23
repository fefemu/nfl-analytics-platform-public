"""
Tests for Spread cover probability calibration.
"""

import numpy as np
import pandas as pd
import pytest

from src.betting.calibrate_spread_cover_probabilities import (
    RESIDUAL_COLUMNS,
    SUMMARY_COLUMNS,
    create_spread_calibration_residuals,
    summarize_spread_calibration,
)
from src.modeling.production_spread_component import (
    FALLBACK_PREDICTION_MODE,
    PRIMARY_PREDICTION_MODE,
)


TEST_VALIDATION_SEASONS = (
    2021,
    2022,
)


def create_development_data() -> pd.DataFrame:
    """Create chronological synthetic spread data."""

    return pd.DataFrame(
        [
            {
                "game_id": "2020_game_1",
                "season": 2020,
                "split_name": "train",
                "target_point_differential": 3.0,
                "external_nfelo_rating_difference": 50.0,
                "external_nfelo_qb_adjustment_difference": 0.5,
            },
            {
                "game_id": "2021_game_1",
                "season": 2021,
                "split_name": "train",
                "target_point_differential": 7.0,
                "external_nfelo_rating_difference": 100.0,
                "external_nfelo_qb_adjustment_difference": 1.0,
            },
            {
                "game_id": "2021_game_2",
                "season": 2021,
                "split_name": "train",
                "target_point_differential": -4.0,
                "external_nfelo_rating_difference": -75.0,
                "external_nfelo_qb_adjustment_difference": -0.5,
            },
            {
                "game_id": "2022_game_1",
                "season": 2022,
                "split_name": "validation",
                "target_point_differential": 10.0,
                "external_nfelo_rating_difference": 125.0,
                "external_nfelo_qb_adjustment_difference": 1.5,
            },
            {
                "game_id": "2022_game_2",
                "season": 2022,
                "split_name": "validation",
                "target_point_differential": -7.0,
                "external_nfelo_rating_difference": -100.0,
                "external_nfelo_qb_adjustment_difference": -1.0,
            },
        ]
    )


def test_create_spread_calibration_residual_schema(
) -> None:
    """Create documented residual rows for both modes."""

    residuals = (
        create_spread_calibration_residuals(
            development_data=(
                create_development_data()
            ),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    assert tuple(residuals.columns) == (
        RESIDUAL_COLUMNS
    )

    assert set(
        residuals["prediction_mode"]
    ) == {
        PRIMARY_PREDICTION_MODE,
        FALLBACK_PREDICTION_MODE,
    }

    assert len(residuals) == 4


def test_calibration_is_chronological() -> None:
    """Each fold trains only on earlier seasons."""

    residuals = (
        create_spread_calibration_residuals(
            development_data=(
                create_development_data()
            ),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    season_2021 = residuals.loc[
        residuals["validation_season"] == 2021
    ]

    season_2022 = residuals.loc[
        residuals["validation_season"] == 2022
    ]

    assert season_2021[
        "training_game_count"
    ].eq(1).all()

    assert season_2022[
        "training_game_count"
    ].eq(3).all()


def test_residual_matches_actual_minus_prediction(
) -> None:
    """Stored residuals reproduce actual model errors."""

    residuals = (
        create_spread_calibration_residuals(
            development_data=(
                create_development_data()
            ),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    calculated_residual = (
        residuals["actual_home_margin"]
        - residuals["predicted_home_margin"]
    )

    assert np.allclose(
        residuals[
            "residual_home_margin"
        ].to_numpy(dtype=float),
        calculated_residual.to_numpy(
            dtype=float
        ),
    )

    assert np.allclose(
        residuals[
            "absolute_error"
        ].to_numpy(dtype=float),
        np.abs(
            calculated_residual.to_numpy(
                dtype=float
            )
        ),
    )


def test_summarize_spread_calibration() -> None:
    """Summarize both production routing modes."""

    residuals = (
        create_spread_calibration_residuals(
            development_data=(
                create_development_data()
            ),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    summary = summarize_spread_calibration(
        residuals
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert len(summary) == 1

    assert summary[
        "fold_count"
    ].eq(2).all()

    assert summary[
        "validation_game_count"
    ].eq(4).all()

    assert summary[
        "residual_standard_deviation"
    ].gt(0.0).all()

    assert summary[
        "root_mean_squared_error"
    ].ge(
        summary["mean_absolute_error"]
    ).all()


def test_holdout_data_is_rejected() -> None:
    """The calibration layer must never accept holdout."""

    development_data = (
        create_development_data()
    )

    development_data.loc[
        development_data["season"] == 2022,
        "split_name",
    ] = "holdout"

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        create_spread_calibration_residuals(
            development_data=development_data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_2025_data_is_rejected() -> None:
    """The protected 2025 holdout season is rejected."""

    development_data = (
        create_development_data()
    )

    extra_row = development_data.iloc[
        [0]
    ].copy()

    extra_row["game_id"] = "2025_game_1"
    extra_row["season"] = 2025
    extra_row["split_name"] = "validation"

    development_data = pd.concat(
        [
            development_data,
            extra_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="before the 2025 holdout",
    ):
        create_spread_calibration_residuals(
            development_data=development_data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_missing_feature_is_rejected() -> None:
    """Reject calibration data with missing schema."""

    development_data = (
        create_development_data().drop(
            columns=[
                    "external_nfelo_qb_adjustment_difference",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_spread_calibration_residuals(
            development_data=development_data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_duplicate_game_is_rejected() -> None:
    """Reject duplicate development game identifiers."""

    development_data = (
        create_development_data()
    )

    development_data = pd.concat(
        [
            development_data,
            development_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        create_spread_calibration_residuals(
            development_data=development_data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_missing_validation_season_is_rejected(
) -> None:
    """Every requested fold season must exist."""

    with pytest.raises(
        ValueError,
        match="Calibration seasons are missing",
    ):
        create_spread_calibration_residuals(
            development_data=(
                create_development_data()
            ),
            validation_seasons=(
                2021,
                2023,
            ),
        )
