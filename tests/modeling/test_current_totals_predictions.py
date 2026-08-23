"""Tests for current totals predictions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modeling.current_totals_predictions import (
    CURRENT_TOTALS_PREDICTION_COLUMNS,
    create_current_totals_features,
    create_current_totals_prediction_frame,
)
from src.modeling.production_totals_component import (
    FALLBACK_PREDICTION_MODE,
    PRIMARY_PREDICTION_MODE,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)


def all_model_features() -> tuple[str, ...]:
    """Return the primary/fallback feature union."""

    return tuple(
        dict.fromkeys(
            (
                *PRODUCTION_TOTALS_MODEL
                .feature_columns,
                *PRODUCTION_TOTALS_MODEL
                .fallback_feature_columns,
            )
        )
    )


def create_historical_data() -> pd.DataFrame:
    """Create production training data."""

    rows: list[dict[str, object]] = []

    for index in range(80):
        features = {
            feature_name: float(
                1 + index % 9
            )
            for feature_name
            in all_model_features()
        }

        rows.append(
            {
                "game_id": f"history_{index}",
                "season": 2022 + index // 30,
                "both_short_windows_complete": True,
                "target_total_points": (
                    35.0
                    + 0.3 * index
                    + 0.2 * sum(features.values())
                ),
                **features,
            }
        )

    fallback_only = {
        **rows[-1],
        "game_id": "fallback_history",
        "season": 2025,
        "both_short_windows_complete": False,
        "offensive_epa_sum_last_4": np.nan,
        "defensive_epa_allowed_sum_last_4": (
            np.nan
        ),
        "listed_qb_rating_sum": np.nan,
    }

    rows.append(fallback_only)

    return pd.DataFrame(rows)


def create_upcoming_games() -> pd.DataFrame:
    """Create primary and fallback upcoming games."""

    common = {
        "season": 2026,
        "game_type": "REG",
        "week": 5,
        "gameday": "2026-10-04",
        "gametime": "13:00",
        "is_neutral": False,
        "home_elo_rating": 1520.0,
        "away_elo_rating": 1480.0,
        "is_indoor": False,
        "has_game_weather": True,
        "cold_degrees_below_50": 5.0,
        "heat_degrees_above_80": 0.0,
        "wind_mph_above_10": 3.0,
        "league_average_total_last_64": 45.5,
    }

    return pd.DataFrame(
        [
            {
                **common,
                "game_id": "primary_game",
                "home_team": "BUF",
                "away_team": "NYJ",
                "home_prior_season_games": 4,
                "away_prior_season_games": 5,
                (
                    "home_offensive_epa_per_"
                    "play_last_4"
                ): 0.10,
                (
                    "away_offensive_epa_per_"
                    "play_last_4"
                ): 0.05,
                (
                    "home_defensive_epa_allowed_"
                    "per_play_last_4"
                ): -0.02,
                (
                    "away_defensive_epa_allowed_"
                    "per_play_last_4"
                ): 0.03,
                "home_listed_qb_rating": 3.0,
                "away_listed_qb_rating": 1.0,
            },
            {
                **common,
                "game_id": "fallback_game",
                "home_team": "LV",
                "away_team": "DEN",
                "home_prior_season_games": 0,
                "away_prior_season_games": 0,
                (
                    "home_offensive_epa_per_"
                    "play_last_4"
                ): np.nan,
                (
                    "away_offensive_epa_per_"
                    "play_last_4"
                ): np.nan,
                (
                    "home_defensive_epa_allowed_"
                    "per_play_last_4"
                ): np.nan,
                (
                    "away_defensive_epa_allowed_"
                    "per_play_last_4"
                ): np.nan,
                "home_listed_qb_rating": np.nan,
                "away_listed_qb_rating": np.nan,
            },
        ]
    )


def test_create_primary_aggregate_features() -> None:
    """Create the selected symmetric totals features."""

    features = create_current_totals_features(
        create_upcoming_games()
    ).set_index(
        "game_id"
    )

    assert features.loc[
        "primary_game",
        "offensive_epa_sum_last_4",
    ] == pytest.approx(0.15)

    assert features.loc[
        "primary_game",
        "defensive_epa_allowed_sum_last_4",
    ] == pytest.approx(0.01)

    assert features.loc[
        "primary_game",
        "listed_qb_rating_sum",
    ] == pytest.approx(4.0)

    assert features.loc[
        "primary_game",
        "elo_rating_sum",
    ] == pytest.approx(3000.0)


def test_incomplete_windows_hide_rolling_features(
) -> None:
    """Do not treat partial windows as primary input."""

    features = create_current_totals_features(
        create_upcoming_games()
    ).set_index(
        "game_id"
    )

    assert not features.loc[
        "fallback_game",
        "both_short_windows_complete",
    ]

    assert pd.isna(
        features.loc[
            "fallback_game",
            "offensive_epa_sum_last_4",
        ]
    )


def test_prediction_frame_routes_both_modes() -> None:
    """Create primary and fallback predictions."""

    generated_at = datetime(
        2026,
        8,
        7,
        11,
        0,
        0,
    )

    predictions = (
        create_current_totals_prediction_frame(
            upcoming_games=create_upcoming_games(),
            historical_data=create_historical_data(),
            prediction_generated_at=generated_at,
        ).set_index(
            "game_id"
        )
    )

    assert predictions.loc[
        "primary_game",
        "prediction_mode",
    ] == PRIMARY_PREDICTION_MODE

    assert predictions.loc[
        "fallback_game",
        "prediction_mode",
    ] == FALLBACK_PREDICTION_MODE

    assert predictions.loc[
        "primary_game",
        "prediction_generated_at",
    ] == generated_at


def test_prediction_output_is_complete() -> None:
    """Return the documented finite output schema."""

    predictions = (
        create_current_totals_prediction_frame(
            upcoming_games=create_upcoming_games(),
            historical_data=create_historical_data(),
        )
    )

    assert tuple(
        predictions.columns
    ) == CURRENT_TOTALS_PREDICTION_COLUMNS

    assert np.isfinite(
        predictions["predicted_total_points"]
    ).all()


def test_missing_current_elo_is_rejected() -> None:
    """Require Elo for the universal fallback."""

    upcoming = create_upcoming_games()

    upcoming.loc[
        upcoming["game_id"] == "fallback_game",
        "home_elo_rating",
    ] = np.nan

    with pytest.raises(
        RuntimeError,
        match="missing current Elo ratings",
    ):
        create_current_totals_features(
            upcoming
        )


def test_duplicate_upcoming_game_is_rejected() -> None:
    """Require one row per upcoming game."""

    upcoming = create_upcoming_games()

    duplicate = pd.concat(
        [
            upcoming,
            upcoming.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        create_current_totals_features(
            duplicate
        )