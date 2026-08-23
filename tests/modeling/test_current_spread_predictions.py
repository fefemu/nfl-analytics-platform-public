"""Tests for current spread prediction frames."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.modeling.current_spread_predictions import (
    CURRENT_SPREAD_PREDICTION_COLUMNS,
    create_current_spread_features,
    create_current_spread_prediction_frame,
)


def create_upcoming_games() -> pd.DataFrame:
    """Create complete and missing-QB games."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "BUF",
                "away_team": "NYJ",
                "location": "Home",
                "home_listed_qb_rating": 7.0,
                "away_listed_qb_rating": 3.0,
                "external_nfelo_rating_difference": 90.0,
                "external_nfelo_qb_adjustment_difference": 5.0,
            },
            {
                "game_id": "game_2",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "LV",
                "away_team": "DEN",
                "location": "Home",
                "home_listed_qb_rating": np.nan,
                "away_listed_qb_rating": 5.0,
                "external_nfelo_rating_difference": -70.0,
                "external_nfelo_qb_adjustment_difference": -3.0,
            },
        ]
    )


def create_elo_predictions() -> pd.DataFrame:
    """Create matching current Elo outputs."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "home_rating_pregame": 1580.0,
                "away_rating_pregame": 1500.0,
                "is_neutral": False,
            },
            {
                "game_id": "game_2",
                "home_rating_pregame": 1460.0,
                "away_rating_pregame": 1540.0,
                "is_neutral": False,
            },
        ]
    )


def create_historical_data() -> pd.DataFrame:
    """Create historical model-training games."""

    rows: list[dict[str, object]] = []

    for index in range(100):
        elo_difference = float(
            -180 + index * 4
        )

        qb_difference = float(
            -8 + index * 0.2
        )

        rows.append(
            {
                "game_id": f"history_{index}",
                "season": 2021 + index // 25,
                "target_point_differential": (
                    2.0
                    + 0.04 * elo_difference
                    + 0.9 * qb_difference
                ),
                "external_nfelo_rating_difference": elo_difference,
                "external_nfelo_qb_adjustment_difference": qb_difference,
                "listed_qb_rating_difference": (
                    qb_difference
                ),
            }
        )

    return pd.DataFrame(rows)


def test_create_features_derives_differences() -> None:
    """Derive current Elo and QB feature definitions."""

    features = create_current_spread_features(
        upcoming_games=create_upcoming_games(),
        elo_predictions=create_elo_predictions(),
    ).set_index(
        "game_id"
    )

    assert (
        features.loc[
            "game_1",
            "external_nfelo_rating_difference",
        ]
        == pytest.approx(90.0)
    )

    assert (
        features.loc[
            "game_1",
            "listed_qb_rating_difference",
        ]
        == pytest.approx(4.0)
    )

    assert bool(
        features.loc[
            "game_1",
            "both_listed_qb_ratings_available",
        ]
    )


def test_missing_qb_is_preserved_for_fallback() -> None:
    """Keep games whose listed QB rating is missing."""

    features = create_current_spread_features(
        upcoming_games=create_upcoming_games(),
        elo_predictions=create_elo_predictions(),
    ).set_index(
        "game_id"
    )

    assert pd.isna(
        features.loc[
            "game_2",
            "listed_qb_rating_difference",
        ]
    )

    assert not bool(
        features.loc[
            "game_2",
            "both_listed_qb_ratings_available",
        ]
    )


def test_create_current_prediction_frame() -> None:
    """Create auditable primary and fallback outputs."""

    generated_at = datetime(
        2026,
        8,
        6,
        15,
        0,
        0,
    )

    predictions = (
        create_current_spread_prediction_frame(
            upcoming_games=create_upcoming_games(),
            elo_predictions=create_elo_predictions(),
            historical_data=create_historical_data(),
            prediction_generated_at=generated_at,
        )
    )

    assert tuple(
        predictions.columns
    ) == CURRENT_SPREAD_PREDICTION_COLUMNS

    assert list(
        predictions["game_id"]
    ) == [
        "game_1",
        "game_2",
    ]

    assert set(
        predictions["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_QB_RIDGE",
    }

    assert set(
        predictions[
            "primary_training_game_count"
        ]
    ) == {
        100,
    }

    assert (
        predictions["prediction_generated_at"]
        == generated_at
    ).all()


def test_prediction_winner_matches_margin() -> None:
    """Keep winner and predicted margin consistent."""

    predictions = (
        create_current_spread_prediction_frame(
            upcoming_games=create_upcoming_games(),
            elo_predictions=create_elo_predictions(),
            historical_data=create_historical_data(),
        )
    )

    for prediction in predictions.itertuples(
        index=False
    ):
        expected_winner = (
            prediction.home_team
            if prediction.predicted_home_margin >= 0.0
            else prediction.away_team
        )

        assert (
            prediction.predicted_winner
            == expected_winner
        )


def test_missing_elo_prediction_is_rejected() -> None:
    """Require an Elo output for every upcoming game."""

    incomplete_elo = (
        create_elo_predictions().iloc[[0]].copy()
    )

    with pytest.raises(
        RuntimeError,
        match="missing current Elo predictions",
    ):
        create_current_spread_features(
            upcoming_games=create_upcoming_games(),
            elo_predictions=incomplete_elo,
        )


def test_duplicate_upcoming_games_are_rejected() -> None:
    """Require one current row per game."""

    upcoming = create_upcoming_games()

    duplicated = pd.concat(
        [
            upcoming,
            upcoming.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        create_current_spread_features(
            upcoming_games=duplicated,
            elo_predictions=create_elo_predictions(),
        )
