"""Tests for current production game predictions."""

from datetime import datetime

import pandas as pd
import pytest

from src.modeling.current_game_predictions import (
    PREDICTION_COLUMNS,
    create_current_prediction_frame,
    is_neutral_location,
)


def create_upcoming_games() -> pd.DataFrame:
    """Create deterministic upcoming schedule inputs."""

    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": pd.Timestamp(
                    "2026-09-10"
                ),
                "gametime": "20:20",
                "home_team": "NE",
                "away_team": "NYJ",
                "location": "Home",
                "home_elo_rating": 1600.0,
                "away_elo_rating": 1400.0,
                "home_rating_season": 2025,
                "away_rating_season": 2025,
                "home_rating_as_of": pd.Timestamp(
                    "2026-01-10"
                ),
                "away_rating_as_of": pd.Timestamp(
                    "2026-01-11"
                ),
            },
            {
                "game_id": "2026_01_BUF_KC",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": pd.Timestamp(
                    "2026-09-11"
                ),
                "gametime": "19:00",
                "home_team": "BUF",
                "away_team": "KC",
                "location": "Neutral",
                "home_elo_rating": 1550.0,
                "away_elo_rating": 1550.0,
                "home_rating_season": 2025,
                "away_rating_season": 2025,
                "home_rating_as_of": pd.Timestamp(
                    "2026-01-12"
                ),
                "away_rating_as_of": pd.Timestamp(
                    "2026-01-12"
                ),
            },
        ]
    )


def test_is_neutral_location() -> None:
    """Normalize neutral-site location values."""

    assert is_neutral_location("Neutral")
    assert is_neutral_location(" NEUTRAL ")
    assert not is_neutral_location("Home")
    assert not is_neutral_location(None)


def test_create_current_prediction_frame() -> None:
    """Create one prediction row per upcoming game."""

    generated_at = datetime(
        2026,
        8,
        2,
        10,
        0,
        0,
    )

    predictions = create_current_prediction_frame(
        upcoming_games=create_upcoming_games(),
        generated_at=generated_at,
    )

    assert len(predictions) == 2
    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )
    assert set(predictions["model_name"]) == {
        "elo"
    }
    assert set(predictions["model_version"]) == {
        "1.0.0"
    }
    assert set(
        predictions["prediction_generated_at"]
    ) == {
        generated_at
    }


def test_prediction_frame_applies_offseason_regression() -> None:
    """Regress current ratings before 2026 week one."""

    predictions = create_current_prediction_frame(
        upcoming_games=create_upcoming_games(),
    )

    patriots_game = predictions.loc[
        predictions["game_id"]
        == "2026_01_NE_NYJ"
    ].iloc[0]

    assert patriots_game[
        "home_rating_pregame"
    ] == pytest.approx(1560.0)

    assert patriots_game[
        "away_rating_pregame"
    ] == pytest.approx(1440.0)

    assert patriots_game[
        "applied_home_advantage"
    ] == pytest.approx(50.0)


def test_prediction_frame_handles_neutral_site() -> None:
    """Remove home advantage from neutral games."""

    predictions = create_current_prediction_frame(
        upcoming_games=create_upcoming_games(),
    )

    neutral_game = predictions.loc[
        predictions["game_id"]
        == "2026_01_BUF_KC"
    ].iloc[0]

    assert bool(neutral_game["is_neutral"])
    assert neutral_game[
        "applied_home_advantage"
    ] == 0.0
    assert neutral_game[
        "home_win_probability"
    ] == pytest.approx(0.5)
    assert neutral_game["predicted_winner"] == "BUF"


def test_prediction_frame_rejects_duplicate_games() -> None:
    """Reject duplicate upcoming schedule records."""

    upcoming_games = create_upcoming_games()

    duplicated_games = pd.concat(
        [
            upcoming_games,
            upcoming_games.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        create_current_prediction_frame(
            duplicated_games
        )


def test_prediction_frame_supports_no_upcoming_games() -> None:
    """Return an empty frame with a stable schema."""

    empty_games = create_upcoming_games().iloc[0:0]

    predictions = create_current_prediction_frame(
        empty_games
    )

    assert predictions.empty
    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )