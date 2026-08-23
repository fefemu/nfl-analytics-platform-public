"""Tests for production Elo game prediction."""

import pytest

from src.modeling.predict_elo_games import (
    create_elo_pregame_prediction,
    regress_rating_across_seasons,
)


def test_rating_is_unchanged_within_same_season() -> None:
    """Do not regress a current in-season rating."""

    rating = regress_rating_across_seasons(
        rating=1600.0,
        last_completed_season=2025,
        target_season=2025,
    )

    assert rating == 1600.0


def test_rating_regresses_across_one_offseason() -> None:
    """Regress one season toward the 1500 Elo mean."""

    rating = regress_rating_across_seasons(
        rating=1600.0,
        last_completed_season=2025,
        target_season=2026,
    )

    assert rating == pytest.approx(1560.0)


def test_rating_regresses_across_multiple_offseasons() -> None:
    """Apply retention once for every crossed offseason."""

    rating = regress_rating_across_seasons(
        rating=1600.0,
        last_completed_season=2024,
        target_season=2026,
    )

    assert rating == pytest.approx(1536.0)


def test_rating_rejects_past_target_season() -> None:
    """Reject predictions before the rating season."""

    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        regress_rating_across_seasons(
            rating=1500.0,
            last_completed_season=2025,
            target_season=2024,
        )


def test_create_home_game_prediction() -> None:
    """Apply the production home-field advantage."""

    prediction = create_elo_pregame_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1550.0,
        away_rating=1500.0,
        home_last_completed_season=2025,
        away_last_completed_season=2025,
        target_season=2025,
    )

    assert prediction.applied_home_advantage == 50.0
    assert prediction.home_rating_pregame == 1550.0
    assert prediction.away_rating_pregame == 1500.0
    assert prediction.home_win_probability == (
        pytest.approx(0.640065)
    )
    assert prediction.away_win_probability == (
        pytest.approx(0.359935)
    )
    assert prediction.predicted_winner == "NE"


def test_create_neutral_game_prediction() -> None:
    """Remove home advantage at a neutral venue."""

    prediction = create_elo_pregame_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1550.0,
        away_rating=1500.0,
        home_last_completed_season=2025,
        away_last_completed_season=2025,
        target_season=2025,
        is_neutral=True,
    )

    assert prediction.applied_home_advantage == 0.0
    assert prediction.home_win_probability == (
        pytest.approx(0.571463)
    )


def test_create_prediction_applies_offseason_regression() -> None:
    """Use regressed ratings for a future season."""

    prediction = create_elo_pregame_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1600.0,
        away_rating=1400.0,
        home_last_completed_season=2025,
        away_last_completed_season=2025,
        target_season=2026,
    )

    assert prediction.home_rating_pregame == (
        pytest.approx(1560.0)
    )
    assert prediction.away_rating_pregame == (
        pytest.approx(1440.0)
    )


def test_create_prediction_rejects_same_team() -> None:
    """Reject an invalid self-matchup."""

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        create_elo_pregame_prediction(
            home_team="NE",
            away_team="NE",
            home_rating=1500.0,
            away_rating=1500.0,
            home_last_completed_season=2025,
            away_last_completed_season=2025,
            target_season=2025,
        )