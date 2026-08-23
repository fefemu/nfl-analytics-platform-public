"""Tests for the core Elo rating model."""

import pytest

from src.models.elo import (
    calculate_actual_score,
    calculate_expected_probability,
    calculate_rating_change,
    process_game,
    regress_rating_to_mean,
    update_ratings,
)


def test_equal_ratings_produce_equal_probability() -> None:
    """Equal ratings on a neutral field should produce 50 percent."""

    probability = calculate_expected_probability(
        team_rating=1500.0,
        opponent_rating=1500.0,
    )

    assert probability == pytest.approx(0.5)


def test_stronger_team_has_higher_probability() -> None:
    """A stronger team should have a win probability above 50 percent."""

    probability = calculate_expected_probability(
        team_rating=1600.0,
        opponent_rating=1500.0,
    )

    assert probability == pytest.approx(
        0.640065,
        abs=0.000001,
    )


def test_rating_advantage_increases_probability() -> None:
    """A positive rating advantage should increase win probability."""

    neutral_probability = calculate_expected_probability(
        team_rating=1500.0,
        opponent_rating=1500.0,
    )
    advantaged_probability = calculate_expected_probability(
        team_rating=1500.0,
        opponent_rating=1500.0,
        rating_advantage=50.0,
    )

    assert advantaged_probability > neutral_probability


def test_expected_win_produces_positive_rating_change() -> None:
    """A win should produce a positive rating change."""

    rating_change = calculate_rating_change(
        actual_score=1.0,
        expected_probability=0.5,
        k_factor=20.0,
    )

    assert rating_change == pytest.approx(10.0)


def test_expected_loss_produces_negative_rating_change() -> None:
    """A loss should produce a negative rating change."""

    rating_change = calculate_rating_change(
        actual_score=0.0,
        expected_probability=0.5,
        k_factor=20.0,
    )

    assert rating_change == pytest.approx(-10.0)


def test_update_ratings_preserves_total_rating_points() -> None:
    """Rating points gained by one team should be lost by the opponent."""

    updated_team_rating, updated_opponent_rating = update_ratings(
        team_rating=1500.0,
        opponent_rating=1500.0,
        actual_score=1.0,
        expected_probability=0.5,
        k_factor=20.0,
    )

    assert updated_team_rating == pytest.approx(1510.0)
    assert updated_opponent_rating == pytest.approx(1490.0)
    assert (
        updated_team_rating + updated_opponent_rating
        == pytest.approx(3000.0)
    )


def test_tie_between_equal_teams_does_not_change_ratings() -> None:
    """A tie between equally rated teams should not change ratings."""

    updated_team_rating, updated_opponent_rating = update_ratings(
        team_rating=1500.0,
        opponent_rating=1500.0,
        actual_score=0.5,
        expected_probability=0.5,
        k_factor=20.0,
    )

    assert updated_team_rating == pytest.approx(1500.0)
    assert updated_opponent_rating == pytest.approx(1500.0)


def test_regress_above_average_rating_toward_mean() -> None:
    """Regress an above-average rating toward the league mean."""

    regressed_rating = regress_rating_to_mean(
        rating=1600.0,
        mean_rating=1500.0,
        retention=0.70,
    )

    assert regressed_rating == pytest.approx(1570.0)


def test_regress_below_average_rating_toward_mean() -> None:
    """Regress a below-average rating toward the league mean."""

    regressed_rating = regress_rating_to_mean(
        rating=1400.0,
        mean_rating=1500.0,
        retention=0.70,
    )

    assert regressed_rating == pytest.approx(1430.0)


def test_league_average_rating_remains_unchanged() -> None:
    """The league-average rating should remain at the mean."""

    regressed_rating = regress_rating_to_mean(
        rating=1500.0,
        mean_rating=1500.0,
        retention=0.70,
    )

    assert regressed_rating == pytest.approx(1500.0)


@pytest.mark.parametrize(
    "invalid_retention",
    [-0.1, 1.1],
)
def test_invalid_season_retention_is_rejected(
    invalid_retention: float,
) -> None:
    """Reject season retention values outside the valid range."""

    with pytest.raises(
        ValueError,
        match="Season retention must be between 0 and 1.",
    ):
        regress_rating_to_mean(
            rating=1600.0,
            retention=invalid_retention,
        )


@pytest.mark.parametrize(
    ("team_score", "opponent_score", "expected_actual_score"),
    [
        (27, 20, 1.0),
        (20, 27, 0.0),
        (20, 20, 0.5),
    ],
)
def test_calculate_actual_score(
    team_score: int,
    opponent_score: int,
    expected_actual_score: float,
) -> None:
    """Convert final scores into the correct Elo result value."""

    actual_score = calculate_actual_score(
        team_score=team_score,
        opponent_score=opponent_score,
    )

    assert actual_score == pytest.approx(
        expected_actual_score
    )


def test_process_game_creates_complete_elo_result() -> None:
    """Create pregame probabilities and postgame ratings."""

    result = process_game(
        home_rating=1500.0,
        away_rating=1500.0,
        home_score=27,
        away_score=20,
        home_advantage=0.0,
        k_factor=20.0,
    )

    assert result.home_rating_pre == pytest.approx(1500.0)
    assert result.away_rating_pre == pytest.approx(1500.0)
    assert result.home_win_probability == pytest.approx(0.5)
    assert result.away_win_probability == pytest.approx(0.5)
    assert result.actual_home_score == pytest.approx(1.0)
    assert result.home_rating_post == pytest.approx(1510.0)
    assert result.away_rating_post == pytest.approx(1490.0)
    assert result.home_rating_change == pytest.approx(10.0)


def test_process_game_applies_home_advantage_before_update() -> None:
    """Apply home advantage to the pregame probability."""

    result = process_game(
        home_rating=1500.0,
        away_rating=1500.0,
        home_score=27,
        away_score=20,
        home_advantage=50.0,
        k_factor=20.0,
    )

    assert result.home_win_probability > 0.5
    assert (
        result.home_win_probability
        + result.away_win_probability
        == pytest.approx(1.0)
    )
    assert result.home_rating_change < 10.0