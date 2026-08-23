"""Tests for production Elo prediction explanations."""

from math import exp

import pytest

from src.modeling.explain_elo_prediction import (
    classify_matchup,
    explain_elo_prediction,
)


def test_classify_matchup_strength() -> None:
    """Classify adjusted Elo matchup edges."""

    assert classify_matchup(0.0) == "toss_up"
    assert classify_matchup(50.0) == "slight_edge"
    assert classify_matchup(-100.0) == "clear_edge"
    assert classify_matchup(175.0) == "strong_edge"


def test_explain_home_favorite() -> None:
    """Explain a stronger home team's prediction."""

    explanation = explain_elo_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1550.0,
        away_rating=1450.0,
        applied_home_advantage=50.0,
    )

    assert explanation.favorite == "NE"
    assert explanation.underdog == "NYJ"
    assert explanation.raw_home_rating_edge == 100.0
    assert explanation.adjusted_home_rating_edge == 150.0
    assert explanation.matchup_label == "strong_edge"
    assert explanation.favorite_win_probability > 0.5
    assert explanation.home_field_probability_lift > 0.0


def test_explain_away_favorite() -> None:
    """Explain an away team overcoming home advantage."""

    explanation = explain_elo_prediction(
        home_team="NYJ",
        away_team="NE",
        home_rating=1400.0,
        away_rating=1600.0,
        applied_home_advantage=50.0,
    )

    assert explanation.favorite == "NE"
    assert explanation.underdog == "NYJ"
    assert explanation.home_win_probability < 0.5
    assert explanation.away_win_probability > 0.5
    assert (
        explanation.adjusted_home_rating_edge
        == -150.0
    )


def test_neutral_site_has_no_home_field_lift() -> None:
    """Show zero venue contribution on neutral ground."""

    explanation = explain_elo_prediction(
        home_team="BUF",
        away_team="KC",
        home_rating=1550.0,
        away_rating=1500.0,
        applied_home_advantage=0.0,
    )

    assert (
        explanation.home_field_probability_lift
        == pytest.approx(0.0)
    )
    assert (
        explanation.home_field_log_odds_contribution
        == pytest.approx(0.0)
    )


def test_log_odds_components_reconstruct_probability() -> None:
    """Reconstruct the prediction from technical components."""

    explanation = explain_elo_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1550.0,
        away_rating=1450.0,
        applied_home_advantage=50.0,
    )

    reconstructed_probability = (
        1.0
        / (
            1.0
            + exp(
                -explanation.total_home_log_odds
            )
        )
    )

    assert reconstructed_probability == (
        pytest.approx(
            explanation.home_win_probability
        )
    )

    assert explanation.total_home_log_odds == (
        pytest.approx(
            explanation.team_strength_log_odds_contribution
            + explanation.home_field_log_odds_contribution
        )
    )


def test_explanation_probabilities_sum_to_one() -> None:
    """Return complementary home and away probabilities."""

    explanation = explain_elo_prediction(
        home_team="NE",
        away_team="NYJ",
        home_rating=1500.0,
        away_rating=1500.0,
        applied_home_advantage=50.0,
    )

    assert (
        explanation.home_win_probability
        + explanation.away_win_probability
    ) == pytest.approx(1.0)


def test_explanation_rejects_same_team() -> None:
    """Reject an invalid self-matchup."""

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        explain_elo_prediction(
            home_team="NE",
            away_team="NE",
            home_rating=1500.0,
            away_rating=1500.0,
            applied_home_advantage=50.0,
        )