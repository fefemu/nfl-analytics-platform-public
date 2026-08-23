"""
NFL Analytics Platform
Production Elo Prediction Explanation

Purpose:
    Decompose one production Elo prediction into user-facing
    and technical explanation components.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass
from math import log

from src.models.elo import (
    ELO_SCALE,
    calculate_expected_probability,
)


@dataclass(frozen=True)
class EloPredictionExplanation:
    """Store a two-level Elo prediction explanation."""

    home_team: str
    away_team: str
    favorite: str
    underdog: str
    favorite_win_probability: float
    home_win_probability: float
    away_win_probability: float
    neutral_home_win_probability: float
    home_field_probability_lift: float
    home_rating: float
    away_rating: float
    raw_home_rating_edge: float
    applied_home_advantage: float
    adjusted_home_rating_edge: float
    team_strength_log_odds_contribution: float
    home_field_log_odds_contribution: float
    total_home_log_odds: float
    matchup_label: str


def classify_matchup(
    adjusted_rating_edge: float,
) -> str:
    """Classify matchup strength from adjusted Elo edge."""

    absolute_edge = abs(adjusted_rating_edge)

    if absolute_edge < 25.0:
        return "toss_up"

    if absolute_edge < 75.0:
        return "slight_edge"

    if absolute_edge < 150.0:
        return "clear_edge"

    return "strong_edge"


def explain_elo_prediction(
    home_team: str,
    away_team: str,
    home_rating: float,
    away_rating: float,
    applied_home_advantage: float,
) -> EloPredictionExplanation:
    """Create user-facing and technical Elo explanations."""

    if not home_team.strip():
        raise ValueError(
            "Home team must not be empty."
        )

    if not away_team.strip():
        raise ValueError(
            "Away team must not be empty."
        )

    if home_team == away_team:
        raise ValueError(
            "Home and away teams must be different."
        )

    raw_home_rating_edge = (
        float(home_rating) - float(away_rating)
    )

    adjusted_home_rating_edge = (
        raw_home_rating_edge
        + float(applied_home_advantage)
    )

    neutral_home_win_probability = (
        calculate_expected_probability(
            team_rating=float(home_rating),
            opponent_rating=float(away_rating),
        )
    )

    home_win_probability = (
        calculate_expected_probability(
            team_rating=float(home_rating),
            opponent_rating=float(away_rating),
            rating_advantage=float(
                applied_home_advantage
            ),
        )
    )

    away_win_probability = (
        1.0 - home_win_probability
    )

    home_field_probability_lift = (
        home_win_probability
        - neutral_home_win_probability
    )

    log_odds_multiplier = (
        log(10.0) / ELO_SCALE
    )

    team_strength_log_odds_contribution = (
        log_odds_multiplier
        * raw_home_rating_edge
    )

    home_field_log_odds_contribution = (
        log_odds_multiplier
        * float(applied_home_advantage)
    )

    total_home_log_odds = (
        team_strength_log_odds_contribution
        + home_field_log_odds_contribution
    )

    if home_win_probability >= 0.5:
        favorite = home_team
        underdog = away_team
        favorite_win_probability = (
            home_win_probability
        )
    else:
        favorite = away_team
        underdog = home_team
        favorite_win_probability = (
            away_win_probability
        )

    return EloPredictionExplanation(
        home_team=home_team,
        away_team=away_team,
        favorite=favorite,
        underdog=underdog,
        favorite_win_probability=(
            favorite_win_probability
        ),
        home_win_probability=(
            home_win_probability
        ),
        away_win_probability=(
            away_win_probability
        ),
        neutral_home_win_probability=(
            neutral_home_win_probability
        ),
        home_field_probability_lift=(
            home_field_probability_lift
        ),
        home_rating=float(home_rating),
        away_rating=float(away_rating),
        raw_home_rating_edge=(
            raw_home_rating_edge
        ),
        applied_home_advantage=float(
            applied_home_advantage
        ),
        adjusted_home_rating_edge=(
            adjusted_home_rating_edge
        ),
        team_strength_log_odds_contribution=(
            team_strength_log_odds_contribution
        ),
        home_field_log_odds_contribution=(
            home_field_log_odds_contribution
        ),
        total_home_log_odds=(
            total_home_log_odds
        ),
        matchup_label=classify_matchup(
            adjusted_home_rating_edge
        ),
    )