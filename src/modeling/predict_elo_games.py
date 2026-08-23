"""
NFL Analytics Platform
Production Elo Game Prediction

Purpose:
    Create reproducible pregame Elo probabilities from
    current ratings for prediction and simulation workflows.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

from src.modeling.production_model import (
    PRODUCTION_MODEL,
)
from src.models.elo import (
    calculate_expected_probability,
    regress_rating_to_mean,
)


@dataclass(frozen=True)
class EloPregamePrediction:
    """Store one production Elo game prediction."""

    home_team: str
    away_team: str
    season: int
    is_neutral: bool
    home_rating_current: float
    away_rating_current: float
    home_rating_pregame: float
    away_rating_pregame: float
    applied_home_advantage: float
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str


def regress_rating_across_seasons(
    rating: float,
    last_completed_season: int,
    target_season: int,
) -> float:
    """Regress a rating once for every crossed offseason."""

    if target_season < last_completed_season:
        raise ValueError(
            "Target season cannot precede the rating's "
            "last completed season."
        )

    regressed_rating = float(rating)

    season_gap = (
        target_season - last_completed_season
    )

    for _ in range(season_gap):
        regressed_rating = regress_rating_to_mean(
            rating=regressed_rating,
            retention=(
                PRODUCTION_MODEL.season_retention
            ),
        )

    return regressed_rating


def create_elo_pregame_prediction(
    home_team: str,
    away_team: str,
    home_rating: float,
    away_rating: float,
    home_last_completed_season: int,
    away_last_completed_season: int,
    target_season: int,
    is_neutral: bool = False,
) -> EloPregamePrediction:
    """Create one production Elo pregame prediction."""

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

    home_rating_pregame = (
        regress_rating_across_seasons(
            rating=home_rating,
            last_completed_season=(
                home_last_completed_season
            ),
            target_season=target_season,
        )
    )

    away_rating_pregame = (
        regress_rating_across_seasons(
            rating=away_rating,
            last_completed_season=(
                away_last_completed_season
            ),
            target_season=target_season,
        )
    )

    applied_home_advantage = (
        0.0
        if is_neutral
        else PRODUCTION_MODEL.home_advantage
    )

    home_win_probability = (
        calculate_expected_probability(
            team_rating=home_rating_pregame,
            opponent_rating=away_rating_pregame,
            rating_advantage=(
                applied_home_advantage
            ),
        )
    )

    away_win_probability = (
        1.0 - home_win_probability
    )

    predicted_winner = (
        home_team
        if home_win_probability
        >= PRODUCTION_MODEL.classification_threshold
        else away_team
    )

    return EloPregamePrediction(
        home_team=home_team,
        away_team=away_team,
        season=target_season,
        is_neutral=is_neutral,
        home_rating_current=float(home_rating),
        away_rating_current=float(away_rating),
        home_rating_pregame=home_rating_pregame,
        away_rating_pregame=away_rating_pregame,
        applied_home_advantage=(
            applied_home_advantage
        ),
        home_win_probability=(
            home_win_probability
        ),
        away_win_probability=(
            away_win_probability
        ),
        predicted_winner=predicted_winner,
    )