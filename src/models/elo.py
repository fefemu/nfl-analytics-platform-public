"""
NFL Analytics Platform
Elo Rating Model

Purpose:
    Provide the core mathematical functions
    for the NFL Elo rating model.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 45.0
DEFAULT_SEASON_RETENTION = 0.60
ELO_SCALE = 400.0

@dataclass(frozen=True)
class EloGameResult:
    """Store the complete Elo calculation for one game."""

    home_rating_pre: float
    away_rating_pre: float
    home_win_probability: float
    away_win_probability: float
    actual_home_score: float
    home_rating_post: float
    away_rating_post: float
    home_rating_change: float

def calculate_expected_probability(
    team_rating: float,
    opponent_rating: float,
    rating_advantage: float = 0.0,
) -> float:
    """Calculate the expected win probability for a team."""

    adjusted_rating = team_rating + rating_advantage
    rating_difference = opponent_rating - adjusted_rating

    return 1.0 / (
        1.0 + 10.0 ** (rating_difference / ELO_SCALE)
    )


def calculate_rating_change(
    actual_score: float,
    expected_probability: float,
    k_factor: float = DEFAULT_K_FACTOR,
) -> float:
    """Calculate the Elo rating change after a game."""

    return k_factor * (
        actual_score - expected_probability
    )


def update_ratings(
    team_rating: float,
    opponent_rating: float,
    actual_score: float,
    expected_probability: float,
    k_factor: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Update both team ratings after a game."""

    rating_change = calculate_rating_change(
        actual_score=actual_score,
        expected_probability=expected_probability,
        k_factor=k_factor,
    )

    updated_team_rating = team_rating + rating_change
    updated_opponent_rating = opponent_rating - rating_change

    return updated_team_rating, updated_opponent_rating


def regress_rating_to_mean(
    rating: float,
    mean_rating: float = DEFAULT_RATING,
    retention: float = DEFAULT_SEASON_RETENTION,
) -> float:
    """Regress a team rating toward the league mean between seasons."""

    if not 0.0 <= retention <= 1.0:
        raise ValueError(
            "Season retention must be between 0 and 1."
        )

    return mean_rating + (
        rating - mean_rating
    ) * retention


def calculate_actual_score(
    team_score: int,
    opponent_score: int,
) -> float:
    """Convert a final game score into an Elo result value."""

    if team_score > opponent_score:
        return 1.0

    if team_score < opponent_score:
        return 0.0

    return 0.5


def process_game(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    home_advantage: float = 0.0,
    k_factor: float = DEFAULT_K_FACTOR,
) -> EloGameResult:
    """Calculate the complete Elo result for one completed game."""

    home_win_probability = calculate_expected_probability(
        team_rating=home_rating,
        opponent_rating=away_rating,
        rating_advantage=home_advantage,
    )
    away_win_probability = 1.0 - home_win_probability

    actual_home_score = calculate_actual_score(
        team_score=home_score,
        opponent_score=away_score,
    )

    home_rating_post, away_rating_post = update_ratings(
        team_rating=home_rating,
        opponent_rating=away_rating,
        actual_score=actual_home_score,
        expected_probability=home_win_probability,
        k_factor=k_factor,
    )

    return EloGameResult(
        home_rating_pre=home_rating,
        away_rating_pre=away_rating,
        home_win_probability=home_win_probability,
        away_win_probability=away_win_probability,
        actual_home_score=actual_home_score,
        home_rating_post=home_rating_post,
        away_rating_post=away_rating_post,
        home_rating_change=home_rating_post - home_rating,
    )