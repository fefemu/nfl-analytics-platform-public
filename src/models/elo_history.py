"""
NFL Analytics Platform
Historical Elo Processor

Purpose:
    Process completed NFL games in chronological order
    and maintain team Elo ratings across seasons.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass
from datetime import date, time

from src.config.nfl_team_mappings import (
    normalize_franchise_code,
)
from src.models.elo import (
    DEFAULT_K_FACTOR,
    DEFAULT_RATING,
    DEFAULT_SEASON_RETENTION,
    process_game,
    regress_rating_to_mean,
)


@dataclass(frozen=True)
class HistoricalGame:
    """Store the model inputs for one completed historical game."""

    game_id: str
    season: int
    game_type: str
    week: int
    gameday: date
    gametime: time | None
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    is_neutral: bool

@dataclass(frozen=True)
class EloHistoryRecord:
    """Store the complete Elo calculation for one historical game."""

    game_id: str
    season: int
    game_type: str
    week: int
    gameday: date
    home_team: str
    away_team: str
    home_franchise: str
    away_franchise: str
    is_neutral: bool
    home_advantage: float
    home_rating_pre: float
    away_rating_pre: float
    home_win_probability: float
    away_win_probability: float
    actual_home_score: float
    home_rating_post: float
    away_rating_post: float
    home_rating_change: float


def sort_games_chronologically(
    games: list[HistoricalGame],
) -> list[HistoricalGame]:
    """Return historical games in deterministic chronological order."""

    return sorted(
        games,
        key=lambda game: (
            game.gameday,
            game.gametime or time.min,
            game.game_id,
        ),
    )


def process_elo_history(
    games: list[HistoricalGame],
    initial_rating: float = DEFAULT_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
    home_advantage: float = 0.0,
    season_retention: float = DEFAULT_SEASON_RETENTION,
) -> tuple[list[EloHistoryRecord], dict[str, float]]:
    """Process historical games and return records and final ratings."""

    sorted_games = sort_games_chronologically(games)

    ratings: dict[str, float] = {}
    history_records: list[EloHistoryRecord] = []
    current_season: int | None = None

    for game in sorted_games:
        if current_season is None:
            current_season = game.season
        elif game.season != current_season:
            ratings = {
                team: regress_rating_to_mean(
                    rating=rating,
                    mean_rating=initial_rating,
                    retention=season_retention,
                )
                for team, rating in ratings.items()
            }
            current_season = game.season

        home_franchise = normalize_franchise_code(
            game.home_team
        )
        away_franchise = normalize_franchise_code(
            game.away_team
        )

        home_rating_pre = ratings.get(
            home_franchise,
            initial_rating,
        )
        away_rating_pre = ratings.get(
            away_franchise,
            initial_rating,
        )

        applied_home_advantage = (
            0.0
            if game.is_neutral
            else home_advantage
        )

        game_result = process_game(
            home_rating=home_rating_pre,
            away_rating=away_rating_pre,
            home_score=game.home_score,
            away_score=game.away_score,
            home_advantage=applied_home_advantage,
            k_factor=k_factor,
        )

        ratings[home_franchise] = (
            game_result.home_rating_post
        )
        ratings[away_franchise] = (
            game_result.away_rating_post
        )

        history_records.append(
            EloHistoryRecord(
                game_id=game.game_id,
                season=game.season,
                game_type=game.game_type,
                week=game.week,
                gameday=game.gameday,
                home_team=game.home_team,
                away_team=game.away_team,
                home_franchise=home_franchise,
                away_franchise=away_franchise,
                is_neutral=game.is_neutral,
                home_advantage=applied_home_advantage,
                home_rating_pre=(
                    game_result.home_rating_pre
                ),
                away_rating_pre=(
                    game_result.away_rating_pre
                ),
                home_win_probability=(
                    game_result.home_win_probability
                ),
                away_win_probability=(
                    game_result.away_win_probability
                ),
                actual_home_score=(
                    game_result.actual_home_score
                ),
                home_rating_post=(
                    game_result.home_rating_post
                ),
                away_rating_post=(
                    game_result.away_rating_post
                ),
                home_rating_change=(
                    game_result.home_rating_change
                ),
            )
        )

    return history_records, ratings