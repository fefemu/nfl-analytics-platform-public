"""
NFL Analytics Platform
Dynamic Elo Season Simulation

Purpose:
    Simulate one NFL regular season while updating team Elo
    ratings after every simulated game.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.modeling.production_model import (
    PRODUCTION_MODEL,
)
from src.models.elo import (
    calculate_expected_probability,
    update_ratings,
)


REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "home_rating_pregame",
    "away_rating_pregame",
}


@dataclass
class SeasonSimulationResult:
    """Store one simulated season's outputs."""

    game_results: pd.DataFrame
    team_records: pd.DataFrame
    final_ratings: dict[str, float]


def extract_initial_ratings(
    schedule: pd.DataFrame,
) -> dict[str, float]:
    """Extract one consistent initial rating per team."""

    missing_columns = sorted(
        REQUIRED_SCHEDULE_COLUMNS
        - set(schedule.columns)
    )

    if missing_columns:
        raise ValueError(
            "Simulation schedule is missing columns: "
            + ", ".join(missing_columns)
        )

    ratings_by_team: dict[
        str,
        list[float],
    ] = {}

    for game in schedule.itertuples(index=False):
        ratings_by_team.setdefault(
            str(game.home_team),
            [],
        ).append(
            float(game.home_rating_pregame)
        )

        ratings_by_team.setdefault(
            str(game.away_team),
            [],
        ).append(
            float(game.away_rating_pregame)
        )

    initial_ratings: dict[str, float] = {}

    for team, rating_values in ratings_by_team.items():
        reference_rating = rating_values[0]

        if any(
            not np.isclose(
                rating,
                reference_rating,
            )
            for rating in rating_values[1:]
        ):
            raise ValueError(
                "Simulation schedule contains "
                f"inconsistent initial ratings for {team}."
            )

        initial_ratings[team] = reference_rating

    return initial_ratings


def simulate_regular_season_once(
    schedule: pd.DataFrame,
    random_generator: np.random.Generator,
) -> SeasonSimulationResult:
    """Run one dynamic Elo regular-season simulation."""

    if schedule.empty:
        raise ValueError(
            "Simulation schedule must not be empty."
        )

    initial_ratings = extract_initial_ratings(
        schedule
    )

    current_ratings = initial_ratings.copy()

    team_records = {
        team: {
            "wins": 0,
            "losses": 0,
        }
        for team in current_ratings
    }

    ordered_schedule = schedule.sort_values(
        by=[
            "week",
            "gameday",
            "gametime",
            "game_id",
        ]
    ).reset_index(drop=True)

    game_rows: list[dict[str, object]] = []

    for game in ordered_schedule.itertuples(
        index=False
    ):
        home_team = str(game.home_team)
        away_team = str(game.away_team)

        home_rating_pre = current_ratings[
            home_team
        ]
        away_rating_pre = current_ratings[
            away_team
        ]

        applied_home_advantage = (
            0.0
            if bool(game.is_neutral)
            else PRODUCTION_MODEL.home_advantage
        )

        home_win_probability = (
            calculate_expected_probability(
                team_rating=home_rating_pre,
                opponent_rating=away_rating_pre,
                rating_advantage=(
                    applied_home_advantage
                ),
            )
        )

        simulated_home_win = bool(
            random_generator.random()
            < home_win_probability
        )

        actual_home_score = (
            1.0 if simulated_home_win else 0.0
        )

        (
            home_rating_post,
            away_rating_post,
        ) = update_ratings(
            team_rating=home_rating_pre,
            opponent_rating=away_rating_pre,
            actual_score=actual_home_score,
            expected_probability=(
                home_win_probability
            ),
            k_factor=PRODUCTION_MODEL.k_factor,
        )

        current_ratings[home_team] = (
            home_rating_post
        )
        current_ratings[away_team] = (
            away_rating_post
        )

        if simulated_home_win:
            winner = home_team
            loser = away_team
        else:
            winner = away_team
            loser = home_team

        team_records[winner]["wins"] += 1
        team_records[loser]["losses"] += 1

        game_rows.append(
            {
                "game_id": str(game.game_id),
                "season": int(game.season),
                "week": int(game.week),
                "home_team": home_team,
                "away_team": away_team,
                "home_rating_pre": (
                    home_rating_pre
                ),
                "away_rating_pre": (
                    away_rating_pre
                ),
                "applied_home_advantage": (
                    applied_home_advantage
                ),
                "home_win_probability": (
                    home_win_probability
                ),
                "simulated_home_win": (
                    simulated_home_win
                ),
                "winner": winner,
                "loser": loser,
                "home_rating_post": (
                    home_rating_post
                ),
                "away_rating_post": (
                    away_rating_post
                ),
            }
        )

    record_rows = [
        {
            "team": team,
            "wins": record["wins"],
            "losses": record["losses"],
            "games": (
                record["wins"]
                + record["losses"]
            ),
            "final_elo_rating": (
                current_ratings[team]
            ),
        }
        for team, record in team_records.items()
    ]

    game_results = pd.DataFrame(game_rows)

    records = pd.DataFrame(
        record_rows
    ).sort_values(
        by=[
            "wins",
            "final_elo_rating",
            "team",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return SeasonSimulationResult(
        game_results=game_results,
        team_records=records,
        final_ratings=current_ratings,
    )