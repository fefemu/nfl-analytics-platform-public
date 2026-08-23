"""
NFL Analytics Platform
Current Game Prediction Records

Purpose:
    Transform upcoming schedule and current Elo rating data
    into versioned production prediction records.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from datetime import datetime, timezone

import pandas as pd

from src.modeling.predict_elo_games import (
    create_elo_pregame_prediction,
)
from src.modeling.production_model import (
    PRODUCTION_MODEL,
)


REQUIRED_INPUT_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "location",
    "home_elo_rating",
    "away_elo_rating",
    "home_rating_season",
    "away_rating_season",
    "home_rating_as_of",
    "away_rating_as_of",
}

PREDICTION_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "model_name",
    "model_version",
    "home_rating_current",
    "away_rating_current",
    "home_rating_pregame",
    "away_rating_pregame",
    "applied_home_advantage",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "home_rating_as_of",
    "away_rating_as_of",
    "prediction_generated_at",
)


def is_neutral_location(
    location: object,
) -> bool:
    """Identify a neutral-site schedule location."""

    if location is None or pd.isna(location):
        return False

    return str(location).strip().upper() == "NEUTRAL"


def create_current_prediction_frame(
    upcoming_games: pd.DataFrame,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Create versioned production predictions."""

    missing_columns = sorted(
        REQUIRED_INPUT_COLUMNS
        - set(upcoming_games.columns)
    )

    if missing_columns:
        raise ValueError(
            "Upcoming game data is missing columns: "
            + ", ".join(missing_columns)
        )

    if upcoming_games["game_id"].duplicated().any():
        raise ValueError(
            "Upcoming game data contains duplicate "
            "game identifiers."
        )

    if generated_at is None:
        generated_at = datetime.now(
            timezone.utc
        ).replace(tzinfo=None)

    prediction_rows: list[dict[str, object]] = []

    for game in upcoming_games.itertuples(
        index=False
    ):
        prediction = create_elo_pregame_prediction(
            home_team=str(game.home_team),
            away_team=str(game.away_team),
            home_rating=float(
                game.home_elo_rating
            ),
            away_rating=float(
                game.away_elo_rating
            ),
            home_last_completed_season=int(
                game.home_rating_season
            ),
            away_last_completed_season=int(
                game.away_rating_season
            ),
            target_season=int(game.season),
            is_neutral=is_neutral_location(
                game.location
            ),
        )

        prediction_rows.append(
            {
                "game_id": str(game.game_id),
                "season": int(game.season),
                "game_type": str(
                    game.game_type
                ),
                "week": int(game.week),
                "gameday": game.gameday,
                "gametime": game.gametime,
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "is_neutral": prediction.is_neutral,
                "model_name": (
                    PRODUCTION_MODEL.model_name
                ),
                "model_version": (
                    PRODUCTION_MODEL.model_version
                ),
                "home_rating_current": (
                    prediction.home_rating_current
                ),
                "away_rating_current": (
                    prediction.away_rating_current
                ),
                "home_rating_pregame": (
                    prediction.home_rating_pregame
                ),
                "away_rating_pregame": (
                    prediction.away_rating_pregame
                ),
                "applied_home_advantage": (
                    prediction.applied_home_advantage
                ),
                "home_win_probability": (
                    prediction.home_win_probability
                ),
                "away_win_probability": (
                    prediction.away_win_probability
                ),
                "predicted_winner": (
                    prediction.predicted_winner
                ),
                "home_rating_as_of": (
                    game.home_rating_as_of
                ),
                "away_rating_as_of": (
                    game.away_rating_as_of
                ),
                "prediction_generated_at": (
                    generated_at
                ),
            }
        )

    return pd.DataFrame(
        prediction_rows,
        columns=PREDICTION_COLUMNS,
    )