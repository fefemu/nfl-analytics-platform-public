"""
NFL Analytics Platform
Current Game Prediction Explanations

Purpose:
    Transform current Elo game predictions into structured
    user-facing and technical explanation records.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import pandas as pd

from src.modeling.explain_elo_prediction import (
    explain_elo_prediction,
)


REQUIRED_PREDICTION_COLUMNS = {
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
    "home_rating_pregame",
    "away_rating_pregame",
    "applied_home_advantage",
    "home_win_probability",
    "away_win_probability",
    "prediction_generated_at",
}

EXPLANATION_COLUMNS = (
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
    "favorite",
    "underdog",
    "favorite_win_probability",
    "home_win_probability",
    "away_win_probability",
    "neutral_home_win_probability",
    "home_field_probability_lift",
    "home_rating",
    "away_rating",
    "raw_home_rating_edge",
    "applied_home_advantage",
    "adjusted_home_rating_edge",
    "team_strength_log_odds_contribution",
    "home_field_log_odds_contribution",
    "total_home_log_odds",
    "matchup_label",
    "prediction_generated_at",
)


def create_prediction_explanation_frame(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create explanations for current predictions."""

    missing_columns = sorted(
        REQUIRED_PREDICTION_COLUMNS
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current predictions are missing explanation "
            "columns: "
            + ", ".join(missing_columns)
        )

    if predictions["game_id"].duplicated().any():
        raise ValueError(
            "Current predictions contain duplicate "
            "game identifiers."
        )

    explanation_rows: list[
        dict[str, object]
    ] = []

    for prediction in predictions.itertuples(
        index=False
    ):
        explanation = explain_elo_prediction(
            home_team=str(
                prediction.home_team
            ),
            away_team=str(
                prediction.away_team
            ),
            home_rating=float(
                prediction.home_rating_pregame
            ),
            away_rating=float(
                prediction.away_rating_pregame
            ),
            applied_home_advantage=float(
                prediction.applied_home_advantage
            ),
        )

        if not (
            abs(
                explanation.home_win_probability
                - float(
                    prediction.home_win_probability
                )
            )
            <= 0.000001
        ):
            raise RuntimeError(
                "Explanation probability does not match "
                f"prediction for {prediction.game_id}."
            )

        explanation_rows.append(
            {
                "game_id": str(
                    prediction.game_id
                ),
                "season": int(
                    prediction.season
                ),
                "game_type": str(
                    prediction.game_type
                ),
                "week": int(prediction.week),
                "gameday": prediction.gameday,
                "gametime": prediction.gametime,
                "home_team": (
                    explanation.home_team
                ),
                "away_team": (
                    explanation.away_team
                ),
                "is_neutral": bool(
                    prediction.is_neutral
                ),
                "model_name": str(
                    prediction.model_name
                ),
                "model_version": str(
                    prediction.model_version
                ),
                "favorite": explanation.favorite,
                "underdog": (
                    explanation.underdog
                ),
                "favorite_win_probability": (
                    explanation.favorite_win_probability
                ),
                "home_win_probability": (
                    explanation.home_win_probability
                ),
                "away_win_probability": (
                    explanation.away_win_probability
                ),
                "neutral_home_win_probability": (
                    explanation.neutral_home_win_probability
                ),
                "home_field_probability_lift": (
                    explanation.home_field_probability_lift
                ),
                "home_rating": (
                    explanation.home_rating
                ),
                "away_rating": (
                    explanation.away_rating
                ),
                "raw_home_rating_edge": (
                    explanation.raw_home_rating_edge
                ),
                "applied_home_advantage": (
                    explanation.applied_home_advantage
                ),
                "adjusted_home_rating_edge": (
                    explanation.adjusted_home_rating_edge
                ),
                "team_strength_log_odds_contribution": (
                    explanation.team_strength_log_odds_contribution
                ),
                "home_field_log_odds_contribution": (
                    explanation.home_field_log_odds_contribution
                ),
                "total_home_log_odds": (
                    explanation.total_home_log_odds
                ),
                "matchup_label": (
                    explanation.matchup_label
                ),
                "prediction_generated_at": (
                    prediction.prediction_generated_at
                ),
            }
        )

    return pd.DataFrame(
        explanation_rows,
        columns=EXPLANATION_COLUMNS,
    )