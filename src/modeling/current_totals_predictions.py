"""
NFL Analytics Platform
Current Totals Predictions

Purpose:
    Create production totals features, train the frozen
    primary and fallback models, and route upcoming games.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from datetime import datetime

import pandas as pd

from src.modeling.production_totals_component import (
    OUTPUT_COLUMNS as COMPONENT_OUTPUT_COLUMNS,
    score_current_totals_predictions,
    train_production_totals_models,
)


CURRENT_TOTALS_PREDICTION_COLUMNS = (
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
    "prediction_mode",
    "prediction_mode_reason",
    "ridge_alpha",
    "primary_training_game_count",
    "fallback_training_game_count",
    "home_prior_season_games",
    "away_prior_season_games",
    "both_short_windows_complete",
    "both_listed_qb_ratings_available",
    "has_complete_primary_features",
    "offensive_epa_sum_last_4",
    "defensive_epa_allowed_sum_last_4",
    "listed_qb_rating_sum",
    "elo_rating_sum",
    "is_indoor",
    "has_game_weather",
    "cold_degrees_below_50",
    "heat_degrees_above_80",
    "wind_mph_above_10",
    "league_average_total_last_64",
    "predicted_total_points",
    "prediction_generated_at",
)

REQUIRED_UPCOMING_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "home_prior_season_games",
    "away_prior_season_games",
    "home_offensive_epa_per_play_last_4",
    "away_offensive_epa_per_play_last_4",
    "home_defensive_epa_allowed_per_play_last_4",
    "away_defensive_epa_allowed_per_play_last_4",
    "home_listed_qb_rating",
    "away_listed_qb_rating",
    "home_elo_rating",
    "away_elo_rating",
    "is_indoor",
    "has_game_weather",
    "cold_degrees_below_50",
    "heat_degrees_above_80",
    "wind_mph_above_10",
    "league_average_total_last_64",
}


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate required DataFrame columns."""

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def create_current_totals_features(
    upcoming_games: pd.DataFrame,
) -> pd.DataFrame:
    """Create primary and fallback totals features."""

    validate_required_columns(
        data=upcoming_games,
        required_columns=REQUIRED_UPCOMING_COLUMNS,
        data_name="Upcoming totals games",
    )

    if upcoming_games["game_id"].duplicated().any():
        raise ValueError(
            "Upcoming totals games contain duplicate "
            "game identifiers."
        )

    features = upcoming_games.copy()

    features[
        "both_short_windows_complete"
    ] = (
        features["home_prior_season_games"].ge(4)
        & features["away_prior_season_games"].ge(4)
    )

    features[
        "offensive_epa_sum_last_4"
    ] = (
        features[
            "home_offensive_epa_per_play_last_4"
        ]
        + features[
            "away_offensive_epa_per_play_last_4"
        ]
    ).where(
        features["both_short_windows_complete"]
    )

    features[
        "defensive_epa_allowed_sum_last_4"
    ] = (
        features[
            (
                "home_defensive_epa_allowed_"
                "per_play_last_4"
            )
        ]
        + features[
            (
                "away_defensive_epa_allowed_"
                "per_play_last_4"
            )
        ]
    ).where(
        features["both_short_windows_complete"]
    )

    both_qb_available = (
        features["home_listed_qb_rating"].notna()
        & features["away_listed_qb_rating"].notna()
    )

    features[
        "both_listed_qb_ratings_available"
    ] = both_qb_available

    features[
        "listed_qb_rating_sum"
    ] = (
        features["home_listed_qb_rating"]
        + features["away_listed_qb_rating"]
    ).where(
        both_qb_available
    )

    if features[
        [
            "home_elo_rating",
            "away_elo_rating",
        ]
    ].isna().any(axis=None):
        missing_game_ids = ", ".join(
            features.loc[
                features[
                    [
                        "home_elo_rating",
                        "away_elo_rating",
                    ]
                ].isna().any(axis=1),
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Upcoming totals games are missing current "
            f"Elo ratings: {missing_game_ids}"
        )

    features["elo_rating_sum"] = (
        features["home_elo_rating"]
        + features["away_elo_rating"]
    )

    return features


def create_current_totals_prediction_frame(
    upcoming_games: pd.DataFrame,
    historical_data: pd.DataFrame,
    prediction_generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Train, route and create current totals output."""

    current_features = (
        create_current_totals_features(
            upcoming_games=upcoming_games
        )
    )

    trained_models = (
        train_production_totals_models(
            historical_data=historical_data
        )
    )

    component_predictions = (
        score_current_totals_predictions(
            current_features=current_features,
            trained_models=trained_models,
        )
    )

    component_columns = [
        "game_id",
        *[
            column_name
            for column_name
            in COMPONENT_OUTPUT_COLUMNS
            if (
                column_name != "game_id"
                and column_name
                not in current_features.columns
            )
        ],
    ]

    predictions = current_features.merge(
        component_predictions.loc[
            :,
            component_columns,
        ],
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    generated_at = (
        datetime.now()
        if prediction_generated_at is None
        else prediction_generated_at
    )

    predictions[
        "prediction_generated_at"
    ] = generated_at

    missing_output_columns = sorted(
        set(CURRENT_TOTALS_PREDICTION_COLUMNS)
        - set(predictions.columns)
    )

    if missing_output_columns:
        raise RuntimeError(
            "Current totals prediction output is "
            "missing columns: "
            + ", ".join(missing_output_columns)
        )

    return predictions.loc[
        :,
        CURRENT_TOTALS_PREDICTION_COLUMNS,
    ]