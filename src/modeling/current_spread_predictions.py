"""
NFL Analytics Platform
Current Spread Predictions

Purpose:
    Create auditable current spread predictions from
    upcoming games, current Elo outputs and historical
    model-training data.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from datetime import datetime

import pandas as pd

from src.modeling.production_spread_component import (
    OUTPUT_COLUMNS as COMPONENT_OUTPUT_COLUMNS,
    score_current_spread_predictions,
    train_production_spread_models,
)


CURRENT_SPREAD_PREDICTION_COLUMNS = (
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
    "external_nfelo_rating_difference",
    "listed_qb_rating_difference",
    "external_nfelo_qb_adjustment_difference",
    "both_listed_qb_ratings_available",
    "predicted_home_margin",
    "predicted_away_margin",
    "predicted_winner",
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
    "location",
    "home_listed_qb_rating",
    "away_listed_qb_rating",
    "external_nfelo_rating_difference",
    "external_nfelo_qb_adjustment_difference",
}

REQUIRED_ELO_COLUMNS = {
    "game_id",
    "home_rating_pregame",
    "away_rating_pregame",
    "is_neutral",
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


def create_current_spread_features(
    upcoming_games: pd.DataFrame,
    elo_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create current Elo and listed-QB differences."""

    validate_required_columns(
        data=upcoming_games,
        required_columns=REQUIRED_UPCOMING_COLUMNS,
        data_name="Upcoming spread games",
    )

    validate_required_columns(
        data=elo_predictions,
        required_columns=REQUIRED_ELO_COLUMNS,
        data_name="Current Elo predictions",
    )

    if upcoming_games["game_id"].duplicated().any():
        raise ValueError(
            "Upcoming spread games contain duplicate "
            "game identifiers."
        )

    if elo_predictions["game_id"].duplicated().any():
        raise ValueError(
            "Current Elo predictions contain duplicate "
            "game identifiers."
        )

    elo_source = elo_predictions.loc[
        :,
        [
            "game_id",
            "home_rating_pregame",
            "away_rating_pregame",
            "is_neutral",
        ],
    ]

    features = upcoming_games.merge(
        elo_source,
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    missing_elo_mask = features[
        [
            "home_rating_pregame",
            "away_rating_pregame",
            "is_neutral",
        ]
    ].isna().any(axis=1)

    if missing_elo_mask.any():
        missing_game_ids = ", ".join(
            features.loc[
                missing_elo_mask,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Upcoming spread games are missing current "
            f"Elo predictions: {missing_game_ids}"
        )

    both_qb_available = (
        features["home_listed_qb_rating"].notna()
        & features["away_listed_qb_rating"].notna()
    )

    features[
        "both_listed_qb_ratings_available"
    ] = both_qb_available

    features[
        "listed_qb_rating_difference"
    ] = (
        features["home_listed_qb_rating"]
        - features["away_listed_qb_rating"]
    ).where(
        both_qb_available
    )

    return features


def create_current_spread_prediction_frame(
    upcoming_games: pd.DataFrame,
    elo_predictions: pd.DataFrame,
    historical_data: pd.DataFrame,
    prediction_generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Train, route and create current spread output."""

    current_features = (
        create_current_spread_features(
            upcoming_games=upcoming_games,
            elo_predictions=elo_predictions,
        )
    )

    trained_models = (
        train_production_spread_models(
            historical_data=historical_data
        )
    )

    component_predictions = (
        score_current_spread_predictions(
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
        "primary_training_game_count"
    ] = (
        trained_models
        .primary_training_game_count
    )

    predictions[
        "fallback_training_game_count"
    ] = (
        trained_models
        .fallback_training_game_count
    )

    predictions[
        "prediction_generated_at"
    ] = generated_at

    missing_output_columns = sorted(
        set(CURRENT_SPREAD_PREDICTION_COLUMNS)
        - set(predictions.columns)
    )

    if missing_output_columns:
        raise RuntimeError(
            "Current spread prediction output is "
            "missing columns: "
            + ", ".join(missing_output_columns)
        )

    return predictions.loc[
        :,
        CURRENT_SPREAD_PREDICTION_COLUMNS,
    ]
