"""
NFL Analytics Platform
Current Production Predictions

Purpose:
    Combine current external nfelo inputs, the selected
    primary logistic model and the external Elo-QB
    fallback into one auditable game-level prediction
    frame.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

import pandas as pd

from src.modeling.current_production_features import (
    create_current_production_feature_frame,
)
from src.modeling.production_logistic_component import (
    LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS,
    LOGISTIC_FEATURE_COVERAGE_COLUMN,
    LOGISTIC_PROBABILITY_COLUMN,
    create_long_logistic_feature_contributions,
    score_current_logistic_component,
    train_production_logistic_component,
)
from src.modeling.production_probability_fallback_component import (
    FALLBACK_FEATURE_COVERAGE_COLUMN,
    FALLBACK_PROBABILITY_COLUMN,
    score_probability_fallback_component,
    train_probability_fallback_component,
)
from src.modeling.production_probability_predictions import (
    create_production_probability_prediction,
)


PRODUCTION_AUDIT_COLUMNS = (
    "prediction_mode",
    "prediction_mode_reason",
    "published_nfelo_home_probability",
    "primary_logistic_home_win_probability",
    "fallback_logistic_home_win_probability",
    "applied_primary_logistic_weight",
    "applied_published_nfelo_weight",
    "elo_home_win_probability",
    "logistic_home_win_probability",
    "applied_logistic_weight",
    "applied_elo_weight",
    "has_complete_injury_data",
    "both_listed_qb_ratings_available",
    "has_complete_production_features",
    "has_complete_fallback_features",
    "external_nfelo_rating_difference",
    "listed_qb_rating_difference",
    "external_nfelo_qb_adjustment_difference",
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
)

CONTRIBUTION_METADATA_COLUMNS = (
    "model_name",
    "model_version",
    "prediction_generated_at",
)

CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS = (
    *LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS,
    *CONTRIBUTION_METADATA_COLUMNS,
)


def create_current_production_predictions(
    upcoming_games: pd.DataFrame,
    elo_predictions: pd.DataFrame,
    historical_data: pd.DataFrame,
    return_feature_contributions: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, pd.DataFrame]
):
    """
    Create current external primary or fallback outputs.

    The existing Elo prediction frame remains the
    transitional table spine so schedule metadata and
    existing downstream consumers are preserved.

    The probabilities and production model metadata are
    replaced by the selected external routing.

    When requested, also return the normalized primary
    logistic feature-contribution frame.
    """

    current_features = (
        create_current_production_feature_frame(
            upcoming_games=upcoming_games,
            elo_predictions=elo_predictions,
        )
    )

    primary_model = (
        train_production_logistic_component(
            historical_data=historical_data
        )
    )

    primary_scored_features = (
        score_current_logistic_component(
            current_features=current_features,
            trained_model=primary_model,
        )
    )

    fallback_model = (
        train_probability_fallback_component(
            historical_data=historical_data
        )
    )

    scored_features = (
        score_probability_fallback_component(
            current_features=(
                primary_scored_features
            ),
            trained_fallback=fallback_model,
        )
    )

    route_rows: list[dict[str, object]] = []

    for game in scored_features.itertuples(
        index=False
    ):
        has_complete_primary_features = bool(
            getattr(
                game,
                LOGISTIC_FEATURE_COVERAGE_COLUMN,
            )
        )

        primary_probability_value = getattr(
            game,
            LOGISTIC_PROBABILITY_COLUMN,
        )

        fallback_probability_value = getattr(
            game,
            FALLBACK_PROBABILITY_COLUMN,
        )

        routed_prediction = (
            create_production_probability_prediction(
                published_nfelo_home_probability=float(
                    game.published_nfelo_home_probability
                ) if pd.notna(
                    game.published_nfelo_home_probability
                ) else None,
                primary_logistic_home_win_probability=(
                    None
                    if pd.isna(
                        primary_probability_value
                    )
                    else float(
                        primary_probability_value
                    )
                ),
                fallback_logistic_home_win_probability=(
                    None
                    if pd.isna(
                        fallback_probability_value
                    )
                    else float(
                        fallback_probability_value
                    )
                ),
                has_complete_primary_features=(
                    has_complete_primary_features
                ),
            )
        )

        route_rows.append(
            {
                "game_id": str(game.game_id),
                "model_name": (
                    routed_prediction.model_name
                ),
                "model_version": (
                    routed_prediction.model_version
                ),
                "prediction_mode": (
                    routed_prediction.prediction_mode
                ),
                "prediction_mode_reason": (
                    routed_prediction
                    .prediction_mode_reason
                ),
                "home_win_probability": (
                    routed_prediction
                    .home_win_probability
                ),
                "away_win_probability": (
                    routed_prediction
                    .away_win_probability
                ),
                "published_nfelo_home_probability": (
                    routed_prediction
                    .published_nfelo_home_probability
                ),
                "primary_logistic_home_win_probability": (
                    routed_prediction
                    .primary_logistic_home_win_probability
                ),
                "fallback_logistic_home_win_probability": (
                    routed_prediction
                    .fallback_logistic_home_win_probability
                ),
                "applied_primary_logistic_weight": (
                    routed_prediction
                    .applied_primary_logistic_weight
                ),
                "applied_published_nfelo_weight": (
                    routed_prediction
                    .applied_published_nfelo_weight
                ),
                "elo_home_win_probability": (
                    routed_prediction
                    .elo_home_win_probability
                ),
                "logistic_home_win_probability": (
                    routed_prediction
                    .logistic_home_win_probability
                ),
                "applied_logistic_weight": (
                    routed_prediction
                    .applied_logistic_weight
                ),
                "applied_elo_weight": (
                    routed_prediction
                    .applied_elo_weight
                ),
            }
        )

    route_frame = pd.DataFrame(
        route_rows
    )

    if elo_predictions.empty:
        empty_predictions = (
            elo_predictions.copy()
        )

        for column_name in (
            PRODUCTION_AUDIT_COLUMNS
        ):
            empty_predictions[
                column_name
            ] = pd.Series(
                dtype="object"
            )

        empty_contributions = pd.DataFrame(
            columns=(
                CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
            )
        )

        if return_feature_contributions:
            return (
                empty_predictions,
                empty_contributions,
            )

        return empty_predictions

    prediction_spine = (
        elo_predictions.drop(
            columns=[
                "model_name",
                "model_version",
                "home_win_probability",
                "away_win_probability",
                "predicted_winner",
            ],
        )
    )

    feature_audit = scored_features.loc[
        :,
        [
            "game_id",
            "has_complete_injury_data",
            "both_listed_qb_ratings_available",
            "has_complete_production_features",
            FALLBACK_FEATURE_COVERAGE_COLUMN,
            "external_nfelo_rating_difference",
            "listed_qb_rating_difference",
            "external_nfelo_qb_adjustment_difference",
            "offense_injury_burden_difference",
            "defense_injury_burden_difference",
            "special_teams_injury_burden_difference",
        ],
    ].rename(
        columns={
            FALLBACK_FEATURE_COVERAGE_COLUMN: (
                "has_complete_fallback_features"
            ),
        }
    )

    feature_contributions = (
        create_long_logistic_feature_contributions(
            current_features=current_features,
            trained_model=primary_model,
        )
    )

    production_predictions = (
        prediction_spine.merge(
            route_frame,
            on="game_id",
            how="left",
            validate="one_to_one",
            sort=False,
        ).merge(
            feature_audit,
            on="game_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    production_predictions[
        "predicted_winner"
    ] = production_predictions[
        "home_team"
    ].where(
        production_predictions[
            "home_win_probability"
        ] >= 0.5,
        production_predictions[
            "away_team"
        ],
    )

    original_columns = list(
        elo_predictions.columns
    )

    output_columns = [
        *original_columns,
        *[
            column_name
            for column_name
            in PRODUCTION_AUDIT_COLUMNS
            if column_name
            not in original_columns
        ],
    ]

    production_predictions = (
        production_predictions.loc[
            :,
            output_columns,
        ]
    )

    contribution_metadata = (
        production_predictions.loc[
            :,
            [
                "game_id",
                *CONTRIBUTION_METADATA_COLUMNS,
            ],
        ]
    )

    feature_contributions = (
        feature_contributions.merge(
            contribution_metadata,
            on="game_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    feature_contributions = (
        feature_contributions.loc[
            :,
            CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS,
        ]
    )

    if return_feature_contributions:
        return (
            production_predictions,
            feature_contributions,
        )

    return production_predictions
