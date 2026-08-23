"""
NFL Analytics Platform
Production Spread Component

Purpose:
    Train the frozen spread model on eligible historical
    games and score current games.

    Complete Elo and QB features use the selected
    Elo + QB Ridge model. Missing QB information uses
    the validation-selected Elo-only fallback.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.evaluate_spread_model_candidates import (
    create_ridge_pipeline,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
    ProductionSpreadModel,
)


PRIMARY_PREDICTION_MODE = "EXTERNAL_NFELO_QB_RIDGE"
FALLBACK_PREDICTION_MODE = "EXTERNAL_NFELO_QB_RIDGE"

COMPLETE_QB_REASON = "complete_external_nfelo_qb_features"
MISSING_QB_REASON = "complete_external_nfelo_qb_features"

OUTPUT_COLUMNS = (
    "game_id",
    "home_team",
    "away_team",
    "model_name",
    "model_version",
    "prediction_mode",
    "prediction_mode_reason",
    "ridge_alpha",
    "external_nfelo_rating_difference",
    "listed_qb_rating_difference",
    "external_nfelo_qb_adjustment_difference",
    "both_listed_qb_ratings_available",
    "predicted_home_margin",
    "predicted_away_margin",
    "predicted_winner",
)


@dataclass(frozen=True)
class TrainedProductionSpreadModels:
    """Store the primary and fallback fitted models."""

    primary_model: Pipeline
    fallback_model: Pipeline
    primary_training_game_count: int
    fallback_training_game_count: int


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


def prepare_production_spread_training_data(
    historical_data: pd.DataFrame,
    production_model: ProductionSpreadModel = (
        PRODUCTION_SPREAD_MODEL
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare primary and fallback training samples."""

    required_columns = {
        "game_id",
        "season",
        production_model.target_column,
        *production_model.feature_columns,
        *production_model.fallback_feature_columns,
    }

    validate_required_columns(
        data=historical_data,
        required_columns=required_columns,
        data_name="Historical spread data",
    )

    if historical_data["game_id"].duplicated().any():
        raise ValueError(
            "Historical spread data contains duplicate "
            "game identifiers."
        )

    eligible_mask = (
        historical_data["season"].lt(
            production_model.forward_test_season
        )
        & historical_data[
            production_model.target_column
        ].notna()
    )

    eligible_data = historical_data.loc[
        eligible_mask
    ].copy()

    fallback_training_data = eligible_data.loc[
        eligible_data[
            list(
                production_model
                .fallback_feature_columns
            )
        ].notna().all(axis=1)
    ].copy()

    primary_training_data = eligible_data.loc[
        eligible_data[
            list(production_model.feature_columns)
        ].notna().all(axis=1)
    ].copy()

    if fallback_training_data.empty:
        raise RuntimeError(
            "No eligible Elo spread training games "
            "are available."
        )

    if primary_training_data.empty:
        raise RuntimeError(
            "No eligible Elo + QB spread training games "
            "are available."
        )

    return (
        primary_training_data,
        fallback_training_data,
    )


def train_production_spread_models(
    historical_data: pd.DataFrame,
    production_model: ProductionSpreadModel = (
        PRODUCTION_SPREAD_MODEL
    ),
) -> TrainedProductionSpreadModels:
    """Train the primary and fallback spread models."""

    (
        primary_training_data,
        fallback_training_data,
    ) = prepare_production_spread_training_data(
        historical_data=historical_data,
        production_model=production_model,
    )

    primary_model = create_ridge_pipeline(
        ridge_alpha=production_model.ridge_alpha
    )

    fallback_model = create_ridge_pipeline(
        ridge_alpha=(
            production_model.fallback_ridge_alpha
        )
    )

    primary_model.fit(
        primary_training_data.loc[
            :,
            list(production_model.feature_columns),
        ],
        primary_training_data[
            production_model.target_column
        ],
    )

    fallback_model.fit(
        fallback_training_data.loc[
            :,
            list(
                production_model
                .fallback_feature_columns
            ),
        ],
        fallback_training_data[
            production_model.target_column
        ],
    )

    return TrainedProductionSpreadModels(
        primary_model=primary_model,
        fallback_model=fallback_model,
        primary_training_game_count=len(
            primary_training_data
        ),
        fallback_training_game_count=len(
            fallback_training_data
        ),
    )


def score_current_spread_predictions(
    current_features: pd.DataFrame,
    trained_models: TrainedProductionSpreadModels,
    production_model: ProductionSpreadModel = (
        PRODUCTION_SPREAD_MODEL
    ),
) -> pd.DataFrame:
    """Score current games with primary/fallback routing."""

    required_columns = {
        "game_id",
        "home_team",
        "away_team",
        *production_model.fallback_feature_columns,
        "listed_qb_rating_difference",
    }

    validate_required_columns(
        data=current_features,
        required_columns=required_columns,
        data_name="Current spread features",
    )

    if current_features["game_id"].duplicated().any():
        raise ValueError(
            "Current spread features contain duplicate "
            "game identifiers."
        )

    missing_elo_mask = current_features[
        list(production_model.fallback_feature_columns)
    ].isna().any(axis=1)

    if missing_elo_mask.any():
        missing_game_ids = ", ".join(
            current_features.loc[
                missing_elo_mask,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Current spread games are missing Elo "
            f"features: {missing_game_ids}"
        )

    scored = current_features.copy()

    complete_primary_mask = scored[
        list(production_model.feature_columns)
    ].notna().all(axis=1)

    scored[
        "both_listed_qb_ratings_available"
    ] = scored[
        "listed_qb_rating_difference"
    ].notna()

    scored["predicted_home_margin"] = np.nan
    scored["prediction_mode"] = ""
    scored["prediction_mode_reason"] = ""
    scored["ridge_alpha"] = np.nan

    if complete_primary_mask.any():
        primary_predictions = (
            trained_models.primary_model.predict(
                scored.loc[
                    complete_primary_mask,
                    list(
                        production_model
                        .feature_columns
                    ),
                ]
            )
        )

        scored.loc[
            complete_primary_mask,
            "predicted_home_margin",
        ] = primary_predictions

        scored.loc[
            complete_primary_mask,
            "prediction_mode",
        ] = PRIMARY_PREDICTION_MODE

        scored.loc[
            complete_primary_mask,
            "prediction_mode_reason",
        ] = COMPLETE_QB_REASON

        scored.loc[
            complete_primary_mask,
            "ridge_alpha",
        ] = production_model.ridge_alpha

    fallback_mask = ~complete_primary_mask

    if fallback_mask.any():
        fallback_predictions = (
            trained_models.fallback_model.predict(
                scored.loc[
                    fallback_mask,
                    list(
                        production_model
                        .fallback_feature_columns
                    ),
                ]
            )
        )

        scored.loc[
            fallback_mask,
            "predicted_home_margin",
        ] = fallback_predictions

        scored.loc[
            fallback_mask,
            "prediction_mode",
        ] = FALLBACK_PREDICTION_MODE

        scored.loc[
            fallback_mask,
            "prediction_mode_reason",
        ] = MISSING_QB_REASON

        scored.loc[
            fallback_mask,
            "ridge_alpha",
        ] = (
            production_model.fallback_ridge_alpha
        )

    scored["predicted_home_margin"] = scored[
        "predicted_home_margin"
    ].astype(float)

    if not np.isfinite(
        scored["predicted_home_margin"]
    ).all():
        raise RuntimeError(
            "Production spread predictions must "
            "be finite."
        )

    scored["predicted_away_margin"] = (
        -scored["predicted_home_margin"]
    )

    scored["predicted_winner"] = np.where(
        scored["predicted_home_margin"] >= 0.0,
        scored["home_team"],
        scored["away_team"],
    )

    scored["model_name"] = np.where(
        complete_primary_mask,
        production_model.model_name,
        production_model.fallback_model_name,
    )

    scored["model_version"] = (
        production_model.model_version
    )

    return scored.loc[
        :,
        OUTPUT_COLUMNS,
    ]
