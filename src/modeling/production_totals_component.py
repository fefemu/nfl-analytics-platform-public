"""
NFL Analytics Platform
Production Totals Component

Purpose:
    Train the frozen primary and fallback totals models
    and route current games according to feature
    availability.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.evaluate_totals_model_candidates import (
    create_ridge_pipeline,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
    ProductionTotalsModel,
)


PRIMARY_PREDICTION_MODE = "RIDGE_TOTALS_PRIMARY"
FALLBACK_PREDICTION_MODE = "RIDGE_TOTALS_FALLBACK"

PRIMARY_REASON = "complete_locked_totals_features"
FALLBACK_REASON = (
    "missing_primary_rolling_or_qb_features"
)

OUTPUT_COLUMNS = (
    "game_id",
    "home_team",
    "away_team",
    "model_name",
    "model_version",
    "prediction_mode",
    "prediction_mode_reason",
    "ridge_alpha",
    "primary_training_game_count",
    "fallback_training_game_count",
    "has_complete_primary_features",
    "predicted_total_points",
)


@dataclass(frozen=True)
class TrainedProductionTotalsModels:
    """Store fitted primary and fallback models."""

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


def prepare_production_totals_training_data(
    historical_data: pd.DataFrame,
    production_model: ProductionTotalsModel = (
        PRODUCTION_TOTALS_MODEL
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare primary and fallback training samples."""

    required_columns = {
        "game_id",
        "season",
        "both_short_windows_complete",
        production_model.target_column,
        *production_model.feature_columns,
        *production_model.fallback_feature_columns,
    }

    validate_required_columns(
        data=historical_data,
        required_columns=required_columns,
        data_name="Historical totals data",
    )

    if historical_data["game_id"].duplicated().any():
        raise ValueError(
            "Historical totals data contains duplicate "
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

    primary_training_data = eligible_data.loc[
        eligible_data[
            "both_short_windows_complete"
        ].fillna(False).astype(bool)
        & eligible_data[
            list(production_model.feature_columns)
        ].notna().all(axis=1)
    ].copy()

    fallback_training_data = eligible_data.loc[
        eligible_data[
            list(
                production_model
                .fallback_feature_columns
            )
        ].notna().all(axis=1)
    ].copy()

    if primary_training_data.empty:
        raise RuntimeError(
            "No eligible complete primary totals "
            "training games are available."
        )

    if fallback_training_data.empty:
        raise RuntimeError(
            "No eligible complete fallback totals "
            "training games are available."
        )

    return (
        primary_training_data,
        fallback_training_data,
    )


def train_production_totals_models(
    historical_data: pd.DataFrame,
    production_model: ProductionTotalsModel = (
        PRODUCTION_TOTALS_MODEL
    ),
) -> TrainedProductionTotalsModels:
    """Train the frozen primary and fallback models."""

    (
        primary_training_data,
        fallback_training_data,
    ) = prepare_production_totals_training_data(
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

    return TrainedProductionTotalsModels(
        primary_model=primary_model,
        fallback_model=fallback_model,
        primary_training_game_count=len(
            primary_training_data
        ),
        fallback_training_game_count=len(
            fallback_training_data
        ),
    )


def score_current_totals_predictions(
    current_features: pd.DataFrame,
    trained_models: TrainedProductionTotalsModels,
    production_model: ProductionTotalsModel = (
        PRODUCTION_TOTALS_MODEL
    ),
) -> pd.DataFrame:
    """Score current games with primary/fallback routing."""

    required_columns = {
        "game_id",
        "home_team",
        "away_team",
        "both_short_windows_complete",
        *production_model.feature_columns,
        *production_model.fallback_feature_columns,
    }

    validate_required_columns(
        data=current_features,
        required_columns=required_columns,
        data_name="Current totals features",
    )

    if current_features["game_id"].duplicated().any():
        raise ValueError(
            "Current totals features contain duplicate "
            "game identifiers."
        )

    missing_fallback_mask = current_features[
        list(
            production_model
            .fallback_feature_columns
        )
    ].isna().any(axis=1)

    if missing_fallback_mask.any():
        missing_game_ids = ", ".join(
            current_features.loc[
                missing_fallback_mask,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Current totals games are missing fallback "
            f"features: {missing_game_ids}"
        )

    scored = current_features.copy()

    complete_primary_mask = (
        scored[
            "both_short_windows_complete"
        ].fillna(False).astype(bool)
        & scored[
            list(production_model.feature_columns)
        ].notna().all(axis=1)
    )

    scored[
        "has_complete_primary_features"
    ] = complete_primary_mask

    scored["predicted_total_points"] = np.nan
    scored["prediction_mode"] = ""
    scored["prediction_mode_reason"] = ""
    scored["ridge_alpha"] = np.nan
    scored["model_name"] = ""

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
            "predicted_total_points",
        ] = primary_predictions

        scored.loc[
            complete_primary_mask,
            "prediction_mode",
        ] = PRIMARY_PREDICTION_MODE

        scored.loc[
            complete_primary_mask,
            "prediction_mode_reason",
        ] = PRIMARY_REASON

        scored.loc[
            complete_primary_mask,
            "ridge_alpha",
        ] = production_model.ridge_alpha

        scored.loc[
            complete_primary_mask,
            "model_name",
        ] = production_model.model_name

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
            "predicted_total_points",
        ] = fallback_predictions

        scored.loc[
            fallback_mask,
            "prediction_mode",
        ] = FALLBACK_PREDICTION_MODE

        scored.loc[
            fallback_mask,
            "prediction_mode_reason",
        ] = FALLBACK_REASON

        scored.loc[
            fallback_mask,
            "ridge_alpha",
        ] = (
            production_model.fallback_ridge_alpha
        )

        scored.loc[
            fallback_mask,
            "model_name",
        ] = production_model.fallback_model_name

    if not np.isfinite(
        scored["predicted_total_points"]
    ).all():
        raise RuntimeError(
            "Production totals predictions must "
            "be finite."
        )

    scored["predicted_total_points"] = scored[
        "predicted_total_points"
    ].astype(float)

    scored["model_version"] = (
        production_model.model_version
    )

    scored[
        "primary_training_game_count"
    ] = trained_models.primary_training_game_count

    scored[
        "fallback_training_game_count"
    ] = trained_models.fallback_training_game_count

    return scored.loc[
        :,
        OUTPUT_COLUMNS,
    ]