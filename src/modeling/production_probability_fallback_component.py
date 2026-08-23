"""
NFL Analytics Platform
Production Probability Fallback Component

Purpose:
    Train and score the selected external Elo-QB
    logistic probability fallback.

    The component is trained on every eligible historical
    game with a binary target and complete external
    fallback features. Injury and listed-QB availability
    are not required.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
    ProductionProbabilityModel,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
    create_logistic_pipeline,
)


FALLBACK_PROBABILITY_COLUMN = (
    "fallback_logistic_home_win_probability"
)

FALLBACK_FEATURE_COVERAGE_COLUMN = (
    "has_complete_fallback_features"
)


@dataclass(frozen=True)
class TrainedProbabilityFallback:
    """Store the fitted fallback and training metadata."""

    model: Pipeline
    feature_columns: tuple[str, ...]
    regularization_c: float
    training_game_count: int


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate one fallback data frame."""

    missing_columns = sorted(
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def prepare_fallback_training_data(
    historical_data: pd.DataFrame,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> pd.DataFrame:
    """Create complete binary fallback training rows."""

    feature_columns = (
        production_model
        .fallback_feature_columns
    )

    validate_required_columns(
        data=historical_data,
        required_columns={
            "game_id",
            TARGET_COLUMN,
            *feature_columns,
        },
        data_name=(
            "Probability fallback training data"
        ),
    )

    if historical_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Probability fallback training data "
            "contains duplicate game identifiers."
        )

    complete_columns = [
        TARGET_COLUMN,
        *feature_columns,
    ]

    training_data = historical_data.loc[
        historical_data[
            complete_columns
        ].notna().all(axis=1)
    ].copy()

    if training_data.empty:
        raise RuntimeError(
            "Probability fallback training data "
            "contains no complete rows."
        )

    target_values = training_data[
        TARGET_COLUMN
    ]

    if not target_values.isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Probability fallback target must contain "
            "only zero and one."
        )

    if target_values.nunique() != 2:
        raise RuntimeError(
            "Probability fallback training data must "
            "contain both target classes."
        )

    feature_values = training_data.loc[
        :,
        feature_columns,
    ].to_numpy(dtype=float)

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Probability fallback training features "
            "must be finite."
        )

    return training_data


def train_probability_fallback_component(
    historical_data: pd.DataFrame,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> TrainedProbabilityFallback:
    """Fit the selected external fallback logistic."""

    training_data = (
        prepare_fallback_training_data(
            historical_data=historical_data,
            production_model=production_model,
        )
    )

    feature_columns = (
        production_model
        .fallback_feature_columns
    )

    model = create_logistic_pipeline(
        feature_columns=feature_columns,
        regularization_c=(
            production_model
            .fallback_regularization_c
        ),
    )

    model.fit(
        training_data.loc[
            :,
            feature_columns,
        ],
        training_data[TARGET_COLUMN],
    )

    return TrainedProbabilityFallback(
        model=model,
        feature_columns=feature_columns,
        regularization_c=(
            production_model
            .fallback_regularization_c
        ),
        training_game_count=len(
            training_data
        ),
    )


def score_probability_fallback_component(
    current_features: pd.DataFrame,
    trained_fallback: TrainedProbabilityFallback,
) -> pd.DataFrame:
    """Score every current game with the fallback."""

    validate_required_columns(
        data=current_features,
        required_columns={
            "game_id",
            FALLBACK_FEATURE_COVERAGE_COLUMN,
            *trained_fallback.feature_columns,
        },
        data_name=(
            "Current probability fallback features"
        ),
    )

    if current_features[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Current probability fallback features "
            "contain duplicate game identifiers."
        )

    if current_features.empty:
        empty_output = current_features.copy()

        empty_output[
            FALLBACK_PROBABILITY_COLUMN
        ] = pd.Series(dtype=float)

        return empty_output

    coverage = current_features[
        FALLBACK_FEATURE_COVERAGE_COLUMN
    ].fillna(False).astype(bool)

    if not coverage.all():
        missing_game_ids = ", ".join(
            current_features.loc[
                ~coverage,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Current games are missing probability "
            "fallback features: "
            f"{missing_game_ids}"
        )

    feature_values = current_features.loc[
        :,
        trained_fallback.feature_columns,
    ].to_numpy(dtype=float)

    if not np.isfinite(
        feature_values
    ).all():
        raise ValueError(
            "Current probability fallback features "
            "must be finite."
        )

    probabilities = (
        trained_fallback.model.predict_proba(
            current_features.loc[
                :,
                trained_fallback.feature_columns,
            ]
        )[:, 1]
    )

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise RuntimeError(
            "Probability fallback model produced "
            "invalid probabilities."
        )

    scored_features = current_features.copy()

    scored_features[
        FALLBACK_PROBABILITY_COLUMN
    ] = probabilities

    return scored_features