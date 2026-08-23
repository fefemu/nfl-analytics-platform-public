"""
NFL Analytics Platform
External Spread Holdout Component

Purpose:
    Compare the locked current Spread routing with the
    locked external nfelo Elo and QB candidate on the
    protected 2025 holdout.

    This module has no standalone main entry point.
    The holdout is opened only by the consolidated
    external-model holdout orchestrator.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import duckdb
import numpy as np
import pandas as pd

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
    LISTED_QB_FEATURE,
)
from src.modeling.backtest_elo_rating_sources import (
    SPREAD_RIDGE_ALPHA,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
)


DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)

SPLIT_FULL_NAME = (
    "analytics.modeling_game_splits"
)

EXTERNAL_RATINGS_FULL_NAME = (
    "processed.external_nfelo_game_ratings"
)

SPREAD_TARGET_COLUMN = (
    "target_point_differential"
)

CURRENT_CANDIDATE = (
    "current_production_spread_routing"
)

EXTERNAL_CANDIDATE = (
    "external_nfelo_external_qb_spread"
)

INTERNAL_ELO_FEATURE = (
    "elo_rating_difference"
)

CURRENT_PRIMARY_FEATURES = tuple(
    PRODUCTION_SPREAD_MODEL.feature_columns
)

CURRENT_FALLBACK_FEATURES = tuple(
    PRODUCTION_SPREAD_MODEL
    .fallback_feature_columns
)

EXTERNAL_FEATURES = (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "primary_training_game_count",
    "fallback_training_game_count",
    "holdout_game_count",
    "primary_holdout_game_count",
    "fallback_holdout_game_count",
    "ridge_alpha",
    "fallback_ridge_alpha",
    "holdout_mae",
    "holdout_rmse",
    "holdout_bias",
    "holdout_r_squared",
)

PREDICTION_COLUMNS = (
    "game_id",
    "actual_home_margin",
    "current_prediction_mode",
    "current_predicted_home_margin",
    "external_predicted_home_margin",
    "current_absolute_error",
    "external_absolute_error",
    "external_absolute_error_delta",
)


def load_spread_holdout_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load pre-2025 training and 2025 holdout rows."""

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,
            dataset.{SPREAD_TARGET_COLUMN},
            dataset.{INTERNAL_ELO_FEATURE},
            dataset.{LISTED_QB_FEATURE},

            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS {EXTERNAL_ELO_FEATURE},

            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS {EXTERNAL_QB_FEATURE}

        FROM {DATASET_FULL_NAME}
            AS dataset

        INNER JOIN {SPLIT_FULL_NAME}
            AS splits
            ON dataset.game_id = splits.game_id

        INNER JOIN {EXTERNAL_RATINGS_FULL_NAME}
            AS external
            ON dataset.game_id
                = external.normalized_game_id

        WHERE dataset.season <= {HOLDOUT_SEASON}

        ORDER BY
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No Spread holdout data is available."
        )

    return data


def prepare_spread_holdout_data(
    source_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create primary, fallback and holdout samples."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        SPREAD_TARGET_COLUMN,
        INTERNAL_ELO_FEATURE,
        LISTED_QB_FEATURE,
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    }

    missing_columns = sorted(
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Spread holdout data is missing columns: "
            + ", ".join(missing_columns)
        )

    if source_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Spread holdout data contains duplicate "
            "game identifiers."
        )

    if int(source_data["season"].max()) > (
        HOLDOUT_SEASON
    ):
        raise ValueError(
            "Spread holdout data contains post-2025 "
            "games."
        )

    common_columns = [
        SPREAD_TARGET_COLUMN,
        INTERNAL_ELO_FEATURE,
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    ]

    common_data = source_data.loc[
        source_data[
            common_columns
        ].notna().all(axis=1)
    ].copy()

    fallback_training_data = common_data.loc[
        common_data["season"]
        < HOLDOUT_SEASON
    ].copy()

    primary_training_data = (
        fallback_training_data.loc[
            fallback_training_data[
                LISTED_QB_FEATURE
            ].notna()
        ].copy()
    )

    holdout_data = common_data.loc[
        (
            common_data["season"]
            == HOLDOUT_SEASON
        )
        & (
            common_data["split_name"]
            == HOLDOUT_SPLIT
        )
    ].copy()

    if fallback_training_data.empty:
        raise RuntimeError(
            "No pre-2025 Spread fallback training "
            "games are available."
        )

    if primary_training_data.empty:
        raise RuntimeError(
            "No pre-2025 Spread primary training "
            "games are available."
        )

    if holdout_data.empty:
        raise RuntimeError(
            "No protected 2025 Spread holdout games "
            "are available."
        )

    return (
        primary_training_data,
        fallback_training_data,
        holdout_data,
    )


def evaluate_locked_spread_holdout(
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate current routing and external candidate."""

    (
        primary_training_data,
        fallback_training_data,
        holdout_data,
    ) = prepare_spread_holdout_data(
        source_data
    )

    current_primary_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_SPREAD_MODEL
                .ridge_alpha
            )
        )
    )

    current_fallback_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_SPREAD_MODEL
                .fallback_ridge_alpha
            )
        )
    )

    external_model = create_ridge_pipeline(
        ridge_alpha=SPREAD_RIDGE_ALPHA
    )

    current_primary_model.fit(
        primary_training_data.loc[
            :,
            CURRENT_PRIMARY_FEATURES,
        ],
        primary_training_data[
            SPREAD_TARGET_COLUMN
        ],
    )

    current_fallback_model.fit(
        fallback_training_data.loc[
            :,
            CURRENT_FALLBACK_FEATURES,
        ],
        fallback_training_data[
            SPREAD_TARGET_COLUMN
        ],
    )

    external_model.fit(
        fallback_training_data.loc[
            :,
            EXTERNAL_FEATURES,
        ],
        fallback_training_data[
            SPREAD_TARGET_COLUMN
        ],
    )

    primary_holdout_mask = holdout_data[
        LISTED_QB_FEATURE
    ].notna()

    fallback_holdout_mask = (
        ~primary_holdout_mask
    )

    current_predictions = np.empty(
        len(holdout_data),
        dtype=float,
    )

    if primary_holdout_mask.any():
        current_predictions[
            primary_holdout_mask.to_numpy()
        ] = current_primary_model.predict(
            holdout_data.loc[
                primary_holdout_mask,
                CURRENT_PRIMARY_FEATURES,
            ]
        )

    if fallback_holdout_mask.any():
        current_predictions[
            fallback_holdout_mask.to_numpy()
        ] = current_fallback_model.predict(
            holdout_data.loc[
                fallback_holdout_mask,
                CURRENT_FALLBACK_FEATURES,
            ]
        )

    external_predictions = external_model.predict(
        holdout_data.loc[
            :,
            EXTERNAL_FEATURES,
        ]
    )

    actual_margin = holdout_data[
        SPREAD_TARGET_COLUMN
    ]

    current_metrics = (
        calculate_regression_metrics(
            actual_margin=actual_margin,
            predicted_margin=(
                current_predictions
            ),
        )
    )

    external_metrics = (
        calculate_regression_metrics(
            actual_margin=actual_margin,
            predicted_margin=(
                external_predictions
            ),
        )
    )

    primary_holdout_count = int(
        primary_holdout_mask.sum()
    )

    fallback_holdout_count = int(
        fallback_holdout_mask.sum()
    )

    summary = pd.DataFrame(
        [
            {
                "candidate_name": (
                    CURRENT_CANDIDATE
                ),
                "primary_training_game_count": (
                    len(primary_training_data)
                ),
                "fallback_training_game_count": (
                    len(fallback_training_data)
                ),
                "holdout_game_count": len(
                    holdout_data
                ),
                "primary_holdout_game_count": (
                    primary_holdout_count
                ),
                "fallback_holdout_game_count": (
                    fallback_holdout_count
                ),
                "ridge_alpha": (
                    PRODUCTION_SPREAD_MODEL
                    .ridge_alpha
                ),
                "fallback_ridge_alpha": (
                    PRODUCTION_SPREAD_MODEL
                    .fallback_ridge_alpha
                ),
                "holdout_mae": current_metrics[
                    "validation_mae"
                ],
                "holdout_rmse": current_metrics[
                    "validation_rmse"
                ],
                "holdout_bias": current_metrics[
                    "validation_bias"
                ],
                "holdout_r_squared": (
                    current_metrics[
                        "validation_r_squared"
                    ]
                ),
            },
            {
                "candidate_name": (
                    EXTERNAL_CANDIDATE
                ),
                "primary_training_game_count": (
                    len(fallback_training_data)
                ),
                "fallback_training_game_count": (
                    len(fallback_training_data)
                ),
                "holdout_game_count": len(
                    holdout_data
                ),
                "primary_holdout_game_count": len(
                    holdout_data
                ),
                "fallback_holdout_game_count": 0,
                "ridge_alpha": (
                    SPREAD_RIDGE_ALPHA
                ),
                "fallback_ridge_alpha": (
                    SPREAD_RIDGE_ALPHA
                ),
                "holdout_mae": external_metrics[
                    "validation_mae"
                ],
                "holdout_rmse": external_metrics[
                    "validation_rmse"
                ],
                "holdout_bias": external_metrics[
                    "validation_bias"
                ],
                "holdout_r_squared": (
                    external_metrics[
                        "validation_r_squared"
                    ]
                ),
            },
        ],
        columns=SUMMARY_COLUMNS,
    ).sort_values(
        by=[
            "holdout_mae",
            "holdout_rmse",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    actual_array = actual_margin.to_numpy(
        dtype=float
    )

    current_absolute_error = np.abs(
        current_predictions
        - actual_array
    )

    external_absolute_error = np.abs(
        external_predictions
        - actual_array
    )

    current_modes = np.where(
        primary_holdout_mask.to_numpy(),
        "PRIMARY",
        "FALLBACK",
    )

    predictions = pd.DataFrame(
        {
            "game_id": holdout_data[
                "game_id"
            ].to_numpy(),
            "actual_home_margin": (
                actual_array
            ),
            "current_prediction_mode": (
                current_modes
            ),
            "current_predicted_home_margin": (
                current_predictions
            ),
            "external_predicted_home_margin": (
                external_predictions
            ),
            "current_absolute_error": (
                current_absolute_error
            ),
            "external_absolute_error": (
                external_absolute_error
            ),
            "external_absolute_error_delta": (
                external_absolute_error
                - current_absolute_error
            ),
        },
        columns=PREDICTION_COLUMNS,
    )

    return summary, predictions