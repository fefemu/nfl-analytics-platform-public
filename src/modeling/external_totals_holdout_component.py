"""
NFL Analytics Platform
External Totals Holdout Component

Purpose:
    Compare the locked current Totals routing with the
    locked external-QB primary and external Elo-QB
    fallback candidates on the protected 2025 holdout.

    Both systems use the same game-level routing mask.

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

from src.modeling.backtest_external_elo_totals_candidates import (
    EXTERNAL_ELO_SUM_FEATURE,
    EXTERNAL_QB_SUM_FEATURE,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.evaluate_totals_model_candidates import (
    RAW_TOTALS_FEATURE_COLUMNS,
    TOTALS_TARGET_COLUMN,
    create_totals_aggregate_features,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
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

CURRENT_CANDIDATE = (
    "current_production_totals_routing"
)

EXTERNAL_CANDIDATE = (
    "external_qb_primary_external_elo_qb_fallback"
)

INTERNAL_ELO_SUM_FEATURE = (
    "elo_rating_sum"
)

CURRENT_PRIMARY_FEATURES = tuple(
    PRODUCTION_TOTALS_MODEL.feature_columns
)

EXTERNAL_PRIMARY_FEATURES = (
    *CURRENT_PRIMARY_FEATURES,
    EXTERNAL_QB_SUM_FEATURE,
)

CURRENT_FALLBACK_FEATURES = tuple(
    PRODUCTION_TOTALS_MODEL
    .fallback_feature_columns
)

EXTERNAL_FALLBACK_FEATURES = (
    "league_average_total_last_64",
    "is_indoor",
    EXTERNAL_ELO_SUM_FEATURE,
    EXTERNAL_QB_SUM_FEATURE,
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "primary_training_game_count",
    "fallback_training_game_count",
    "holdout_game_count",
    "primary_holdout_game_count",
    "fallback_holdout_game_count",
    "primary_ridge_alpha",
    "fallback_ridge_alpha",
    "holdout_mae",
    "holdout_rmse",
    "holdout_bias",
    "holdout_r_squared",
)

PREDICTION_COLUMNS = (
    "game_id",
    "actual_total",
    "prediction_mode",
    "current_predicted_total",
    "external_predicted_total",
    "current_absolute_error",
    "external_absolute_error",
    "external_absolute_error_delta",
)


def load_totals_holdout_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load and aggregate Totals holdout inputs."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name
        in RAW_TOTALS_FEATURE_COLUMNS
    )

    source_data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,
            dataset.both_short_windows_complete,
            dataset.{TOTALS_TARGET_COLUMN},
            dataset.home_elo_rating,
            dataset.away_elo_rating,

            {feature_select},

            external.starting_nfelo_home
                + external.starting_nfelo_away
                AS {EXTERNAL_ELO_SUM_FEATURE},

            external.home_538_qb_adj
                + external.away_538_qb_adj
                AS {EXTERNAL_QB_SUM_FEATURE}

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

          AND splits.split_name IN (
            'train',
            'validation',
            'holdout'
        )

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if source_data.empty:
        raise RuntimeError(
            "No Totals holdout data is available."
        )

    data = create_totals_aggregate_features(
        source_data
    )

    data[INTERNAL_ELO_SUM_FEATURE] = (
        data["home_elo_rating"]
        + data["away_elo_rating"]
    )

    return data


def prepare_totals_routing_data(
    source_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create primary training, fallback and holdout."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        "both_short_windows_complete",
        TOTALS_TARGET_COLUMN,
        *CURRENT_PRIMARY_FEATURES,
        *EXTERNAL_PRIMARY_FEATURES,
        *CURRENT_FALLBACK_FEATURES,
        *EXTERNAL_FALLBACK_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Totals routing data is missing columns: "
            + ", ".join(missing_columns)
        )

    if source_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Totals routing data contains duplicate "
            "game identifiers."
        )

    if int(source_data["season"].max()) > (
        HOLDOUT_SEASON
    ):
        raise ValueError(
            "Totals routing data contains post-2025 "
            "games."
        )

    fallback_complete_columns = [
        TOTALS_TARGET_COLUMN,
        *CURRENT_FALLBACK_FEATURES,
        *EXTERNAL_FALLBACK_FEATURES,
    ]

    fallback_complete_data = source_data.loc[
        source_data[
            fallback_complete_columns
        ].notna().all(axis=1)
    ].copy()

    fallback_training_data = (
        fallback_complete_data.loc[
            fallback_complete_data["season"]
            < HOLDOUT_SEASON
        ].copy()
    )

    holdout_data = fallback_complete_data.loc[
        (
            fallback_complete_data["season"]
            == HOLDOUT_SEASON
        )
        & (
            fallback_complete_data["split_name"]
            == HOLDOUT_SPLIT
        )
    ].copy()

    primary_complete_columns = [
        *CURRENT_PRIMARY_FEATURES,
        *EXTERNAL_PRIMARY_FEATURES,
    ]

    primary_training_data = (
        fallback_training_data.loc[
            fallback_training_data[
                "both_short_windows_complete"
            ].fillna(False).astype(bool)
            & fallback_training_data[
                primary_complete_columns
            ].notna().all(axis=1)
        ].copy()
    )

    if fallback_training_data.empty:
        raise RuntimeError(
            "No pre-2025 Totals fallback training "
            "games are available."
        )

    if primary_training_data.empty:
        raise RuntimeError(
            "No pre-2025 Totals primary training "
            "games are available."
        )

    if holdout_data.empty:
        raise RuntimeError(
            "No protected 2025 Totals holdout games "
            "are available."
        )

    return (
        primary_training_data,
        fallback_training_data,
        holdout_data,
    )


def evaluate_locked_totals_routing_holdout(
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate current and external Totals routing."""

    (
        primary_training_data,
        fallback_training_data,
        holdout_data,
    ) = prepare_totals_routing_data(
        source_data
    )

    current_primary_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_TOTALS_MODEL
                .ridge_alpha
            )
        )
    )

    external_primary_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_TOTALS_MODEL
                .ridge_alpha
            )
        )
    )

    current_fallback_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_TOTALS_MODEL
                .fallback_ridge_alpha
            )
        )
    )

    external_fallback_model = (
        create_ridge_pipeline(
            ridge_alpha=(
                PRODUCTION_TOTALS_MODEL
                .fallback_ridge_alpha
            )
        )
    )

    current_primary_model.fit(
        primary_training_data.loc[
            :,
            CURRENT_PRIMARY_FEATURES,
        ],
        primary_training_data[
            TOTALS_TARGET_COLUMN
        ],
    )

    external_primary_model.fit(
        primary_training_data.loc[
            :,
            EXTERNAL_PRIMARY_FEATURES,
        ],
        primary_training_data[
            TOTALS_TARGET_COLUMN
        ],
    )

    current_fallback_model.fit(
        fallback_training_data.loc[
            :,
            CURRENT_FALLBACK_FEATURES,
        ],
        fallback_training_data[
            TOTALS_TARGET_COLUMN
        ],
    )

    external_fallback_model.fit(
        fallback_training_data.loc[
            :,
            EXTERNAL_FALLBACK_FEATURES,
        ],
        fallback_training_data[
            TOTALS_TARGET_COLUMN
        ],
    )

    primary_holdout_mask = (
        holdout_data[
            "both_short_windows_complete"
        ].fillna(False).astype(bool)
        & holdout_data[
            list(CURRENT_PRIMARY_FEATURES)
        ].notna().all(axis=1)
        & holdout_data[
            list(EXTERNAL_PRIMARY_FEATURES)
        ].notna().all(axis=1)
    )

    fallback_holdout_mask = (
        ~primary_holdout_mask
    )

    current_predictions = np.empty(
        len(holdout_data),
        dtype=float,
    )

    external_predictions = np.empty(
        len(holdout_data),
        dtype=float,
    )

    primary_mask_array = (
        primary_holdout_mask.to_numpy()
    )

    fallback_mask_array = (
        fallback_holdout_mask.to_numpy()
    )

    if primary_holdout_mask.any():
        current_predictions[
            primary_mask_array
        ] = current_primary_model.predict(
            holdout_data.loc[
                primary_holdout_mask,
                CURRENT_PRIMARY_FEATURES,
            ]
        )

        external_predictions[
            primary_mask_array
        ] = external_primary_model.predict(
            holdout_data.loc[
                primary_holdout_mask,
                EXTERNAL_PRIMARY_FEATURES,
            ]
        )

    if fallback_holdout_mask.any():
        current_predictions[
            fallback_mask_array
        ] = current_fallback_model.predict(
            holdout_data.loc[
                fallback_holdout_mask,
                CURRENT_FALLBACK_FEATURES,
            ]
        )

        external_predictions[
            fallback_mask_array
        ] = external_fallback_model.predict(
            holdout_data.loc[
                fallback_holdout_mask,
                EXTERNAL_FALLBACK_FEATURES,
            ]
        )

    actual_total = holdout_data[
        TOTALS_TARGET_COLUMN
    ]

    current_metrics = (
        calculate_regression_metrics(
            actual_margin=actual_total,
            predicted_margin=(
                current_predictions
            ),
        )
    )

    external_metrics = (
        calculate_regression_metrics(
            actual_margin=actual_total,
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
                "primary_ridge_alpha": (
                    PRODUCTION_TOTALS_MODEL
                    .ridge_alpha
                ),
                "fallback_ridge_alpha": (
                    PRODUCTION_TOTALS_MODEL
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
                "primary_ridge_alpha": (
                    PRODUCTION_TOTALS_MODEL
                    .ridge_alpha
                ),
                "fallback_ridge_alpha": (
                    PRODUCTION_TOTALS_MODEL
                    .fallback_ridge_alpha
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

    actual_array = actual_total.to_numpy(
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

    prediction_modes = np.where(
        primary_mask_array,
        "PRIMARY",
        "FALLBACK",
    )

    predictions = pd.DataFrame(
        {
            "game_id": holdout_data[
                "game_id"
            ].to_numpy(),
            "actual_total": actual_array,
            "prediction_mode": (
                prediction_modes
            ),
            "current_predicted_total": (
                current_predictions
            ),
            "external_predicted_total": (
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