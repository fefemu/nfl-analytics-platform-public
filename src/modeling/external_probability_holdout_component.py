"""
NFL Analytics Platform
External Probability Holdout Component

Purpose:
    Train the locked current and external probability
    champions on pre-2025 games and prepare one paired
    evaluation on the protected 2025 holdout.

    This module has no standalone main entry point.
    The holdout is opened only by the final consolidated
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
    CURRENT_BLEND_CANDIDATE,
    EXTERNAL_BLEND_CANDIDATE,
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_LOGISTIC_FEATURES,
    EXTERNAL_QB_FEATURE,
    INTERNAL_LOGISTIC_FEATURES,
    LISTED_QB_FEATURE,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
    create_logistic_pipeline,
    evaluate_probabilities,
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

HOLDOUT_SEASON = 2025
HOLDOUT_SPLIT = "holdout"

SUMMARY_COLUMNS = (
    "candidate_name",
    "training_game_count",
    "holdout_game_count",
    "primary_holdout_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
)

PREDICTION_COLUMNS = (
    "game_id",
    "actual_home_win",
    "current_home_win_probability",
    "external_home_win_probability",
    "current_brier_loss",
    "external_brier_loss",
    "external_brier_loss_delta",
)


def load_probability_holdout_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load pre-2025 training and 2025 holdout rows."""

    injury_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name in UNIT_BURDEN_FEATURES
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,

            CAST(
                dataset.{TARGET_COLUMN}
                AS INTEGER
            ) AS {TARGET_COLUMN},

            dataset.{INJURY_AVAILABILITY_COLUMN},
            dataset.elo_rating_difference,
            dataset.{LISTED_QB_FEATURE},
            dataset.elo_home_win_probability,

            {injury_select},

            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS {EXTERNAL_ELO_FEATURE},

            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS {EXTERNAL_QB_FEATURE},

            external.nfelo_home_probability_open
                AS published_nfelo_home_probability

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

          AND splits.is_core_model_eligible = TRUE

          AND dataset.{INJURY_AVAILABILITY_COLUMN}
                = TRUE

        ORDER BY
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No probability holdout data is available."
        )

    return data


def prepare_probability_holdout_data(
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create identical training and holdout samples."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        TARGET_COLUMN,
        INJURY_AVAILABILITY_COLUMN,
        "elo_home_win_probability",
        "published_nfelo_home_probability",
        *INTERNAL_LOGISTIC_FEATURES,
        *EXTERNAL_LOGISTIC_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Probability holdout data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    if source_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Probability holdout data contains "
            "duplicate game identifiers."
        )

    if int(source_data["season"].max()) > (
        HOLDOUT_SEASON
    ):
        raise ValueError(
            "Probability holdout data contains "
            "post-2025 games."
        )

    complete_columns = [
        TARGET_COLUMN,
        "elo_home_win_probability",
        "published_nfelo_home_probability",
        *INTERNAL_LOGISTIC_FEATURES,
        *EXTERNAL_LOGISTIC_FEATURES,
    ]

    complete_data = source_data.loc[
        source_data[
            INJURY_AVAILABILITY_COLUMN
        ].fillna(False).astype(bool)
        & source_data[
            complete_columns
        ].notna().all(axis=1)
    ].copy()

    training_data = complete_data.loc[
        complete_data["season"]
        < HOLDOUT_SEASON
    ].copy()

    holdout_data = complete_data.loc[
        (
            complete_data["season"]
            == HOLDOUT_SEASON
        )
        & (
            complete_data["split_name"]
            == HOLDOUT_SPLIT
        )
    ].copy()

    if training_data.empty:
        raise RuntimeError(
            "No pre-2025 probability training "
            "games are available."
        )

    if holdout_data.empty:
        raise RuntimeError(
            "No protected 2025 probability holdout "
            "games are available."
        )

    if (
        training_data[TARGET_COLUMN].nunique()
        != 2
    ):
        raise RuntimeError(
            "Probability training data must contain "
            "both target classes."
        )

    if set(
        holdout_data["split_name"]
    ) != {
        HOLDOUT_SPLIT,
    }:
        raise RuntimeError(
            "Probability holdout routing contains "
            "unexpected splits."
        )

    probability_columns = [
        "elo_home_win_probability",
        "published_nfelo_home_probability",
    ]

    probability_values = complete_data[
        probability_columns
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(
            probability_values
        ).all()
        or (
            probability_values <= 0.0
        ).any()
        or (
            probability_values >= 1.0
        ).any()
    ):
        raise ValueError(
            "Holdout source probabilities must be "
            "between zero and one."
        )

    return training_data, holdout_data


def evaluate_locked_probability_holdout(
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both locked probability champions."""

    (
        training_data,
        holdout_data,
    ) = prepare_probability_holdout_data(
        source_data
    )

    regularization_c = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_regularization_c
    )

    logistic_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_weight
    )

    elo_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .elo_weight
    )

    internal_model = create_logistic_pipeline(
        feature_columns=(
            INTERNAL_LOGISTIC_FEATURES
        ),
        regularization_c=regularization_c,
    )

    external_model = create_logistic_pipeline(
        feature_columns=(
            EXTERNAL_LOGISTIC_FEATURES
        ),
        regularization_c=regularization_c,
    )

    internal_model.fit(
        training_data.loc[
            :,
            INTERNAL_LOGISTIC_FEATURES,
        ],
        training_data[TARGET_COLUMN],
    )

    external_model.fit(
        training_data.loc[
            :,
            EXTERNAL_LOGISTIC_FEATURES,
        ],
        training_data[TARGET_COLUMN],
    )

    internal_logistic_probability = (
        internal_model.predict_proba(
            holdout_data.loc[
                :,
                INTERNAL_LOGISTIC_FEATURES,
            ]
        )[:, 1]
    )

    external_logistic_probability = (
        external_model.predict_proba(
            holdout_data.loc[
                :,
                EXTERNAL_LOGISTIC_FEATURES,
            ]
        )[:, 1]
    )

    current_probability = (
        logistic_weight
        * internal_logistic_probability
        + elo_weight
        * holdout_data[
            "elo_home_win_probability"
        ].to_numpy(dtype=float)
    )

    external_probability = (
        logistic_weight
        * external_logistic_probability
        + elo_weight
        * holdout_data[
            "published_nfelo_home_probability"
        ].to_numpy(dtype=float)
    )

    actual_values = holdout_data[
        TARGET_COLUMN
    ]

    current_evaluation = evaluate_probabilities(
        actual_values=actual_values,
        probabilities=current_probability,
    )

    external_evaluation = evaluate_probabilities(
        actual_values=actual_values,
        probabilities=external_probability,
    )

    summary = pd.DataFrame(
        [
            {
                "candidate_name": (
                    CURRENT_BLEND_CANDIDATE
                ),
                "training_game_count": len(
                    training_data
                ),
                "holdout_game_count": len(
                    holdout_data
                ),
                "primary_holdout_game_count": len(
                    holdout_data
                ),
                "accuracy": (
                    current_evaluation.accuracy
                ),
                "brier_score": (
                    current_evaluation.brier_score
                ),
                "log_loss": (
                    current_evaluation.log_loss
                ),
            },
            {
                "candidate_name": (
                    EXTERNAL_BLEND_CANDIDATE
                ),
                "training_game_count": len(
                    training_data
                ),
                "holdout_game_count": len(
                    holdout_data
                ),
                "primary_holdout_game_count": len(
                    holdout_data
                ),
                "accuracy": (
                    external_evaluation.accuracy
                ),
                "brier_score": (
                    external_evaluation.brier_score
                ),
                "log_loss": (
                    external_evaluation.log_loss
                ),
            },
        ],
        columns=SUMMARY_COLUMNS,
    ).sort_values(
        by=[
            "brier_score",
            "log_loss",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    actual_array = actual_values.to_numpy(
        dtype=int
    )

    current_brier_loss = np.square(
        current_probability - actual_array
    )

    external_brier_loss = np.square(
        external_probability - actual_array
    )

    predictions = pd.DataFrame(
        {
            "game_id": holdout_data[
                "game_id"
            ].to_numpy(),
            "actual_home_win": actual_array,
            "current_home_win_probability": (
                current_probability
            ),
            "external_home_win_probability": (
                external_probability
            ),
            "current_brier_loss": (
                current_brier_loss
            ),
            "external_brier_loss": (
                external_brier_loss
            ),
            "external_brier_loss_delta": (
                external_brier_loss
                - current_brier_loss
            ),
        },
        columns=PREDICTION_COLUMNS,
    )

    return summary, predictions