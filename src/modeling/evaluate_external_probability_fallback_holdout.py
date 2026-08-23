"""
NFL Analytics Platform
External Probability Fallback Holdout

Purpose:
    Evaluate the development-locked external Elo and QB
    logistic fallback on the 2025 games excluded from the
    complete injury-enhanced primary holdout sample.

    The fallback candidate and logistic C were locked
    before this evaluation.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_LOGISTIC_FEATURES,
    EXTERNAL_QB_FEATURE,
    INTERNAL_LOGISTIC_FEATURES,
)
from src.modeling.backtest_external_probability_fallback import (
    CURRENT_FALLBACK_CANDIDATE,
    EXTERNAL_LOGISTIC_CANDIDATE,
    PRIMARY_ELIGIBILITY_COLUMN,
)
from src.modeling.evaluate_external_model_upgrade_holdout import (
    create_layer_paired_summary,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    create_logistic_pipeline,
    evaluate_probabilities,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)

SPLIT_FULL_NAME = (
    "analytics.modeling_game_splits"
)

EXTERNAL_RATINGS_FULL_NAME = (
    "processed.external_nfelo_game_ratings"
)

FALLBACK_FEATURES = (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "training_game_count",
    "fallback_holdout_game_count",
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


def load_probability_fallback_holdout_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load all inputs required for fallback routing."""

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
            splits.is_core_model_eligible,

            CAST(
                dataset.{TARGET_COLUMN}
                AS INTEGER
            ) AS {TARGET_COLUMN},

            dataset.{INJURY_AVAILABILITY_COLUMN},
            dataset.elo_rating_difference,
            dataset.listed_qb_rating_difference,
            dataset.elo_home_win_probability,

            {injury_select},

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
            "No probability fallback holdout data "
            "is available."
        )

    return data


def prepare_probability_fallback_holdout(
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and routed fallback holdout."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        "is_core_model_eligible",
        TARGET_COLUMN,
        INJURY_AVAILABILITY_COLUMN,
        "elo_home_win_probability",
        *INTERNAL_LOGISTIC_FEATURES,
        *EXTERNAL_LOGISTIC_FEATURES,
    }

    missing_columns = sorted(
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Probability fallback holdout data is "
            "missing columns: "
            + ", ".join(missing_columns)
        )

    if source_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Probability fallback holdout data "
            "contains duplicate game identifiers."
        )

    if int(source_data["season"].max()) > (
        HOLDOUT_SEASON
    ):
        raise ValueError(
            "Probability fallback holdout data "
            "contains post-2025 games."
        )

    common_columns = [
        TARGET_COLUMN,
        "elo_home_win_probability",
        *FALLBACK_FEATURES,
    ]

    sample = source_data.loc[
        source_data[
            common_columns
        ].notna().all(axis=1)
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete probability fallback "
            "holdout data is available."
        )

    internal_primary_complete = sample[
        list(INTERNAL_LOGISTIC_FEATURES)
    ].notna().all(axis=1)

    external_primary_complete = sample[
        list(EXTERNAL_LOGISTIC_FEATURES)
    ].notna().all(axis=1)

    sample[
        PRIMARY_ELIGIBILITY_COLUMN
    ] = (
        sample[
            "is_core_model_eligible"
        ].fillna(False).astype(bool)
        & sample[
            INJURY_AVAILABILITY_COLUMN
        ].fillna(False).astype(bool)
        & internal_primary_complete
        & external_primary_complete
    )

    training_data = sample.loc[
        sample["season"] < HOLDOUT_SEASON
    ].copy()

    fallback_holdout = sample.loc[
        (
            sample["season"]
            == HOLDOUT_SEASON
        )
        & (
            sample["split_name"]
            == HOLDOUT_SPLIT
        )
        & (
            ~sample[
                PRIMARY_ELIGIBILITY_COLUMN
            ]
        )
    ].copy()

    if training_data.empty:
        raise RuntimeError(
            "No pre-2025 probability fallback "
            "training games are available."
        )

    if fallback_holdout.empty:
        raise RuntimeError(
            "No routed 2025 probability fallback "
            "games are available."
        )

    if (
        training_data[TARGET_COLUMN].nunique()
        != 2
    ):
        raise RuntimeError(
            "Probability fallback training data "
            "must contain both target classes."
        )

    internal_probabilities = sample[
        "elo_home_win_probability"
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(
            internal_probabilities
        ).all()
        or (
            internal_probabilities <= 0.0
        ).any()
        or (
            internal_probabilities >= 1.0
        ).any()
    ):
        raise ValueError(
            "Internal fallback probabilities must "
            "be between zero and one."
        )

    return training_data, fallback_holdout


def evaluate_locked_probability_fallback_holdout(
    source_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate the locked external fallback."""

    (
        training_data,
        fallback_holdout,
    ) = prepare_probability_fallback_holdout(
        source_data
    )

    external_model = (
        create_logistic_pipeline(
            feature_columns=FALLBACK_FEATURES,
            regularization_c=(
                PRODUCTION_PROBABILITY_MODEL
                .logistic_regularization_c
            ),
        )
    )

    external_model.fit(
        training_data.loc[
            :,
            FALLBACK_FEATURES,
        ],
        training_data[TARGET_COLUMN],
    )

    current_probability = fallback_holdout[
        "elo_home_win_probability"
    ].to_numpy(dtype=float)

    external_probability = (
        external_model.predict_proba(
            fallback_holdout.loc[
                :,
                FALLBACK_FEATURES,
            ]
        )[:, 1]
    )

    actual_values = fallback_holdout[
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
                    CURRENT_FALLBACK_CANDIDATE
                ),
                "training_game_count": len(
                    training_data
                ),
                "fallback_holdout_game_count": (
                    len(fallback_holdout)
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
                    EXTERNAL_LOGISTIC_CANDIDATE
                ),
                "training_game_count": len(
                    training_data
                ),
                "fallback_holdout_game_count": (
                    len(fallback_holdout)
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
        current_probability
        - actual_array
    )

    external_brier_loss = np.square(
        external_probability
        - actual_array
    )

    predictions = pd.DataFrame(
        {
            "game_id": fallback_holdout[
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

    paired_summary = pd.DataFrame(
        [
            create_layer_paired_summary(
                model_layer=(
                    "PROBABILITY_FALLBACK"
                ),
                loss_metric="BRIER_SCORE",
                current_losses=predictions[
                    "current_brier_loss"
                ],
                external_losses=predictions[
                    "external_brier_loss"
                ],
                bootstrap_iterations=10_000,
                random_seed=62,
            )
        ]
    )

    return (
        summary,
        paired_summary,
        predictions,
    )


def run_probability_fallback_holdout(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the locked fallback holdout evaluation."""

    validate_database_file(database_file)

    logger.info(
        "Starting locked 2025 probability fallback "
        "holdout evaluation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        source_data = (
            load_probability_fallback_holdout_data(
                connection
            )
        )

    results = (
        evaluate_locked_probability_fallback_holdout(
            source_data
        )
    )

    logger.info(
        "Locked 2025 probability fallback holdout "
        "evaluation completed."
    )

    return results


def main() -> None:
    """Run and print fallback holdout results."""

    (
        summary,
        paired_summary,
        _,
    ) = run_probability_fallback_holdout()

    print(
        "\nFINAL PROBABILITY FALLBACK HOLDOUT\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPAIRED PROBABILITY FALLBACK HOLDOUT\n"
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()