"""
NFL Analytics Platform
Logistic Injury Feature Time-Series CV

Purpose:
    Evaluate whether leakage-safe non-QB injury features
    improve Elo plus QB predictions across expanding
    development-season folds.

Coverage policy:
    Compare every candidate on the same games with
    complete home and away injury-report coverage.

Holdout policy:
    Load train and validation splits only. The 2025
    holdout is never loaded by this experiment.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.run_logistic_time_cv import (
    CV_REGULARIZATION_GRID,
    aggregate_time_cv_results,
    log_best_model_fold_results,
    log_time_cv_results,
    run_logistic_time_cv,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    DATASET_FULL_NAME,
    DATASET_SCHEMA,
    DATASET_TABLE,
    HOLDOUT_SPLIT,
    SPLIT_FULL_NAME,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    validate_database_file,
    validate_source_tables,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


INJURY_AVAILABILITY_COLUMN = (
    "has_complete_injury_data"
)

BASE_FEATURES = (
    "elo_rating_difference",
    "listed_qb_rating_difference",
)

NON_QB_BURDEN_FEATURES = (
    "non_qb_injury_burden_difference",
)

UNIT_BURDEN_FEATURES = (
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
)

INJURY_COUNT_FEATURES = (
    "out_player_count_difference",
    "doubtful_player_count_difference",
    "questionable_player_count_difference",
    "starter_out_count_difference",
)

INJURY_FEATURE_GROUPS = {
    "elo_plus_qb": BASE_FEATURES,
    "elo_qb_non_qb_burden": (
        *BASE_FEATURES,
        *NON_QB_BURDEN_FEATURES,
    ),
    "elo_qb_unit_burdens": (
        *BASE_FEATURES,
        *UNIT_BURDEN_FEATURES,
    ),
    "elo_qb_injury_counts": (
        *BASE_FEATURES,
        *INJURY_COUNT_FEATURES,
    ),
    "elo_qb_full_injury": (
        *BASE_FEATURES,
        *UNIT_BURDEN_FEATURES,
        *INJURY_COUNT_FEATURES,
    ),
}

INJURY_DEVELOPMENT_COLUMNS = tuple(
    dict.fromkeys(
        (
            *BASE_FEATURES,
            *NON_QB_BURDEN_FEATURES,
            *UNIT_BURDEN_FEATURES,
            *INJURY_COUNT_FEATURES,
        )
    )
)

REQUIRED_INJURY_DATASET_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    TARGET_COLUMN,
    "elo_home_win_probability",
    INJURY_AVAILABILITY_COLUMN,
    *INJURY_DEVELOPMENT_COLUMNS,
}


def validate_injury_feature_groups(
    feature_groups: Mapping[
        str,
        tuple[str, ...],
    ] = INJURY_FEATURE_GROUPS,
) -> None:
    """Validate injury experiment feature groups."""

    if not feature_groups:
        raise ValueError(
            "Injury feature groups must not be empty."
        )

    known_features = set(
        INJURY_DEVELOPMENT_COLUMNS
    )

    for model_name, feature_columns in feature_groups.items():
        if not model_name.strip():
            raise ValueError(
                "Injury model names must not be empty."
            )

        if not feature_columns:
            raise ValueError(
                f"Injury feature group is empty: {model_name}"
            )

        if len(feature_columns) != len(
            set(feature_columns)
        ):
            raise ValueError(
                "Injury feature group contains duplicates: "
                f"{model_name}"
            )

        unknown_features = sorted(
            set(feature_columns)
            - known_features
        )

        if unknown_features:
            raise ValueError(
                f"Unknown injury features in {model_name}: "
                + ", ".join(
                    unknown_features
                )
            )


def validate_injury_dataset_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate injury columns in the modeling dataset."""

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [
                DATASET_SCHEMA,
                DATASET_TABLE,
            ],
        ).fetchall()
    }

    missing_columns = sorted(
        REQUIRED_INJURY_DATASET_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Modeling dataset is missing injury columns: "
            + ", ".join(
                missing_columns
            )
        )


def load_injury_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load complete-coverage development games only."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name in INJURY_DEVELOPMENT_COLUMNS
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

            dataset.elo_home_win_probability,
            dataset.{INJURY_AVAILABILITY_COLUMN},

            {feature_select}

        FROM {DATASET_FULL_NAME} AS dataset

        INNER JOIN {SPLIT_FULL_NAME} AS splits
            ON dataset.game_id = splits.game_id

        WHERE splits.split_name IN (
                '{TRAIN_SPLIT}',
                '{VALIDATION_SPLIT}'
              )

          AND splits.split_name
                != '{HOLDOUT_SPLIT}'

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
            "The injury experiment has no development games."
        )

    loaded_splits = set(
        data["split_name"].unique()
    )

    required_splits = {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    }

    missing_splits = sorted(
        required_splits
        - loaded_splits
    )

    if missing_splits:
        raise RuntimeError(
            "Injury development data is missing splits: "
            + ", ".join(
                missing_splits
            )
        )

    if HOLDOUT_SPLIT in loaded_splits:
        raise RuntimeError(
            "The injury experiment loaded holdout games."
        )

    if not data[
        INJURY_AVAILABILITY_COLUMN
    ].all():
        raise RuntimeError(
            "Injury development data contains "
            "incomplete coverage."
        )

    logger.info(
        "Injury development data loaded: "
        "%s train games and %s validation games.",
        int(
            (
                data["split_name"]
                == TRAIN_SPLIT
            ).sum()
        ),
        int(
            (
                data["split_name"]
                == VALIDATION_SPLIT
            ).sum()
        ),
    )

    return data


def run_injury_time_cv_experiment(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run expanding-window injury feature evaluation."""

    validate_database_file(
        database_file
    )

    validate_injury_feature_groups()

    logger.info(
        "Starting logistic injury-feature time-CV..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(
            connection
        )

        validate_injury_dataset_columns(
            connection
        )

        development_data = (
            load_injury_development_data(
                connection
            )
        )

    fold_results = run_logistic_time_cv(
        development_data=development_data,
        feature_groups=INJURY_FEATURE_GROUPS,
        regularization_grid=(
            CV_REGULARIZATION_GRID
        ),
    )

    aggregate_results = (
        aggregate_time_cv_results(
            fold_results
        )
    )

    log_time_cv_results(
        aggregate_results
    )

    log_best_model_fold_results(
        fold_results=fold_results,
        aggregate_results=aggregate_results,
    )

    logger.info(
        "Logistic injury-feature time-CV "
        "completed successfully."
    )

    return aggregate_results


def main() -> None:
    """Run the injury feature time-CV experiment."""

    try:
        run_injury_time_cv_experiment()

    except Exception:
        logger.exception(
            "Logistic injury-feature time-CV failed."
        )
        raise


if __name__ == "__main__":
    main()