"""
NFL Analytics Platform
Expanding-Window Model Governance Scorecard

Purpose:
    Compare frozen champion and challenger models on
    identical leakage-safe games across 2020-2025.

Evaluation policy:
    Every season is predicted using models trained only
    on strictly earlier seasons.

Coverage policy:
    Use core-eligible games with complete two-sided
    injury-report coverage so every candidate receives
    the same evaluation sample.

Holdout policy:
    The former 2025 holdout is now included only as a
    historical governance audit. The next untouched
    forward-test season is 2026.

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

from src.modeling.model_governance_candidates import (
    GOVERNANCE_CANDIDATES,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    DATASET_FULL_NAME,
    DATASET_SCHEMA,
    DATASET_TABLE,
    SPLIT_FULL_NAME,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    evaluate_probabilities,
    train_logistic_model,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


GOVERNANCE_VALIDATION_SEASONS = (
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
)

GOVERNANCE_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        feature_name
        for candidate in GOVERNANCE_CANDIDATES
        for feature_name in candidate.feature_columns
    )
)

REQUIRED_GOVERNANCE_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    TARGET_COLUMN,
    "elo_home_win_probability",
    "has_complete_injury_data",
    *GOVERNANCE_FEATURE_COLUMNS,
}


def validate_governance_dataset_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate scorecard columns in the modeling dataset."""

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
        REQUIRED_GOVERNANCE_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Modeling dataset is missing governance columns: "
            + ", ".join(
                missing_columns
            )
        )


def load_governance_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load the complete historical governance sample."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name in GOVERNANCE_FEATURE_COLUMNS
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
            dataset.has_complete_injury_data,

            {feature_select}

        FROM {DATASET_FULL_NAME} AS dataset

        INNER JOIN {SPLIT_FULL_NAME} AS splits
            ON dataset.game_id = splits.game_id

        WHERE splits.is_core_model_eligible = TRUE

          AND dataset.has_complete_injury_data = TRUE

          AND dataset.{TARGET_COLUMN} IS NOT NULL

          AND dataset.season
                <= {
                    max(
                        GOVERNANCE_VALIDATION_SEASONS
                    )
                }

        ORDER BY
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "The model governance dataset is empty."
        )

    if data[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Model governance data contains missing targets."
        )

    missing_validation_seasons = sorted(
        set(GOVERNANCE_VALIDATION_SEASONS)
        - set(data["season"].unique())
    )

    if missing_validation_seasons:
        raise RuntimeError(
            "Model governance data is missing seasons: "
            + ", ".join(
                str(season)
                for season in missing_validation_seasons
            )
        )

    logger.info(
        "Model governance data loaded: %s games "
        "across seasons %s-%s.",
        len(data),
        int(data["season"].min()),
        int(data["season"].max()),
    )

    return data


def create_governance_fold(
    governance_data: pd.DataFrame,
    validation_season: int,
) -> pd.DataFrame:
    """Create one strictly chronological season fold."""

    fold_data = governance_data.loc[
        governance_data["season"]
        <= validation_season
    ].copy()

    fold_data["split_name"] = TRAIN_SPLIT

    fold_data.loc[
        fold_data["season"]
        == validation_season,
        "split_name",
    ] = VALIDATION_SPLIT

    training_data = fold_data.loc[
        fold_data["split_name"]
        == TRAIN_SPLIT
    ]

    validation_data = fold_data.loc[
        fold_data["split_name"]
        == VALIDATION_SPLIT
    ]

    if training_data.empty:
        raise RuntimeError(
            "Governance fold has no training games for "
            f"season {validation_season}."
        )

    if validation_data.empty:
        raise RuntimeError(
            "Governance fold has no validation games for "
            f"season {validation_season}."
        )

    latest_training_date = training_data[
        "game_date"
    ].max()

    earliest_validation_date = validation_data[
        "game_date"
    ].min()

    if latest_training_date >= earliest_validation_date:
        raise RuntimeError(
            "Governance fold chronology is invalid for "
            f"season {validation_season}."
        )

    return fold_data.sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(
        drop=True
    )


def evaluate_governance_models(
    governance_data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all frozen models in every season."""

    result_rows: list[
        dict[str, object]
    ] = []

    for validation_season in (
        GOVERNANCE_VALIDATION_SEASONS
    ):
        fold_data = create_governance_fold(
            governance_data=governance_data,
            validation_season=validation_season,
        )

        validation_data = fold_data.loc[
            fold_data["split_name"]
            == VALIDATION_SPLIT
        ].copy()

        validation_target = validation_data[
            TARGET_COLUMN
        ]

        elo_evaluation = evaluate_probabilities(
            actual_values=validation_target,
            probabilities=validation_data[
                "elo_home_win_probability"
            ].to_numpy(
                dtype=float
            ),
        )

        result_rows.append(
            {
                "validation_season": (
                    validation_season
                ),
                "model_name": "elo",
                "model_version": "production_1.0.0",
                "feature_count": 1,
                "regularization_c": np.nan,
                "game_count": (
                    elo_evaluation.game_count
                ),
                "accuracy": (
                    elo_evaluation.accuracy
                ),
                "brier_score": (
                    elo_evaluation.brier_score
                ),
                "log_loss": (
                    elo_evaluation.log_loss
                ),
            }
        )

        for candidate in GOVERNANCE_CANDIDATES:
            model = train_logistic_model(
                development_data=fold_data,
                feature_columns=(
                    candidate.feature_columns
                ),
                regularization_c=(
                    candidate.regularization_c
                ),
            )

            probabilities = model.predict_proba(
                validation_data.loc[
                    :,
                    candidate.feature_columns,
                ]
            )[:, 1]

            evaluation = evaluate_probabilities(
                actual_values=validation_target,
                probabilities=probabilities,
            )

            result_rows.append(
                {
                    "validation_season": (
                        validation_season
                    ),
                    "model_name": (
                        candidate.model_name
                    ),
                    "model_version": (
                        candidate.model_version
                    ),
                    "feature_count": len(
                        candidate.feature_columns
                    ),
                    "regularization_c": (
                        candidate.regularization_c
                    ),
                    "game_count": (
                        evaluation.game_count
                    ),
                    "accuracy": (
                        evaluation.accuracy
                    ),
                    "brier_score": (
                        evaluation.brier_score
                    ),
                    "log_loss": (
                        evaluation.log_loss
                    ),
                }
            )

    return pd.DataFrame(
        result_rows
    ).sort_values(
        by=[
            "validation_season",
            "brier_score",
            "log_loss",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )


def aggregate_governance_results(
    season_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate weighted performance and robustness."""

    if season_results.empty:
        raise ValueError(
            "Governance season results must not be empty."
        )

    aggregate_rows: list[
        dict[str, object]
    ] = []

    grouped_results = season_results.groupby(
        [
            "model_name",
            "model_version",
            "feature_count",
        ],
        sort=False,
        dropna=False,
    )

    for (
        model_name,
        model_version,
        feature_count,
    ), group in grouped_results:
        weights = group[
            "game_count"
        ].to_numpy(
            dtype=float
        )

        aggregate_rows.append(
            {
                "model_name": model_name,
                "model_version": model_version,
                "feature_count": int(
                    feature_count
                ),
                "season_count": int(
                    group[
                        "validation_season"
                    ].nunique()
                ),
                "game_count": int(
                    weights.sum()
                ),
                "accuracy": float(
                    np.average(
                        group["accuracy"],
                        weights=weights,
                    )
                ),
                "brier_score": float(
                    np.average(
                        group["brier_score"],
                        weights=weights,
                    )
                ),
                "log_loss": float(
                    np.average(
                        group["log_loss"],
                        weights=weights,
                    )
                ),
                "worst_season_brier": float(
                    group["brier_score"].max()
                ),
                "worst_season_log_loss": float(
                    group["log_loss"].max()
                ),
                "brier_season_std": float(
                    group["brier_score"].std(
                        ddof=0
                    )
                ),
                "log_loss_season_std": float(
                    group["log_loss"].std(
                        ddof=0
                    )
                ),
            }
        )

    aggregate_results = pd.DataFrame(
        aggregate_rows
    )

    elo_result = aggregate_results.loc[
        aggregate_results["model_name"]
        == "elo"
    ]

    if len(elo_result) != 1:
        raise RuntimeError(
            "Governance scorecard requires one Elo result."
        )

    elo_brier = float(
        elo_result.iloc[0]["brier_score"]
    )

    elo_log_loss = float(
        elo_result.iloc[0]["log_loss"]
    )

    aggregate_results[
        "brier_improvement_vs_elo"
    ] = (
        elo_brier
        - aggregate_results["brier_score"]
    )

    aggregate_results[
        "log_loss_improvement_vs_elo"
    ] = (
        elo_log_loss
        - aggregate_results["log_loss"]
    )

    return aggregate_results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "worst_season_brier",
            "feature_count",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )


def log_governance_scorecard(
    season_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
) -> None:
    """Log aggregate ranking and season details."""

    logger.info(
        "2020-2025 model governance scorecard:"
    )

    for row in aggregate_results.itertuples(
        index=False
    ):
        logger.info(
            "%s | Version=%s | Features=%s | "
            "Seasons=%s | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f | Worst Brier=%.6f | "
            "Brier SD=%.6f | Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.model_name,
            row.model_version,
            row.feature_count,
            row.season_count,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.worst_season_brier,
            row.brier_season_std,
            row.brier_improvement_vs_elo,
            row.log_loss_improvement_vs_elo,
        )

    logger.info(
        "Governance scorecard season results:"
    )

    for row in season_results.itertuples(
        index=False
    ):
        logger.info(
            "Season=%s | Model=%s | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f",
            row.validation_season,
            row.model_name,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
        )

    best_result = aggregate_results.iloc[0]

    logger.info(
        "Best aggregate governance candidate: %s | "
        "Version=%s",
        best_result["model_name"],
        best_result["model_version"],
    )


def run_model_governance_scorecard(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run the full historical governance scorecard."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting 2020-2025 model governance scorecard."
    )

    logger.info(
        "The former 2025 holdout is included as a "
        "historical audit; 2026 is the next forward test."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_governance_dataset_columns(
            connection
        )

        governance_data = load_governance_data(
            connection
        )

    season_results = evaluate_governance_models(
        governance_data
    )

    aggregate_results = (
        aggregate_governance_results(
            season_results
        )
    )

    log_governance_scorecard(
        season_results=season_results,
        aggregate_results=aggregate_results,
    )

    logger.info(
        "Model governance scorecard completed successfully."
    )

    return aggregate_results


def main() -> None:
    """Run model governance scorecard."""

    try:
        run_model_governance_scorecard()

    except Exception:
        logger.exception(
            "Model governance scorecard failed."
        )
        raise


if __name__ == "__main__":
    main()