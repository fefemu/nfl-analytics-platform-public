"""
NFL Analytics Platform
Logistic Regression Time-Series CV

Purpose:
    Compare logistic feature groups and regularization
    using leakage-safe expanding-season folds.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.run_logistic_ablation import (
    FEATURE_GROUPS,
)
from src.modeling.time_series_validation import (
    create_expanding_season_folds,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    evaluate_validation_models,
    load_development_data,
    train_logistic_model,
    validate_database_file,
    validate_source_tables,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


CV_REGULARIZATION_GRID = (
    0.01,
    0.1,
    1.0,
    10.0,
)


def validate_cv_configuration(
    feature_groups: Mapping[
        str,
        tuple[str, ...],
    ],
    regularization_grid: tuple[
        float, ...
    ],
) -> None:
    """Validate the time-CV experiment configuration."""

    if not feature_groups:
        raise ValueError(
            "Time-CV requires at least one feature group."
        )

    if any(
        not model_name.strip()
        for model_name in feature_groups
    ):
        raise ValueError(
            "Time-CV model names must not be empty."
        )

    if any(
        not feature_columns
        for feature_columns in feature_groups.values()
    ):
        raise ValueError(
            "Time-CV feature groups must not be empty."
        )

    if not regularization_grid:
        raise ValueError(
            "Time-CV regularization grid must not be empty."
        )

    if len(regularization_grid) != len(
        set(regularization_grid)
    ):
        raise ValueError(
            "Time-CV regularization values must be unique."
        )

    if any(
        regularization_c <= 0.0
        for regularization_c in regularization_grid
    ):
        raise ValueError(
            "Every time-CV regularization C must be positive."
        )


def run_logistic_time_cv(
    development_data: pd.DataFrame,
    feature_groups: Mapping[
        str,
        tuple[str, ...],
    ] = FEATURE_GROUPS,
    regularization_grid: tuple[
        float, ...
    ] = CV_REGULARIZATION_GRID,
) -> pd.DataFrame:
    """Evaluate every model configuration on every fold."""

    validate_cv_configuration(
        feature_groups=feature_groups,
        regularization_grid=regularization_grid,
    )

    folds = create_expanding_season_folds(
        development_data
    )

    result_rows: list[dict[str, object]] = []

    for model_name, feature_columns in feature_groups.items():
        for regularization_c in regularization_grid:
            for fold in folds:
                logger.info(
                    "Time-CV model: %s | C=%s | "
                    "validation season=%s",
                    model_name,
                    regularization_c,
                    fold.validation_season,
                )

                model = train_logistic_model(
                    development_data=fold.development_data,
                    feature_columns=feature_columns,
                    regularization_c=regularization_c,
                )

                evaluations = evaluate_validation_models(
                    model=model,
                    development_data=fold.development_data,
                    feature_columns=feature_columns,
                )

                logistic_evaluation = evaluations[
                    "logistic"
                ]

                elo_evaluation = evaluations["elo"]

                result_rows.append(
                    {
                        "model_name": model_name,
                        "feature_count": len(
                            feature_columns
                        ),
                        "regularization_c": (
                            regularization_c
                        ),
                        "validation_season": (
                            fold.validation_season
                        ),
                        "game_count": (
                            logistic_evaluation.game_count
                        ),
                        "accuracy": (
                            logistic_evaluation.accuracy
                        ),
                        "brier_score": (
                            logistic_evaluation.brier_score
                        ),
                        "log_loss": (
                            logistic_evaluation.log_loss
                        ),
                        "elo_accuracy": (
                            elo_evaluation.accuracy
                        ),
                        "elo_brier_score": (
                            elo_evaluation.brier_score
                        ),
                        "elo_log_loss": (
                            elo_evaluation.log_loss
                        ),
                    }
                )

    return pd.DataFrame(result_rows)


def aggregate_time_cv_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate fold metrics using validation-game weights."""

    required_columns = {
        "model_name",
        "feature_count",
        "regularization_c",
        "validation_season",
        "game_count",
        "accuracy",
        "brier_score",
        "log_loss",
        "elo_accuracy",
        "elo_brier_score",
        "elo_log_loss",
    }

    missing_columns = sorted(
        required_columns - set(fold_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold results are missing columns: "
            + ", ".join(missing_columns)
        )

    if fold_results.empty:
        raise ValueError(
            "Fold results must not be empty."
        )

    aggregate_rows: list[
        dict[str, object]
    ] = []

    grouped_results = fold_results.groupby(
        [
            "model_name",
            "feature_count",
            "regularization_c",
        ],
        sort=False,
    )

    for (
        model_name,
        feature_count,
        regularization_c,
    ), group in grouped_results:
        weights = group[
            "game_count"
        ].to_numpy(dtype=float)

        total_game_count = int(
            weights.sum()
        )

        accuracy = float(
            np.average(
                group["accuracy"],
                weights=weights,
            )
        )

        brier_score = float(
            np.average(
                group["brier_score"],
                weights=weights,
            )
        )

        model_log_loss = float(
            np.average(
                group["log_loss"],
                weights=weights,
            )
        )

        elo_accuracy = float(
            np.average(
                group["elo_accuracy"],
                weights=weights,
            )
        )

        elo_brier_score = float(
            np.average(
                group["elo_brier_score"],
                weights=weights,
            )
        )

        elo_log_loss = float(
            np.average(
                group["elo_log_loss"],
                weights=weights,
            )
        )

        aggregate_rows.append(
            {
                "model_name": model_name,
                "feature_count": feature_count,
                "regularization_c": regularization_c,
                "fold_count": group[
                    "validation_season"
                ].nunique(),
                "game_count": total_game_count,
                "accuracy": accuracy,
                "brier_score": brier_score,
                "log_loss": model_log_loss,
                "elo_accuracy": elo_accuracy,
                "elo_brier_score": elo_brier_score,
                "elo_log_loss": elo_log_loss,
                "brier_improvement_vs_elo": (
                    elo_brier_score - brier_score
                ),
                "log_loss_improvement_vs_elo": (
                    elo_log_loss - model_log_loss
                ),
            }
        )

    aggregate_results = pd.DataFrame(
        aggregate_rows
    )

    return aggregate_results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "feature_count",
            "regularization_c",
        ],
        ascending=True,
    ).reset_index(drop=True)


def select_best_per_feature_group(
    aggregate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Select the best C for each feature group."""

    if aggregate_results.empty:
        raise ValueError(
            "Aggregate results must not be empty."
        )

    ordered_results = aggregate_results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "feature_count",
            "regularization_c",
        ],
        ascending=True,
    )

    return (
        ordered_results
        .drop_duplicates(
            subset=["model_name"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def select_best_model_fold_results(
    fold_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Select season rows for the best aggregate model."""

    if fold_results.empty:
        raise ValueError(
            "Fold results must not be empty."
        )

    if aggregate_results.empty:
        raise ValueError(
            "Aggregate results must not be empty."
        )

    best_result = aggregate_results.iloc[0]

    selected_results = fold_results.loc[
        (
            fold_results["model_name"]
            == best_result["model_name"]
        )
        & np.isclose(
            fold_results["regularization_c"],
            best_result["regularization_c"],
        )
    ].copy()

    if selected_results.empty:
        raise RuntimeError(
            "No fold results match the best aggregate model."
        )

    return selected_results.sort_values(
        by="validation_season"
    ).reset_index(drop=True)


def log_best_model_fold_results(
    fold_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
) -> None:
    """Log season-level results for the best model."""

    selected_results = (
        select_best_model_fold_results(
            fold_results=fold_results,
            aggregate_results=aggregate_results,
        )
    )

    best_result = aggregate_results.iloc[0]

    logger.info(
        "Season results for best model: %s | C=%s",
        best_result["model_name"],
        best_result["regularization_c"],
    )

    for row in selected_results.itertuples(index=False):
        logger.info(
            "Season=%s | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f | "
            "Elo accuracy=%.2f%% | "
            "Elo Brier=%.6f | "
            "Elo log loss=%.6f",
            row.validation_season,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            100.0 * row.elo_accuracy,
            row.elo_brier_score,
            row.elo_log_loss,
        )


def log_time_cv_results(
    aggregate_results: pd.DataFrame,
) -> None:
    """Log the best time-CV result per feature group."""

    best_results = select_best_per_feature_group(
        aggregate_results
    )

    logger.info(
        "Best expanding-window result per feature group:"
    )

    for row in best_results.itertuples(index=False):
        logger.info(
            "%s | Features=%s | C=%s | Folds=%s | "
            "Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f | "
            "Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.model_name,
            row.feature_count,
            row.regularization_c,
            row.fold_count,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.brier_improvement_vs_elo,
            row.log_loss_improvement_vs_elo,
        )

    best_result = aggregate_results.iloc[0]

    logger.info(
        "Best overall expanding-window model: %s | C=%s",
        best_result["model_name"],
        best_result["regularization_c"],
    )


def run_time_cv_experiment(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load development data and run time-series CV."""

    validate_database_file(database_file)

    logger.info(
        "Starting logistic expanding-window experiment..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    fold_results = run_logistic_time_cv(
        development_data=development_data,
    )

    aggregate_results = aggregate_time_cv_results(
        fold_results
    )

    log_time_cv_results(
        aggregate_results
    )

    log_best_model_fold_results(
        fold_results=fold_results,
        aggregate_results=aggregate_results,
    )

    logger.info(
        "Logistic expanding-window experiment "
        "completed successfully."
    )

    return aggregate_results


def main() -> None:
    """Run the logistic time-series CV experiment."""

    try:
        run_time_cv_experiment()

    except Exception:
        logger.exception(
            "Logistic expanding-window experiment failed."
        )
        raise


if __name__ == "__main__":
    main()