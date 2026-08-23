"""
NFL Analytics Platform
Logistic Regression Regularization Tuning

Purpose:
    Tune logistic-regression regularization strength
    on validation data without loading the holdout split.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.run_logistic_ablation import (
    FEATURE_GROUPS,
    evaluation_to_row,
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


REGULARIZATION_GRID = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
)


def validate_regularization_grid(
    regularization_grid: tuple[float, ...],
) -> None:
    """Validate candidate logistic C values."""

    if not regularization_grid:
        raise ValueError(
            "Regularization grid must not be empty."
        )

    if len(regularization_grid) != len(
        set(regularization_grid)
    ):
        raise ValueError(
            "Regularization grid contains duplicates."
        )

    if any(
        regularization_c <= 0.0
        for regularization_c in regularization_grid
    ):
        raise ValueError(
            "Every regularization C must be greater than zero."
        )


def run_regularization_tuning(
    development_data: pd.DataFrame,
    regularization_grid: tuple[
        float, ...
    ] = REGULARIZATION_GRID,
) -> pd.DataFrame:
    """Evaluate every feature-group and C combination."""

    validate_regularization_grid(
        regularization_grid
    )

    result_rows: list[dict[str, object]] = []

    for model_name, feature_columns in FEATURE_GROUPS.items():
        for regularization_c in regularization_grid:
            logger.info(
                "Tuning model: %s | Features=%s | C=%s",
                model_name,
                len(feature_columns),
                regularization_c,
            )

            model = train_logistic_model(
                development_data=development_data,
                feature_columns=feature_columns,
                regularization_c=regularization_c,
            )

            evaluations = evaluate_validation_models(
                model=model,
                development_data=development_data,
                feature_columns=feature_columns,
            )

            result_rows.append(
                evaluation_to_row(
                    model_name=model_name,
                    feature_columns=feature_columns,
                    evaluation=evaluations["logistic"],
                    elo_evaluation=evaluations["elo"],
                    regularization_c=regularization_c,
                )
            )

    results = pd.DataFrame(result_rows)

    return results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "feature_count",
            "regularization_c",
        ],
        ascending=True,
    ).reset_index(drop=True)


def select_best_per_feature_group(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Select the best C for every feature group."""

    required_columns = {
        "model_name",
        "brier_score",
        "log_loss",
    }

    missing_columns = sorted(
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Tuning results are missing columns: "
            + ", ".join(missing_columns)
        )

    if results.empty:
        raise ValueError(
            "Tuning results must not be empty."
        )

    ordered_results = results.sort_values(
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


def log_tuning_results(
    results: pd.DataFrame,
) -> None:
    """Log the best regularization for each feature group."""

    best_results = select_best_per_feature_group(
        results
    )

    logger.info(
        "Best regularization result per feature group:"
    )

    for row in best_results.itertuples(index=False):
        logger.info(
            "%s | Features=%s | C=%s | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f | "
            "Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.model_name,
            row.feature_count,
            row.regularization_c,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.brier_improvement_vs_elo,
            row.log_loss_improvement_vs_elo,
        )

    best_result = results.iloc[0]

    logger.info(
        "Best overall validation model: %s | C=%s",
        best_result["model_name"],
        best_result["regularization_c"],
    )


def tune_logistic_regularization(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load development data and tune regularization."""

    validate_database_file(database_file)

    logger.info(
        "Starting logistic regularization tuning..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    results = run_regularization_tuning(
        development_data=development_data,
    )

    log_tuning_results(results)

    logger.info(
        "Logistic regularization tuning completed successfully."
    )

    return results


def main() -> None:
    """Run logistic regularization tuning."""

    try:
        tune_logistic_regularization()

    except Exception:
        logger.exception(
            "Logistic regularization tuning failed."
        )
        raise


if __name__ == "__main__":
    main()