"""
NFL Analytics Platform
XGBoost Time-Series CV

Purpose:
    Evaluate XGBoost home-win probability models
    using leakage-safe expanding-season folds.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.run_hist_gradient_boosting_time_cv import (
    aggregate_boosting_time_cv_results,
)
from src.modeling.time_series_validation import (
    create_expanding_season_folds,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    evaluate_validation_models,
    load_development_data,
    validate_database_file,
    validate_source_tables,
)
from src.modeling.train_xgboost import (
    XGBOOST_CONFIGURATIONS,
    XGBOOST_FEATURE_GROUPS,
    train_xgboost_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_xgboost_time_cv(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every XGBoost configuration and fold."""

    folds = create_expanding_season_folds(
        development_data
    )

    result_rows: list[dict[str, object]] = []

    for (
        configuration_name,
        config,
    ) in XGBOOST_CONFIGURATIONS.items():
        for (
            model_name,
            feature_columns,
        ) in XGBOOST_FEATURE_GROUPS.items():
            for fold in folds:
                logger.info(
                    "XGBoost time-CV model: %s | "
                    "configuration=%s | "
                    "validation season=%s",
                    model_name,
                    configuration_name,
                    fold.validation_season,
                )

                model = train_xgboost_model(
                    development_data=(
                        fold.development_data
                    ),
                    feature_columns=feature_columns,
                    config=config,
                )

                evaluations = evaluate_validation_models(
                    model=model,
                    development_data=(
                        fold.development_data
                    ),
                    feature_columns=feature_columns,
                )

                xgboost_evaluation = evaluations[
                    "logistic"
                ]

                elo_evaluation = evaluations["elo"]

                result_rows.append(
                    {
                        "model_name": model_name,
                        "configuration_name": (
                            configuration_name
                        ),
                        "feature_count": len(
                            feature_columns
                        ),
                        "validation_season": (
                            fold.validation_season
                        ),
                        "game_count": (
                            xgboost_evaluation.game_count
                        ),
                        "accuracy": (
                            xgboost_evaluation.accuracy
                        ),
                        "brier_score": (
                            xgboost_evaluation.brier_score
                        ),
                        "log_loss": (
                            xgboost_evaluation.log_loss
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


def log_xgboost_results(
    fold_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
) -> None:
    """Log aggregate and season-level XGBoost results."""

    logger.info(
        "XGBoost time-CV results:"
    )

    for row in aggregate_results.itertuples(index=False):
        logger.info(
            "%s | Configuration=%s | Features=%s | "
            "Folds=%s | Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f | "
            "Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.model_name,
            row.configuration_name,
            row.feature_count,
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
        "Best XGBoost model: %s | Configuration=%s",
        best_result["model_name"],
        best_result["configuration_name"],
    )

    best_fold_results = fold_results.loc[
        (
            fold_results["model_name"]
            == best_result["model_name"]
        )
        & (
            fold_results["configuration_name"]
            == best_result["configuration_name"]
        )
    ].sort_values(
        by="validation_season"
    )

    for row in best_fold_results.itertuples(index=False):
        logger.info(
            "Season=%s | Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f | "
            "Elo Brier=%.6f | Elo log loss=%.6f",
            row.validation_season,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.elo_brier_score,
            row.elo_log_loss,
        )


def run_xgboost_time_cv_experiment(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load development data and evaluate XGBoost."""

    validate_database_file(database_file)

    logger.info(
        "Starting XGBoost time-CV experiment..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    fold_results = run_xgboost_time_cv(
        development_data
    )

    aggregate_results = (
        aggregate_boosting_time_cv_results(
            fold_results
        )
    )

    log_xgboost_results(
        fold_results=fold_results,
        aggregate_results=aggregate_results,
    )

    logger.info(
        "XGBoost time-CV experiment "
        "completed successfully."
    )

    return aggregate_results


def main() -> None:
    """Run the XGBoost time-CV experiment."""

    try:
        run_xgboost_time_cv_experiment()

    except Exception:
        logger.exception(
            "XGBoost time-CV experiment failed."
        )
        raise


if __name__ == "__main__":
    main()