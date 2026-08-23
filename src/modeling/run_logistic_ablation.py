"""
NFL Analytics Platform
Logistic Regression Feature Ablation

Purpose:
    Compare nested logistic-regression feature groups
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

from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    MODEL_FEATURE_COLUMNS,
    ModelEvaluation,
    evaluate_validation_models,
    load_development_data,
    train_logistic_model,
    validate_database_file,
    validate_source_tables,
    DEVELOPMENT_FEATURE_COLUMNS,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


ELO_FEATURES = (
    "elo_rating_difference",
)

ELO_QB_FEATURES = (
    *ELO_FEATURES,
    "listed_qb_rating_difference",
)

ELO_QB_SCHEDULE_FEATURES = (
    *ELO_QB_FEATURES,
    *SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)

ELO_QB_REST_DIFFERENCE_FEATURES = (
    *ELO_QB_FEATURES,
    "rest_days_difference",
)

ELO_QB_SHORT_WEEK_FEATURES = (
    *ELO_QB_FEATURES,
    "short_week_difference",
)

ELO_QB_EXTENDED_REST_FEATURES = (
    *ELO_QB_FEATURES,
    "extended_rest_difference",
)

ELO_QB_POST_BYE_FEATURES = (
    *ELO_QB_FEATURES,
    "post_bye_difference",
)

STABLE_EPA_FEATURES = (
    *ELO_FEATURES,
    "offensive_epa_per_play_difference_last_4",
    "success_rate_difference_last_4",
    "defensive_epa_allowed_per_play_difference_last_4",
    "defensive_success_rate_allowed_difference_last_4",
)

FEATURE_GROUPS = {
    "elo_only": ELO_FEATURES,
    "elo_plus_qb": ELO_QB_FEATURES,
    "elo_qb_rest_difference": (
        ELO_QB_REST_DIFFERENCE_FEATURES
    ),
    "elo_qb_short_week": (
        ELO_QB_SHORT_WEEK_FEATURES
    ),
    "elo_qb_extended_rest": (
        ELO_QB_EXTENDED_REST_FEATURES
    ),
    "elo_qb_post_bye": (
        ELO_QB_POST_BYE_FEATURES
    ),
    "elo_qb_schedule": ELO_QB_SCHEDULE_FEATURES,
    "elo_plus_stable_epa": STABLE_EPA_FEATURES,
    "full_core": MODEL_FEATURE_COLUMNS,
}

DEFAULT_REGULARIZATION_C = 1.0


def validate_feature_groups() -> None:
    """Validate ablation feature-group definitions."""

    known_features = set(
        DEVELOPMENT_FEATURE_COLUMNS
    )

    for model_name, feature_columns in FEATURE_GROUPS.items():
        if not feature_columns:
            raise RuntimeError(
                f"Feature group is empty: {model_name}"
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise RuntimeError(
                "Feature group contains duplicates: "
                f"{model_name}"
            )

        unknown_features = sorted(
            set(feature_columns) - known_features
        )

        if unknown_features:
            raise RuntimeError(
                f"Unknown features in {model_name}: "
                + ", ".join(unknown_features)
            )


def evaluation_to_row(
    model_name: str,
    feature_columns: tuple[str, ...],
    evaluation: ModelEvaluation,
    elo_evaluation: ModelEvaluation,
    regularization_c: float,
) -> dict[str, object]:
    """Convert one validation result to a table row."""

    return {
        "model_name": model_name,
        "feature_count": len(feature_columns),
        "regularization_c": regularization_c,
        "game_count": evaluation.game_count,
        "accuracy": evaluation.accuracy,
        "brier_score": evaluation.brier_score,
        "log_loss": evaluation.log_loss,
        "brier_improvement_vs_elo": (
            elo_evaluation.brier_score
            - evaluation.brier_score
        ),
        "log_loss_improvement_vs_elo": (
            elo_evaluation.log_loss
            - evaluation.log_loss
        ),
    }


def run_feature_ablation(
    development_data: pd.DataFrame,
    regularization_c: float = DEFAULT_REGULARIZATION_C,
) -> pd.DataFrame:
    """Train and compare all configured feature groups."""

    validate_feature_groups()

    result_rows: list[dict[str, object]] = []

    for model_name, feature_columns in FEATURE_GROUPS.items():
        logger.info(
            "Training ablation model: %s | "
            "features=%s | C=%s",
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

        logistic_evaluation = evaluations["logistic"]
        elo_evaluation = evaluations["elo"]

        result_rows.append(
            evaluation_to_row(
                model_name=model_name,
                feature_columns=feature_columns,
                evaluation=logistic_evaluation,
                elo_evaluation=elo_evaluation,
                regularization_c=regularization_c,
            )
        )

    results = pd.DataFrame(result_rows)

    results = results.sort_values(
        by=[
            "brier_score",
            "log_loss",
        ],
        ascending=True,
    ).reset_index(drop=True)

    return results


def log_ablation_results(
    results: pd.DataFrame,
) -> None:
    """Log validation results ordered by probability quality."""

    logger.info(
        "Feature ablation validation results:"
    )

    for row in results.itertuples(index=False):
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
        "Best feature group by Brier score: %s",
        best_result["model_name"],
    )


def run_logistic_ablation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load development data and run feature ablation."""

    validate_database_file(database_file)

    logger.info(
        "Starting logistic feature ablation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    results = run_feature_ablation(
        development_data=development_data,
    )

    log_ablation_results(results)

    logger.info(
        "Logistic feature ablation completed successfully."
    )

    return results


def main() -> None:
    """Run logistic feature ablation."""

    try:
        run_logistic_ablation()

    except Exception:
        logger.exception(
            "Logistic feature ablation failed."
        )
        raise


if __name__ == "__main__":
    main()