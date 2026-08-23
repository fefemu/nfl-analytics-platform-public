"""
NFL Analytics Platform
Candidate Model Evaluation

Purpose:
    Compare selected probability models on the external
    validation period without loading the holdout split.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.run_logistic_ablation import (
    ELO_QB_FEATURES,
)
from src.modeling.selected_model import (
    SELECTED_MODEL,
)
from src.modeling.train_hist_gradient_boosting import (
    VERY_CONSERVATIVE_CONFIG,
    train_hist_gradient_boosting_model,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    MODEL_FEATURE_COLUMNS,
    evaluate_validation_models,
    load_development_data,
    train_logistic_model,
    validate_database_file,
    validate_source_tables,
)
from src.modeling.train_xgboost import (
    MODERATE_XGBOOST_CONFIG,
    train_xgboost_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


CandidateModel = tuple[
    Pipeline,
    tuple[str, ...],
]


def train_candidate_models(
    development_data: pd.DataFrame,
) -> dict[str, CandidateModel]:
    """Train every selected development candidate."""

    return {
        "logistic_full_core": (
            train_logistic_model(
                development_data=development_data,
                feature_columns=MODEL_FEATURE_COLUMNS,
                regularization_c=0.01,
            ),
            MODEL_FEATURE_COLUMNS,
        ),
        "logistic_elo_plus_qb": (
            train_logistic_model(
                development_data=development_data,
                feature_columns=ELO_QB_FEATURES,
                regularization_c=1.0,
            ),
            ELO_QB_FEATURES,
        ),
        SELECTED_MODEL.model_name: (
            train_logistic_model(
                development_data=development_data,
                feature_columns=(
                    SELECTED_MODEL.feature_columns
                ),
                regularization_c=(
                    SELECTED_MODEL.regularization_c
                ),
            ),
            SELECTED_MODEL.feature_columns,
        ),
        "hist_gradient_boosting_full_core": (
            train_hist_gradient_boosting_model(
                development_data=development_data,
                feature_columns=MODEL_FEATURE_COLUMNS,
                config=VERY_CONSERVATIVE_CONFIG,
            ),
            MODEL_FEATURE_COLUMNS,
        ),
        "xgboost_full_core": (
            train_xgboost_model(
                development_data=development_data,
                feature_columns=MODEL_FEATURE_COLUMNS,
                config=MODERATE_XGBOOST_CONFIG,
            ),
            MODEL_FEATURE_COLUMNS,
        ),
    }


def evaluate_model_candidates(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate selected candidates on external validation."""

    candidate_models = train_candidate_models(
        development_data
    )

    result_rows: list[dict[str, object]] = []
    elo_evaluation = None

    for (
        model_name,
        (
            model,
            feature_columns,
        ),
    ) in candidate_models.items():
        evaluations = evaluate_validation_models(
            model=model,
            development_data=development_data,
            feature_columns=feature_columns,
        )

        model_evaluation = evaluations[
            "logistic"
        ]

        if elo_evaluation is None:
            elo_evaluation = evaluations["elo"]

        result_rows.append(
            {
                "model_name": model_name,
                "game_count": (
                    model_evaluation.game_count
                ),
                "accuracy": (
                    model_evaluation.accuracy
                ),
                "brier_score": (
                    model_evaluation.brier_score
                ),
                "log_loss": (
                    model_evaluation.log_loss
                ),
                "brier_improvement_vs_elo": (
                    elo_evaluation.brier_score
                    - model_evaluation.brier_score
                ),
                "log_loss_improvement_vs_elo": (
                    elo_evaluation.log_loss
                    - model_evaluation.log_loss
                ),
            }
        )

    if elo_evaluation is None:
        raise RuntimeError(
            "No candidate evaluation produced an Elo result."
        )

    result_rows.append(
        {
            "model_name": "elo",
            "game_count": elo_evaluation.game_count,
            "accuracy": elo_evaluation.accuracy,
            "brier_score": elo_evaluation.brier_score,
            "log_loss": elo_evaluation.log_loss,
            "brier_improvement_vs_elo": 0.0,
            "log_loss_improvement_vs_elo": 0.0,
        }
    )

    results = pd.DataFrame(result_rows)

    return results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "model_name",
        ],
        ascending=True,
    ).reset_index(drop=True)


def log_candidate_results(
    results: pd.DataFrame,
) -> None:
    """Log the external validation comparison."""

    logger.info(
        "External validation candidate results:"
    )

    for row in results.itertuples(index=False):
        logger.info(
            "%s | Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f | "
            "Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.model_name,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.brier_improvement_vs_elo,
            row.log_loss_improvement_vs_elo,
        )

    best_result = results.iloc[0]

    logger.info(
        "Best external validation candidate: %s",
        best_result["model_name"],
    )


def run_candidate_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load development data and evaluate candidates."""

    validate_database_file(database_file)

    logger.info(
        "Starting external validation candidate evaluation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    results = evaluate_model_candidates(
        development_data
    )

    log_candidate_results(results)

    logger.info(
        "External validation candidate evaluation "
        "completed successfully."
    )

    return results


def main() -> None:
    """Run external validation candidate evaluation."""

    try:
        run_candidate_evaluation()

    except Exception:
        logger.exception(
            "External validation candidate evaluation failed."
        )
        raise


if __name__ == "__main__":
    main()