"""
NFL Analytics Platform
External Validation Injury Candidate Evaluation

Purpose:
    Compare Elo, Elo plus QB and Elo plus QB plus
    unit-level injury burdens on the untouched
    2023-2024 external validation period.

Coverage policy:
    Every candidate uses the same games with complete
    home and away injury-report data.

Holdout policy:
    The 2025 holdout is never loaded.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.run_logistic_injury_time_cv import (
    BASE_FEATURES,
    UNIT_BURDEN_FEATURES,
    load_injury_development_data,
    validate_injury_dataset_columns,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    VALIDATION_SPLIT,
    ModelEvaluation,
    evaluate_probabilities,
    train_logistic_model,
    validate_database_file,
    validate_source_tables,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InjuryCandidateConfig:
    """Describe one logistic injury candidate."""

    feature_columns: tuple[str, ...]
    regularization_c: float


INJURY_CANDIDATES = {
    "logistic_elo_plus_qb": InjuryCandidateConfig(
        feature_columns=BASE_FEATURES,
        regularization_c=1.0,
    ),
    "logistic_elo_qb_unit_burdens": (
        InjuryCandidateConfig(
            feature_columns=(
                *BASE_FEATURES,
                *UNIT_BURDEN_FEATURES,
            ),
            regularization_c=0.1,
        )
    ),
}


def evaluation_to_row(
    model_name: str,
    evaluation: ModelEvaluation,
    elo_evaluation: ModelEvaluation,
) -> dict[str, object]:
    """Convert an evaluation to one comparison row."""

    return {
        "model_name": model_name,
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


def evaluate_injury_candidates(
    development_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate overall and season-level validation results."""

    validation_data = development_data.loc[
        development_data["split_name"]
        == VALIDATION_SPLIT
    ].copy()

    if validation_data.empty:
        raise RuntimeError(
            "No injury validation games are available."
        )

    validation_target = validation_data[
        TARGET_COLUMN
    ]

    elo_probabilities = validation_data[
        "elo_home_win_probability"
    ].to_numpy(
        dtype=float
    )

    elo_evaluation = evaluate_probabilities(
        actual_values=validation_target,
        probabilities=elo_probabilities,
    )

    candidate_probabilities: dict[
        str,
        np.ndarray,
    ] = {
        "elo": elo_probabilities,
    }

    overall_rows = [
        evaluation_to_row(
            model_name="elo",
            evaluation=elo_evaluation,
            elo_evaluation=elo_evaluation,
        )
    ]

    for (
        model_name,
        config,
    ) in INJURY_CANDIDATES.items():
        model = train_logistic_model(
            development_data=development_data,
            feature_columns=config.feature_columns,
            regularization_c=config.regularization_c,
        )

        probabilities = model.predict_proba(
            validation_data.loc[
                :,
                config.feature_columns,
            ]
        )[:, 1]

        candidate_probabilities[
            model_name
        ] = probabilities

        evaluation = evaluate_probabilities(
            actual_values=validation_target,
            probabilities=probabilities,
        )

        overall_rows.append(
            evaluation_to_row(
                model_name=model_name,
                evaluation=evaluation,
                elo_evaluation=elo_evaluation,
            )
        )

    season_rows: list[
        dict[str, object]
    ] = []

    for season in sorted(
        validation_data["season"].unique()
    ):
        season_mask = (
            validation_data["season"]
            == season
        )

        season_target = validation_data.loc[
            season_mask,
            TARGET_COLUMN,
        ]

        for (
            model_name,
            probabilities,
        ) in candidate_probabilities.items():
            season_evaluation = (
                evaluate_probabilities(
                    actual_values=season_target,
                    probabilities=probabilities[
                        season_mask.to_numpy()
                    ],
                )
            )

            season_rows.append(
                {
                    "season": int(season),
                    "model_name": model_name,
                    "game_count": (
                        season_evaluation.game_count
                    ),
                    "accuracy": (
                        season_evaluation.accuracy
                    ),
                    "brier_score": (
                        season_evaluation.brier_score
                    ),
                    "log_loss": (
                        season_evaluation.log_loss
                    ),
                }
            )

    overall_results = pd.DataFrame(
        overall_rows
    ).sort_values(
        by=[
            "brier_score",
            "log_loss",
            "model_name",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )

    season_results = pd.DataFrame(
        season_rows
    ).sort_values(
        by=[
            "season",
            "brier_score",
            "log_loss",
        ],
        ascending=True,
    ).reset_index(
        drop=True
    )

    return (
        overall_results,
        season_results,
    )


def log_injury_candidate_results(
    overall_results: pd.DataFrame,
    season_results: pd.DataFrame,
) -> None:
    """Log overall and season-level comparisons."""

    logger.info(
        "External validation injury candidate results:"
    )

    for row in overall_results.itertuples(
        index=False
    ):
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

    logger.info(
        "External validation injury results by season:"
    )

    for row in season_results.itertuples(
        index=False
    ):
        logger.info(
            "Season=%s | Model=%s | Games=%s | "
            "Accuracy=%.2f%% | Brier=%.6f | "
            "Log loss=%.6f",
            row.season,
            row.model_name,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
        )

    best_result = overall_results.iloc[0]

    logger.info(
        "Best external validation injury candidate: %s",
        best_result["model_name"],
    )


def run_injury_candidate_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run the complete injury candidate evaluation."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting external validation injury "
        "candidate evaluation..."
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

    (
        overall_results,
        season_results,
    ) = evaluate_injury_candidates(
        development_data
    )

    log_injury_candidate_results(
        overall_results=overall_results,
        season_results=season_results,
    )

    logger.info(
        "External validation injury candidate "
        "evaluation completed successfully."
    )

    return overall_results


def main() -> None:
    """Run external validation injury evaluation."""

    try:
        run_injury_candidate_evaluation()

    except Exception:
        logger.exception(
            "External validation injury candidate "
            "evaluation failed."
        )
        raise


if __name__ == "__main__":
    main()