"""
NFL Analytics Platform
Selected Model Holdout Diagnostics

Purpose:
    Diagnose the frozen selected model's 2025 holdout
    generalization without tuning a new model.

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
from sklearn.pipeline import Pipeline

from src.modeling.evaluate_selected_model_holdout import (
    FINAL_TRAINING_SPLITS,
    load_final_evaluation_data,
    train_frozen_selected_model,
)
from src.modeling.selected_model import (
    SELECTED_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    HOLDOUT_SPLIT,
    TARGET_COLUMN,
    evaluate_probabilities,
    validate_database_file,
    validate_source_tables,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


PHASE_ORDER = (
    "early_regular_season",
    "middle_regular_season",
    "late_regular_season",
    "postseason",
)


def assign_season_phase(
    game_type: str,
    week: int,
) -> str:
    """Assign a game to a descriptive season phase."""

    if game_type != "REG":
        return "postseason"

    if week <= 6:
        return "early_regular_season"

    if week <= 12:
        return "middle_regular_season"

    return "late_regular_season"


def create_holdout_diagnostic_predictions(
    model: Pipeline,
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create holdout predictions for diagnostic analysis."""

    required_columns = {
        "game_id",
        "season",
        "game_type",
        "week",
        "game_date",
        "split_name",
        TARGET_COLUMN,
        "elo_home_win_probability",
        *SELECTED_MODEL.feature_columns,
    }

    missing_columns = sorted(
        required_columns - set(evaluation_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Diagnostic data is missing columns: "
            + ", ".join(missing_columns)
        )

    holdout_data = evaluation_data.loc[
        evaluation_data["split_name"] == HOLDOUT_SPLIT
    ].copy()

    if holdout_data.empty:
        raise RuntimeError(
            "No holdout games are available for "
            "diagnostics."
        )

    holdout_data[
        "logistic_home_win_probability"
    ] = model.predict_proba(
        holdout_data.loc[
            :,
            SELECTED_MODEL.feature_columns,
        ]
    )[:, 1]

    holdout_data["season_phase"] = [
        assign_season_phase(
            game_type=str(game_type),
            week=int(week),
        )
        for game_type, week in zip(
            holdout_data["game_type"],
            holdout_data["week"],
            strict=True,
        )
    ]

    holdout_data["logistic_squared_error"] = (
        holdout_data[
            "logistic_home_win_probability"
        ]
        - holdout_data[TARGET_COLUMN]
    ) ** 2

    holdout_data["elo_squared_error"] = (
        holdout_data["elo_home_win_probability"]
        - holdout_data[TARGET_COLUMN]
    ) ** 2

    holdout_data["logistic_minus_elo_probability"] = (
        holdout_data[
            "logistic_home_win_probability"
        ]
        - holdout_data["elo_home_win_probability"]
    )

    return holdout_data.sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(drop=True)


def build_phase_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare logistic and Elo metrics by season phase."""

    required_columns = {
        "season_phase",
        TARGET_COLUMN,
        "logistic_home_win_probability",
        "elo_home_win_probability",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction data is missing columns: "
            + ", ".join(missing_columns)
        )

    if predictions.empty:
        raise ValueError(
            "Prediction data must not be empty."
        )

    rows: list[dict[str, object]] = []

    for season_phase in PHASE_ORDER:
        phase_data = predictions.loc[
            predictions["season_phase"]
            == season_phase
        ]

        if phase_data.empty:
            continue

        logistic_evaluation = evaluate_probabilities(
            phase_data[TARGET_COLUMN],
            phase_data[
                "logistic_home_win_probability"
            ],
        )

        elo_evaluation = evaluate_probabilities(
            phase_data[TARGET_COLUMN],
            phase_data["elo_home_win_probability"],
        )

        rows.append(
            {
                "season_phase": season_phase,
                "game_count": (
                    logistic_evaluation.game_count
                ),
                "logistic_accuracy": (
                    logistic_evaluation.accuracy
                ),
                "logistic_brier_score": (
                    logistic_evaluation.brier_score
                ),
                "logistic_log_loss": (
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
                "brier_improvement_vs_elo": (
                    elo_evaluation.brier_score
                    - logistic_evaluation.brier_score
                ),
                "log_loss_improvement_vs_elo": (
                    elo_evaluation.log_loss
                    - logistic_evaluation.log_loss
                ),
            }
        )

    return pd.DataFrame(rows)


def build_disagreement_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnose logistic probability changes versus Elo."""

    required_columns = {
        "season_phase",
        TARGET_COLUMN,
        "elo_home_win_probability",
        "logistic_home_win_probability",
        "listed_qb_rating_difference",
        "post_bye_difference",
    }

    missing_columns = sorted(
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Disagreement data is missing columns: "
            + ", ".join(missing_columns)
        )

    if predictions.empty:
        raise ValueError(
            "Disagreement data must not be empty."
        )

    diagnostic_data = predictions.copy()

    diagnostic_data["logistic_prediction"] = (
        diagnostic_data[
            "logistic_home_win_probability"
        ]
        >= SELECTED_MODEL.classification_threshold
    ).astype(int)

    diagnostic_data["elo_prediction"] = (
        diagnostic_data["elo_home_win_probability"]
        >= 0.5
    ).astype(int)

    diagnostic_data["models_disagree"] = (
        diagnostic_data["logistic_prediction"]
        != diagnostic_data["elo_prediction"]
    )

    diagnostic_data["logistic_correct"] = (
        diagnostic_data["logistic_prediction"]
        == diagnostic_data[TARGET_COLUMN]
    )

    diagnostic_data["elo_correct"] = (
        diagnostic_data["elo_prediction"]
        == diagnostic_data[TARGET_COLUMN]
    )

    diagnostic_data["absolute_probability_change"] = (
        diagnostic_data[
            "logistic_home_win_probability"
        ]
        - diagnostic_data["elo_home_win_probability"]
    ).abs()

    outcome_direction = (
        2.0 * diagnostic_data[TARGET_COLUMN] - 1.0
    )

    diagnostic_data["outcome_directed_change"] = (
        diagnostic_data[
            "logistic_home_win_probability"
        ]
        - diagnostic_data["elo_home_win_probability"]
    ) * outcome_direction

    groups = (
        ("all_holdout", diagnostic_data),
        *(
            (
                season_phase,
                diagnostic_data.loc[
                    diagnostic_data["season_phase"]
                    == season_phase
                ],
            )
            for season_phase in PHASE_ORDER
        ),
    )

    rows: list[dict[str, object]] = []

    for group_name, group_data in groups:
        if group_data.empty:
            continue

        disagreement_data = group_data.loc[
            group_data["models_disagree"]
        ]

        logistic_disagreement_wins = int(
            (
                disagreement_data["logistic_correct"]
                & ~disagreement_data["elo_correct"]
            ).sum()
        )

        elo_disagreement_wins = int(
            (
                disagreement_data["elo_correct"]
                & ~disagreement_data["logistic_correct"]
            ).sum()
        )

        rows.append(
            {
                "diagnostic_group": group_name,
                "game_count": len(group_data),
                "disagreement_count": len(
                    disagreement_data
                ),
                "disagreement_rate": (
                    float(
                        disagreement_data.shape[0]
                        / group_data.shape[0]
                    )
                ),
                "logistic_disagreement_wins": (
                    logistic_disagreement_wins
                ),
                "elo_disagreement_wins": (
                    elo_disagreement_wins
                ),
                "mean_absolute_probability_change": (
                    float(
                        group_data[
                            "absolute_probability_change"
                        ].mean()
                    )
                ),
                "mean_outcome_directed_change": (
                    float(
                        group_data[
                            "outcome_directed_change"
                        ].mean()
                    )
                ),
                "mean_absolute_qb_difference_on_disagreements": (
                    float(
                        disagreement_data[
                            "listed_qb_rating_difference"
                        ].abs().mean()
                    )
                    if not disagreement_data.empty
                    else np.nan
                ),
                "post_bye_rate_on_disagreements": (
                    float(
                        (
                            disagreement_data[
                                "post_bye_difference"
                            ]
                            != 0
                        ).mean()
                    )
                    if not disagreement_data.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feature_drift_table(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Measure selected-feature drift into the holdout."""

    development_data = evaluation_data.loc[
        evaluation_data["split_name"].isin(
            FINAL_TRAINING_SPLITS
        )
    ]

    holdout_data = evaluation_data.loc[
        evaluation_data["split_name"] == HOLDOUT_SPLIT
    ]

    if development_data.empty:
        raise RuntimeError(
            "No development games are available for "
            "feature-drift analysis."
        )

    if holdout_data.empty:
        raise RuntimeError(
            "No holdout games are available for "
            "feature-drift analysis."
        )

    rows: list[dict[str, object]] = []

    for feature_name in SELECTED_MODEL.feature_columns:
        development_values = pd.to_numeric(
            development_data[feature_name],
            errors="coerce",
        )

        holdout_values = pd.to_numeric(
            holdout_data[feature_name],
            errors="coerce",
        )

        development_mean = float(
            development_values.mean()
        )
        holdout_mean = float(
            holdout_values.mean()
        )
        development_standard_deviation = float(
            development_values.std(ddof=0)
        )

        if (
            np.isfinite(
                development_standard_deviation
            )
            and development_standard_deviation > 0.0
        ):
            standardized_mean_difference = (
                holdout_mean - development_mean
            ) / development_standard_deviation
        else:
            standardized_mean_difference = np.nan

        rows.append(
            {
                "feature_name": feature_name,
                "development_game_count": int(
                    development_values.notna().sum()
                ),
                "holdout_game_count": int(
                    holdout_values.notna().sum()
                ),
                "development_mean": development_mean,
                "holdout_mean": holdout_mean,
                "development_standard_deviation": (
                    development_standard_deviation
                ),
                "standardized_mean_difference": float(
                    standardized_mean_difference
                ),
            }
        )

    return pd.DataFrame(rows)


def log_phase_diagnostics(
    diagnostics: pd.DataFrame,
) -> None:
    """Log season-phase diagnostic metrics."""

    logger.info(
        "2025 holdout results by season phase:"
    )

    for row in diagnostics.itertuples(index=False):
        logger.info(
            "%s | Games=%s | Logistic accuracy=%.2f%% | "
            "Elo accuracy=%.2f%% | Logistic Brier=%.6f | "
            "Elo Brier=%.6f | Brier vs Elo=%+.6f | "
            "Log loss vs Elo=%+.6f",
            row.season_phase,
            row.game_count,
            row.logistic_accuracy * 100.0,
            row.elo_accuracy * 100.0,
            row.logistic_brier_score,
            row.elo_brier_score,
            row.brier_improvement_vs_elo,
            row.log_loss_improvement_vs_elo,
        )


def log_disagreement_diagnostics(
    diagnostics: pd.DataFrame,
) -> None:
    """Log logistic-versus-Elo disagreement diagnostics."""

    logger.info(
        "Logistic versus Elo holdout diagnostics:"
    )

    for row in diagnostics.itertuples(index=False):
        logger.info(
            "%s | Games=%s | Disagreements=%s "
            "(%.2f%%) | Logistic wins=%s | "
            "Elo wins=%s | Mean absolute probability "
            "change=%.4f | Outcome-directed "
            "change=%+.4f | Mean absolute QB "
            "difference=%.4f | Post-bye rate=%.2f%%",
            row.diagnostic_group,
            row.game_count,
            row.disagreement_count,
            row.disagreement_rate * 100.0,
            row.logistic_disagreement_wins,
            row.elo_disagreement_wins,
            row.mean_absolute_probability_change,
            row.mean_outcome_directed_change,
            row.mean_absolute_qb_difference_on_disagreements,
            row.post_bye_rate_on_disagreements * 100.0,
        )


def log_feature_drift(
    drift_table: pd.DataFrame,
) -> None:
    """Log selected-feature distribution drift."""

    logger.info(
        "Selected-feature drift into 2025 holdout:"
    )

    for row in drift_table.itertuples(index=False):
        logger.info(
            "%s | Development mean=%.6f | "
            "Holdout mean=%.6f | "
            "Standardized mean difference=%+.3f",
            row.feature_name,
            row.development_mean,
            row.holdout_mean,
            row.standardized_mean_difference,
        )


def run_holdout_diagnostics(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run descriptive diagnostics for the frozen model."""

    validate_database_file(database_file)

    logger.info(
        "Starting selected-model holdout diagnostics..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        evaluation_data = load_final_evaluation_data(
            connection
        )

    model = train_frozen_selected_model(
        evaluation_data
    )

    predictions = (
        create_holdout_diagnostic_predictions(
            model=model,
            evaluation_data=evaluation_data,
        )
    )

    phase_diagnostics = build_phase_diagnostics(
        predictions
    )

    disagreement_diagnostics = (
        build_disagreement_diagnostics(
            predictions
        )
    )

    feature_drift = build_feature_drift_table(
        evaluation_data
    )

    log_phase_diagnostics(phase_diagnostics)
    log_disagreement_diagnostics(
        disagreement_diagnostics
    )
    log_feature_drift(feature_drift)

    logger.info(
        "Selected-model holdout diagnostics "
        "completed successfully."
    )

    return (
        phase_diagnostics,
        disagreement_diagnostics,
        feature_drift,
    )


def main() -> None:
    """Run the holdout diagnostic entry point."""

    try:
        run_holdout_diagnostics()
    except Exception:
        logger.exception(
            "Selected-model holdout diagnostics failed."
        )
        raise


if __name__ == "__main__":
    main()