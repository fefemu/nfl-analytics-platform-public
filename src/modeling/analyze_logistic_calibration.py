"""
NFL Analytics Platform
Logistic Model Calibration Analysis

Purpose:
    Diagnose season stability and probability calibration
    for the leading Elo plus QB logistic model.

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

from src.modeling.selected_model import (
    SELECTED_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    VALIDATION_SPLIT,
    evaluate_probabilities,
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


CALIBRATION_BIN_COUNT = 10

SELECTED_REGULARIZATION_C = (
    SELECTED_MODEL.regularization_c
)
SELECTED_LOGISTIC_FEATURES = (
    SELECTED_MODEL.feature_columns
)


def create_validation_predictions(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create logistic and Elo validation predictions."""

    model = train_logistic_model(
        development_data=development_data,
        feature_columns=SELECTED_LOGISTIC_FEATURES,
        regularization_c=SELECTED_REGULARIZATION_C,
    )

    validation_data = development_data.loc[
        development_data["split_name"]
        == VALIDATION_SPLIT
    ].copy()

    if validation_data.empty:
        raise RuntimeError(
            "No external validation games are available."
        )

    logistic_probabilities = model.predict_proba(
        validation_data.loc[:, SELECTED_LOGISTIC_FEATURES]
    )[:, 1]

    predictions = validation_data.loc[
        :,
        [
            "game_id",
            "season",
            "game_date",
            TARGET_COLUMN,
            "elo_home_win_probability",
        ],
    ].copy()

    predictions = predictions.rename(
        columns={
            TARGET_COLUMN: "actual_home_win",
            "elo_home_win_probability": (
                "elo_probability"
            ),
        }
    )

    predictions[
        "logistic_probability"
    ] = logistic_probabilities

    predictions = predictions.sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(drop=True)

    return predictions


def evaluate_predictions_by_season(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate logistic and Elo results by season."""

    required_columns = {
        "season",
        "actual_home_win",
        "logistic_probability",
        "elo_probability",
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

    result_rows: list[
        dict[str, object]
    ] = []

    for season, season_data in predictions.groupby(
        "season",
        sort=True,
    ):
        for (
            model_name,
            probability_column,
        ) in (
            (
                "logistic_elo_qb_post_bye",
                "logistic_probability",
            ),
            (
                "elo",
                "elo_probability",
            ),
        ):
            evaluation = evaluate_probabilities(
                actual_values=season_data[
                    "actual_home_win"
                ],
                probabilities=season_data[
                    probability_column
                ],
            )

            result_rows.append(
                {
                    "season": int(season),
                    "model_name": model_name,
                    "game_count": evaluation.game_count,
                    "accuracy": evaluation.accuracy,
                    "brier_score": (
                        evaluation.brier_score
                    ),
                    "log_loss": evaluation.log_loss,
                }
            )

    return pd.DataFrame(result_rows)


def build_calibration_table(
    actual_values: pd.Series,
    probabilities: pd.Series,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> pd.DataFrame:
    """Group predictions into fixed probability bins."""

    actual = pd.Series(
        actual_values,
        dtype=int,
    ).reset_index(drop=True)

    predicted = pd.Series(
        probabilities,
        dtype=float,
    ).reset_index(drop=True)

    if len(actual) == 0:
        raise ValueError(
            "Calibration analysis requires games."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual values and probabilities "
            "must have equal length."
        )

    if bin_count < 2:
        raise ValueError(
            "Calibration analysis requires at least two bins."
        )

    if not actual.isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Calibration targets must be binary."
        )

    if (
        (predicted < 0.0)
        | (predicted > 1.0)
    ).any():
        raise ValueError(
            "Calibration probabilities must be "
            "between zero and one."
        )

    bin_edges = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
    )

    calibration_data = pd.DataFrame(
        {
            "actual": actual,
            "probability": predicted,
        }
    )

    calibration_data["bin_index"] = pd.cut(
        calibration_data["probability"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    grouped = calibration_data.groupby(
        "bin_index",
        observed=True,
        sort=True,
    )

    calibration_table = grouped.agg(
        game_count=(
            "actual",
            "size",
        ),
        mean_probability=(
            "probability",
            "mean",
        ),
        observed_home_win_rate=(
            "actual",
            "mean",
        ),
    ).reset_index()

    calibration_table[
        "bin_lower_bound"
    ] = calibration_table[
        "bin_index"
    ] / bin_count

    calibration_table[
        "bin_upper_bound"
    ] = (
        calibration_table["bin_index"] + 1
    ) / bin_count

    calibration_table[
        "calibration_gap"
    ] = (
        calibration_table[
            "observed_home_win_rate"
        ]
        - calibration_table[
            "mean_probability"
        ]
    )

    calibration_table[
        "absolute_calibration_gap"
    ] = calibration_table[
        "calibration_gap"
    ].abs()

    return calibration_table.loc[
        :,
        [
            "bin_index",
            "bin_lower_bound",
            "bin_upper_bound",
            "game_count",
            "mean_probability",
            "observed_home_win_rate",
            "calibration_gap",
            "absolute_calibration_gap",
        ],
    ]


def calculate_expected_calibration_error(
    calibration_table: pd.DataFrame,
) -> float:
    """Calculate game-weighted absolute calibration error."""

    if calibration_table.empty:
        raise ValueError(
            "Calibration table must not be empty."
        )

    total_game_count = calibration_table[
        "game_count"
    ].sum()

    if total_game_count <= 0:
        raise ValueError(
            "Calibration table must contain games."
        )

    weighted_error = (
        calibration_table["game_count"]
        * calibration_table[
            "absolute_calibration_gap"
        ]
    ).sum()

    return float(
        weighted_error / total_game_count
    )


def log_calibration_analysis(
    season_results: pd.DataFrame,
    calibration_table: pd.DataFrame,
    expected_calibration_error: float,
) -> None:
    """Log season and calibration diagnostics."""

    logger.info(
        "External validation results by season:"
    )

    for row in season_results.itertuples(index=False):
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

    logger.info(
        "Logistic calibration table:"
    )

    for row in calibration_table.itertuples(index=False):
        logger.info(
            "Bin=%.1f-%.1f | Games=%s | "
            "Mean prediction=%.3f | "
            "Observed rate=%.3f | Gap=%+.3f",
            row.bin_lower_bound,
            row.bin_upper_bound,
            row.game_count,
            row.mean_probability,
            row.observed_home_win_rate,
            row.calibration_gap,
        )

    logger.info(
        "Logistic expected calibration error: %.6f",
        expected_calibration_error,
    )


def run_logistic_calibration_analysis(
    database_file: Path = DATABASE_FILE,
) -> dict[str, object]:
    """Run external validation calibration diagnostics."""

    validate_database_file(database_file)

    logger.info(
        "Starting logistic calibration analysis..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    predictions = create_validation_predictions(
        development_data
    )

    season_results = evaluate_predictions_by_season(
        predictions
    )

    calibration_table = build_calibration_table(
        actual_values=predictions[
            "actual_home_win"
        ],
        probabilities=predictions[
            "logistic_probability"
        ],
    )

    expected_calibration_error = (
        calculate_expected_calibration_error(
            calibration_table
        )
    )

    log_calibration_analysis(
        season_results=season_results,
        calibration_table=calibration_table,
        expected_calibration_error=(
            expected_calibration_error
        ),
    )

    logger.info(
        "Logistic calibration analysis "
        "completed successfully."
    )

    return {
        "predictions": predictions,
        "season_results": season_results,
        "calibration_table": calibration_table,
        "expected_calibration_error": (
            expected_calibration_error
        ),
    }


def main() -> None:
    """Run logistic calibration analysis."""

    try:
        run_logistic_calibration_analysis()

    except Exception:
        logger.exception(
            "Logistic calibration analysis failed."
        )
        raise


if __name__ == "__main__":
    main()