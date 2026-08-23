"""
NFL Analytics Platform
Logistic Calibration Method Evaluation

Purpose:
    Fit probability calibrators on time-series
    out-of-fold predictions and evaluate them on
    the external validation period.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.modeling.analyze_logistic_calibration import (
    build_calibration_table,
    calculate_expected_calibration_error,
    create_validation_predictions,
    SELECTED_LOGISTIC_FEATURES,
)
from src.modeling.time_series_validation import (
    create_expanding_season_folds,
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


SELECTED_REGULARIZATION_C = 1.0
PROBABILITY_EPSILON = 0.000001


@dataclass(frozen=True)
class FittedProbabilityCalibrators:
    """Store fitted sigmoid and isotonic calibrators."""

    sigmoid: LogisticRegression
    isotonic: IsotonicRegression


def probabilities_to_logits(
    probabilities: np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert probabilities to finite log odds."""

    predicted = np.asarray(
        probabilities,
        dtype=float,
    )

    if predicted.ndim != 1:
        raise ValueError(
            "Probability input must be one-dimensional."
        )

    if (
        (predicted < 0.0)
        | (predicted > 1.0)
    ).any():
        raise ValueError(
            "Probabilities must be between zero and one."
        )

    clipped = np.clip(
        predicted,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    return np.log(
        clipped / (1.0 - clipped)
    )


def create_time_cv_oof_predictions(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create expanding-window out-of-fold predictions."""

    folds = create_expanding_season_folds(
        development_data
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    for fold in folds:
        model = train_logistic_model(
            development_data=fold.development_data,
            feature_columns=SELECTED_LOGISTIC_FEATURES,
            regularization_c=(
                SELECTED_REGULARIZATION_C
            ),
        )

        validation_data = fold.development_data.loc[
            fold.development_data["split_name"]
            == VALIDATION_SPLIT
        ].copy()

        probabilities = model.predict_proba(
            validation_data.loc[:, SELECTED_LOGISTIC_FEATURES]
        )[:, 1]

        fold_predictions = validation_data.loc[
            :,
            [
                "game_id",
                "season",
                "game_date",
                TARGET_COLUMN,
            ],
        ].copy()

        fold_predictions = fold_predictions.rename(
            columns={
                TARGET_COLUMN: "actual_home_win",
            }
        )

        fold_predictions[
            "raw_probability"
        ] = probabilities

        prediction_frames.append(
            fold_predictions
        )

    oof_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return oof_predictions.sort_values(
        by=[
            "game_date",
            "game_id",
        ]
    ).reset_index(drop=True)


def fit_probability_calibrators(
    oof_predictions: pd.DataFrame,
) -> FittedProbabilityCalibrators:
    """Fit calibrators using out-of-fold predictions."""

    required_columns = {
        "actual_home_win",
        "raw_probability",
    }

    missing_columns = sorted(
        required_columns - set(oof_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "OOF predictions are missing columns: "
            + ", ".join(missing_columns)
        )

    if oof_predictions.empty:
        raise ValueError(
            "OOF predictions must not be empty."
        )

    target = oof_predictions[
        "actual_home_win"
    ].to_numpy(dtype=int)

    if len(np.unique(target)) != 2:
        raise ValueError(
            "OOF predictions must contain both classes."
        )

    raw_probabilities = oof_predictions[
        "raw_probability"
    ].to_numpy(dtype=float)

    logits = probabilities_to_logits(
        raw_probabilities
    ).reshape(-1, 1)

    sigmoid = LogisticRegression(
        C=1000000.0,
        max_iter=2000,
        random_state=42,
    )

    sigmoid.fit(
        logits,
        target,
    )

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        raw_probabilities,
        target,
    )

    return FittedProbabilityCalibrators(
        sigmoid=sigmoid,
        isotonic=isotonic,
    )


def apply_probability_calibrators(
    raw_probabilities: np.ndarray | pd.Series,
    calibrators: FittedProbabilityCalibrators,
) -> dict[str, np.ndarray]:
    """Apply fitted calibrators to raw probabilities."""

    raw = np.asarray(
        raw_probabilities,
        dtype=float,
    )

    logits = probabilities_to_logits(
        raw
    ).reshape(-1, 1)

    sigmoid_probabilities = (
        calibrators.sigmoid.predict_proba(
            logits
        )[:, 1]
    )

    isotonic_probabilities = (
        calibrators.isotonic.predict(
            raw
        )
    )

    return {
        "sigmoid_probability": (
            sigmoid_probabilities
        ),
        "isotonic_probability": (
            isotonic_probabilities
        ),
    }


def create_calibrated_validation_predictions(
    development_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create OOF and calibrated external predictions."""

    oof_predictions = (
        create_time_cv_oof_predictions(
            development_data
        )
    )

    calibrators = fit_probability_calibrators(
        oof_predictions
    )

    validation_predictions = (
        create_validation_predictions(
            development_data
        )
    )

    calibrated = apply_probability_calibrators(
        raw_probabilities=validation_predictions[
            "logistic_probability"
        ],
        calibrators=calibrators,
    )

    validation_predictions[
        "sigmoid_probability"
    ] = calibrated["sigmoid_probability"]

    validation_predictions[
        "isotonic_probability"
    ] = calibrated["isotonic_probability"]

    return (
        oof_predictions,
        validation_predictions,
    )


def evaluate_calibration_methods(
    validation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate raw and calibrated validation probabilities."""

    probability_columns = {
        "logistic_raw": "logistic_probability",
        "logistic_sigmoid": "sigmoid_probability",
        "logistic_isotonic": "isotonic_probability",
        "elo": "elo_probability",
    }

    required_columns = {
        "actual_home_win",
        *probability_columns.values(),
    }

    missing_columns = sorted(
        required_columns
        - set(validation_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Validation predictions are missing columns: "
            + ", ".join(missing_columns)
        )

    result_rows: list[
        dict[str, object]
    ] = []

    for (
        model_name,
        probability_column,
    ) in probability_columns.items():
        probabilities = validation_predictions[
            probability_column
        ]

        evaluation = evaluate_probabilities(
            actual_values=validation_predictions[
                "actual_home_win"
            ],
            probabilities=probabilities,
        )

        calibration_table = build_calibration_table(
            actual_values=validation_predictions[
                "actual_home_win"
            ],
            probabilities=probabilities,
        )

        expected_calibration_error = (
            calculate_expected_calibration_error(
                calibration_table
            )
        )

        result_rows.append(
            {
                "model_name": model_name,
                "game_count": evaluation.game_count,
                "accuracy": evaluation.accuracy,
                "brier_score": (
                    evaluation.brier_score
                ),
                "log_loss": evaluation.log_loss,
                "expected_calibration_error": (
                    expected_calibration_error
                ),
            }
        )

    results = pd.DataFrame(result_rows)

    return results.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ],
        ascending=True,
    ).reset_index(drop=True)


def log_calibration_method_results(
    results: pd.DataFrame,
) -> None:
    """Log raw and calibrated external results."""

    logger.info(
        "External validation calibration-method results:"
    )

    for row in results.itertuples(index=False):
        logger.info(
            "%s | Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f | ECE=%.6f",
            row.model_name,
            row.game_count,
            100.0 * row.accuracy,
            row.brier_score,
            row.log_loss,
            row.expected_calibration_error,
        )

    best_result = results.iloc[0]

    logger.info(
        "Best calibration method by Brier score: %s",
        best_result["model_name"],
    )


def run_calibration_method_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run leakage-safe calibration comparison."""

    validate_database_file(database_file)

    logger.info(
        "Starting calibration-method evaluation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    (
        oof_predictions,
        validation_predictions,
    ) = create_calibrated_validation_predictions(
        development_data
    )

    logger.info(
        "Calibration training data: %s OOF games.",
        len(oof_predictions),
    )

    results = evaluate_calibration_methods(
        validation_predictions
    )

    log_calibration_method_results(results)

    logger.info(
        "Calibration-method evaluation "
        "completed successfully."
    )

    return results


def main() -> None:
    """Run calibration-method evaluation."""

    try:
        run_calibration_method_evaluation()

    except Exception:
        logger.exception(
            "Calibration-method evaluation failed."
        )
        raise


if __name__ == "__main__":
    main()