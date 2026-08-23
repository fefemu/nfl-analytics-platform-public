"""
NFL Analytics Platform
Probability Sharpening Backtest

Purpose:
    Evaluate fixed logit-scaling factors on chronological
    out-of-fold predictions from the production fallback model.

    A factor above one moves probabilities away from 50%.
    This is a development-only benchmark and does not alter
    the production probability model.

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

from src.modeling.analyze_logistic_calibration import (
    build_calibration_table,
    calculate_expected_calibration_error,
)
from src.modeling.backtest_external_probability_fallback import (
    EXTERNAL_LOGISTIC_CANDIDATE,
    create_fallback_oof_predictions,
    load_probability_fallback_data,
)
from src.modeling.evaluate_logistic_calibration_methods import (
    probabilities_to_logits,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    evaluate_probabilities,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


DEFAULT_SHARPENING_FACTORS = (
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "sharpening_factor",
    "fold_count",
    "game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "probability_standard_deviation",
    "minimum_probability",
    "maximum_probability",
    "brier_score_delta_vs_raw",
    "log_loss_delta_vs_raw",
)

SEASON_COLUMNS = (
    "candidate_name",
    "sharpening_factor",
    "validation_season",
    "game_count",
    "brier_score",
    "log_loss",
    "probability_standard_deviation",
)


def apply_logit_sharpening(
    probabilities: pd.Series | np.ndarray,
    sharpening_factor: float,
) -> np.ndarray:
    """Scale probability log odds around the neutral 50%."""

    if sharpening_factor <= 0.0:
        raise ValueError(
            "Sharpening factor must be greater than zero."
        )

    logits = probabilities_to_logits(probabilities)
    scaled_logits = sharpening_factor * logits

    return 1.0 / (1.0 + np.exp(-scaled_logits))


def create_sharpening_predictions(
    raw_predictions: pd.DataFrame,
    sharpening_factors: tuple[float, ...] = (
        DEFAULT_SHARPENING_FACTORS
    ),
) -> pd.DataFrame:
    """Create aligned candidate probabilities."""

    required_columns = {
        "game_id",
        "validation_season",
        "actual_home_win",
        "home_win_probability",
    }
    missing_columns = sorted(
        required_columns - set(raw_predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sharpening input is missing columns: "
            + ", ".join(missing_columns)
        )

    if not sharpening_factors:
        raise ValueError(
            "At least one sharpening factor is required."
        )

    prediction_frames = []

    for factor in sharpening_factors:
        candidate = raw_predictions.loc[
            :,
            [
                "game_id",
                "validation_season",
                "actual_home_win",
            ],
        ].copy()
        candidate["sharpening_factor"] = float(factor)
        candidate["candidate_name"] = (
            f"fallback_logit_scale_{factor:.2f}"
        )
        candidate["home_win_probability"] = (
            apply_logit_sharpening(
                raw_predictions["home_win_probability"],
                sharpening_factor=float(factor),
            )
        )
        prediction_frames.append(candidate)

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def evaluate_sharpening_candidates(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate pooled and season-level candidate metrics."""

    summary_rows = []
    season_rows = []

    for (
        candidate_name,
        candidate_predictions,
    ) in predictions.groupby("candidate_name", sort=False):
        factor = float(
            candidate_predictions["sharpening_factor"].iloc[0]
        )
        evaluation = evaluate_probabilities(
            actual_values=candidate_predictions["actual_home_win"],
            probabilities=candidate_predictions[
                "home_win_probability"
            ],
        )
        calibration_table = build_calibration_table(
            actual_values=candidate_predictions["actual_home_win"],
            probabilities=candidate_predictions[
                "home_win_probability"
            ],
        )

        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "sharpening_factor": factor,
                "fold_count": int(
                    candidate_predictions[
                        "validation_season"
                    ].nunique()
                ),
                "game_count": len(candidate_predictions),
                "accuracy": evaluation.accuracy,
                "brier_score": evaluation.brier_score,
                "log_loss": evaluation.log_loss,
                "expected_calibration_error": (
                    calculate_expected_calibration_error(
                        calibration_table
                    )
                ),
                "probability_standard_deviation": float(
                    candidate_predictions[
                        "home_win_probability"
                    ].std(ddof=0)
                ),
                "minimum_probability": float(
                    candidate_predictions[
                        "home_win_probability"
                    ].min()
                ),
                "maximum_probability": float(
                    candidate_predictions[
                        "home_win_probability"
                    ].max()
                ),
            }
        )

        for validation_season, season_predictions in (
            candidate_predictions.groupby(
                "validation_season",
                sort=True,
            )
        ):
            season_evaluation = evaluate_probabilities(
                actual_values=season_predictions[
                    "actual_home_win"
                ],
                probabilities=season_predictions[
                    "home_win_probability"
                ],
            )
            season_rows.append(
                {
                    "candidate_name": candidate_name,
                    "sharpening_factor": factor,
                    "validation_season": int(validation_season),
                    "game_count": len(season_predictions),
                    "brier_score": season_evaluation.brier_score,
                    "log_loss": season_evaluation.log_loss,
                    "probability_standard_deviation": float(
                        season_predictions[
                            "home_win_probability"
                        ].std(ddof=0)
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    raw_row = summary.loc[
        np.isclose(summary["sharpening_factor"], 1.0)
    ]

    if len(raw_row) != 1:
        raise ValueError(
            "Sharpening candidates must include factor 1.00."
        )

    raw_brier = float(raw_row["brier_score"].iloc[0])
    raw_log_loss = float(raw_row["log_loss"].iloc[0])
    summary["brier_score_delta_vs_raw"] = (
        summary["brier_score"] - raw_brier
    )
    summary["log_loss_delta_vs_raw"] = (
        summary["log_loss"] - raw_log_loss
    )

    summary = summary.loc[:, SUMMARY_COLUMNS].sort_values(
        by=["brier_score", "log_loss", "sharpening_factor"],
        kind="stable",
    ).reset_index(drop=True)

    season_results = pd.DataFrame(
        season_rows,
        columns=SEASON_COLUMNS,
    ).sort_values(
        by=["validation_season", "sharpening_factor"],
        kind="stable",
    ).reset_index(drop=True)

    return summary, season_results


def run_probability_sharpening_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe OOF fallback sharpening benchmark."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        source_data = load_probability_fallback_data(connection)

    all_predictions, _ = create_fallback_oof_predictions(
        source_data=source_data
    )
    raw_predictions = all_predictions.loc[
        all_predictions["candidate_name"]
        == EXTERNAL_LOGISTIC_CANDIDATE
    ].copy()

    sharpening_predictions = create_sharpening_predictions(
        raw_predictions=raw_predictions
    )
    summary, season_results = evaluate_sharpening_candidates(
        sharpening_predictions
    )

    logger.info(
        "Probability sharpening backtest completed: "
        "%s OOF games across %s folds.",
        len(raw_predictions),
        raw_predictions["validation_season"].nunique(),
    )

    return summary, season_results


def main() -> None:
    """Run and print fallback sharpening results."""

    summary, season_results = (
        run_probability_sharpening_backtest()
    )

    print("\nPROBABILITY SHARPENING BACKTEST SUMMARY\n")
    print(summary.to_string(index=False))

    print("\nSEASON-LEVEL SHARPENING RESULTS\n")
    print(season_results.to_string(index=False))


if __name__ == "__main__":
    main()
