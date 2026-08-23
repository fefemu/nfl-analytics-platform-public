"""Leakage-safe dispersion calibration for the production Totals fallback.

The module evaluates simple post-model scaling around each fold's training
mean.  It never reads the protected holdout and does not mutate production.
"""

from pathlib import Path
import logging

import duckdb
import numpy as np
import pandas as pd

from src.modeling.evaluate_totals_fallback_candidates import (
    TOTALS_TARGET_COLUMN,
    create_ridge_pipeline,
    load_totals_fallback_development_data,
    prepare_common_fallback_sample,
)
from src.modeling.evaluate_spread_model_candidates import calculate_regression_metrics
from src.modeling.production_totals_model import PRODUCTION_TOTALS_MODEL
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file


logger = logging.getLogger(__name__)

VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
DISPERSION_FACTORS = (0.75, 0.90, 1.00, 1.10, 1.25, 1.50)


def create_locked_fallback_oof_predictions(
    development_data: pd.DataFrame,
    validation_seasons: tuple[int, ...] = VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create chronological predictions for the locked fallback model."""

    sample = prepare_common_fallback_sample(development_data)
    rows: list[pd.DataFrame] = []
    features = list(PRODUCTION_TOTALS_MODEL.fallback_feature_columns)

    for season in validation_seasons:
        train = sample.loc[sample["season"] < season].copy()
        validation = sample.loc[sample["season"] == season].copy()
        if train.empty or validation.empty:
            raise RuntimeError(f"Missing chronological sample for {season}.")

        model = create_ridge_pipeline(
            ridge_alpha=PRODUCTION_TOTALS_MODEL.fallback_ridge_alpha
        )
        model.fit(train[features], train[TOTALS_TARGET_COLUMN])
        fold = validation[["game_id", "season", TOTALS_TARGET_COLUMN]].copy()
        fold["training_mean_total"] = float(train[TOTALS_TARGET_COLUMN].mean())
        fold["raw_predicted_total"] = model.predict(validation[features])
        rows.append(fold)

    return pd.concat(rows, ignore_index=True)


def evaluate_dispersion_factors(
    predictions: pd.DataFrame,
    factors: tuple[float, ...] = DISPERSION_FACTORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare scaling factors around the chronological training mean."""

    required = {
        "game_id", "season", TOTALS_TARGET_COLUMN,
        "training_mean_total", "raw_predicted_total",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Calibration predictions are missing columns: " + ", ".join(missing))
    if not factors or any(factor <= 0 for factor in factors):
        raise ValueError("Dispersion factors must be positive.")

    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    actual_series = predictions[TOTALS_TARGET_COLUMN].astype(float)
    actual = actual_series.to_numpy(dtype=float)

    for factor in factors:
        calibrated = (
            predictions["training_mean_total"].to_numpy(dtype=float)
            + float(factor)
            * (
                predictions["raw_predicted_total"].to_numpy(dtype=float)
                - predictions["training_mean_total"].to_numpy(dtype=float)
            )
        )
        metrics = calculate_regression_metrics(
            actual_margin=actual_series,
            predicted_margin=calibrated,
        )
        summary_rows.append({
            "dispersion_factor": float(factor),
            "validation_game_count": len(predictions),
            "prediction_standard_deviation": float(np.std(calibrated, ddof=0)),
            "actual_standard_deviation": float(np.std(actual, ddof=0)),
            **metrics,
        })
        detail = predictions[["game_id", "season", TOTALS_TARGET_COLUMN]].copy()
        detail["dispersion_factor"] = float(factor)
        detail["predicted_total"] = calibrated
        detail_rows.append(detail)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["validation_mae", "validation_rmse", "dispersion_factor"],
        kind="stable",
    ).reset_index(drop=True)
    return summary, pd.concat(detail_rows, ignore_index=True)


def run_totals_dispersion_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the read-only chronological calibration benchmark."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        development = load_totals_fallback_development_data(connection)
    predictions = create_locked_fallback_oof_predictions(development)
    result = evaluate_dispersion_factors(predictions)
    logger.info(
        "Totals dispersion calibration completed on %s validation games without holdout.",
        len(predictions),
    )
    return result


def main() -> None:
    summary, _ = run_totals_dispersion_backtest()
    print("\nTOTALS DISPERSION CALIBRATION\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
