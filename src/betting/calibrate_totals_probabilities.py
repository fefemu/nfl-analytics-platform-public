"""Leakage-safe residual calibration for production Totals models."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.evaluate_totals_model_candidates import create_ridge_pipeline
from src.modeling.build_current_totals_predictions import (
    load_production_totals_training_data,
)
from src.modeling.production_totals_component import (
    FALLBACK_PREDICTION_MODE,
    PRIMARY_PREDICTION_MODE,
)
from src.modeling.production_totals_model import PRODUCTION_TOTALS_MODEL
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
RESIDUAL_COLUMNS = (
    "prediction_mode", "model_name", "ridge_alpha", "validation_season",
    "game_id", "training_game_count", "actual_total_points",
    "predicted_total_points", "residual_total_points", "absolute_error",
)


def load_totals_calibration_data(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load development-only Totals inputs."""
    data = load_production_totals_training_data(connection).drop(columns=["split_name"])
    splits = connection.execute(
        """
        SELECT game_id, split_name
        FROM analytics.modeling_game_splits
        WHERE split_name IN ('train', 'validation')
        """
    ).fetchdf()
    return data.merge(splits, on="game_id", how="inner", validate="one_to_one")


def _mode_residuals(
    data: pd.DataFrame,
    prediction_mode: str,
    model_name: str,
    features: tuple[str, ...],
    alpha: float,
    require_short_windows: bool,
    validation_seasons: tuple[int, ...],
) -> pd.DataFrame:
    complete = data[list(features)].notna().all(axis=1) & data["target_total_points"].notna()
    if require_short_windows:
        complete &= data["both_short_windows_complete"].fillna(False).astype(bool)
    sample = data.loc[complete].copy()
    rows: list[dict[str, object]] = []
    for season in validation_seasons:
        train = sample.loc[sample["season"] < season]
        validation = sample.loc[sample["season"] == season]
        if train.empty or validation.empty:
            raise RuntimeError(f"Incomplete Totals calibration fold: {season} / {prediction_mode}.")
        model = create_ridge_pipeline(ridge_alpha=alpha)
        model.fit(train[list(features)], train["target_total_points"])
        predicted = model.predict(validation[list(features)])
        for game_id, actual, estimate in zip(
            validation["game_id"], validation["target_total_points"], predicted, strict=True
        ):
            residual = float(actual - estimate)
            rows.append({
                "prediction_mode": prediction_mode,
                "model_name": model_name,
                "ridge_alpha": alpha,
                "validation_season": season,
                "game_id": game_id,
                "training_game_count": len(train),
                "actual_total_points": float(actual),
                "predicted_total_points": float(estimate),
                "residual_total_points": residual,
                "absolute_error": abs(residual),
            })
    return pd.DataFrame(rows, columns=RESIDUAL_COLUMNS)


def create_totals_calibration_residuals(
    development_data: pd.DataFrame,
    validation_seasons: tuple[int, ...] = VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create chronological residual distributions for both routing modes."""
    required = {
        "game_id", "season", "split_name", "target_total_points",
        "both_short_windows_complete", *PRODUCTION_TOTALS_MODEL.feature_columns,
        *PRODUCTION_TOTALS_MODEL.fallback_feature_columns,
    }
    missing = sorted(required - set(development_data.columns))
    if missing:
        raise ValueError("Totals calibration data is missing columns: " + ", ".join(missing))
    if development_data["game_id"].duplicated().any():
        raise ValueError("Totals calibration data contains duplicate game identifiers.")
    if set(development_data["split_name"]) - {"train", "validation"}:
        raise ValueError("Totals calibration must not contain holdout data.")
    if int(development_data["season"].max()) >= 2025:
        raise ValueError("Totals calibration must end before the 2025 holdout.")
    primary = _mode_residuals(
        development_data, PRIMARY_PREDICTION_MODE, PRODUCTION_TOTALS_MODEL.model_name,
        PRODUCTION_TOTALS_MODEL.feature_columns, PRODUCTION_TOTALS_MODEL.ridge_alpha,
        True, validation_seasons,
    )
    fallback = _mode_residuals(
        development_data, FALLBACK_PREDICTION_MODE,
        PRODUCTION_TOTALS_MODEL.fallback_model_name,
        PRODUCTION_TOTALS_MODEL.fallback_feature_columns,
        PRODUCTION_TOTALS_MODEL.fallback_ridge_alpha, False, validation_seasons,
    )
    result = pd.concat([primary, fallback], ignore_index=True)
    if not np.isfinite(result[["residual_total_points", "absolute_error"]]).all().all():
        raise RuntimeError("Totals calibration residuals must be finite.")
    return result.sort_values(["prediction_mode", "validation_season", "game_id"]).reset_index(drop=True)


def run_totals_calibration(database_file: Path = DATABASE_FILE) -> pd.DataFrame:
    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        data = load_totals_calibration_data(connection)
    return create_totals_calibration_residuals(data)


if __name__ == "__main__":
    output = run_totals_calibration()
    print(output.groupby("prediction_mode").agg(
        validation_games=("game_id", "count"), mae=("absolute_error", "mean")
    ).to_string())
