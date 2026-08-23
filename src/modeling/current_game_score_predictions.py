"""Derive model-implied team scores from Spread and Totals predictions."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCORE_PREDICTION_COLUMNS = (
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "home_team", "away_team", "spread_model_name", "spread_model_version",
    "spread_prediction_mode", "totals_model_name", "totals_model_version",
    "totals_prediction_mode", "predicted_home_margin", "predicted_total_points",
    "implied_home_score", "implied_away_score", "implied_score_winner",
    "spread_prediction_generated_at", "totals_prediction_generated_at",
    "score_prediction_generated_at",
)


def create_current_game_score_predictions(
    spread_predictions: pd.DataFrame,
    totals_predictions: pd.DataFrame,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """Create algebraically consistent home and away score estimates."""
    spread_required = {
        "game_id", "season", "game_type", "week", "gameday", "gametime",
        "home_team", "away_team", "model_name", "model_version", "prediction_mode",
        "predicted_home_margin", "prediction_generated_at",
    }
    totals_required = {
        "game_id", "season", "game_type", "week", "gameday", "gametime",
        "home_team", "away_team", "model_name", "model_version", "prediction_mode",
        "predicted_total_points", "prediction_generated_at",
    }
    for data, required, name in (
        (spread_predictions, spread_required, "Spread predictions"),
        (totals_predictions, totals_required, "Totals predictions"),
    ):
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{name} are missing columns: " + ", ".join(missing))
        if data["game_id"].duplicated().any():
            raise ValueError(f"{name} contain duplicate game identifiers.")

    spread = spread_predictions.rename(columns={
        "model_name": "spread_model_name",
        "model_version": "spread_model_version",
        "prediction_mode": "spread_prediction_mode",
        "prediction_generated_at": "spread_prediction_generated_at",
    })
    totals = totals_predictions.rename(columns={
        "model_name": "totals_model_name",
        "model_version": "totals_model_version",
        "prediction_mode": "totals_prediction_mode",
        "prediction_generated_at": "totals_prediction_generated_at",
    })
    metadata = ["season", "game_type", "week", "gameday", "gametime", "home_team", "away_team"]
    result = spread.merge(
        totals,
        on="game_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_totals"),
    )
    if len(result) != len(spread) or len(result) != len(totals):
        raise RuntimeError("Spread and Totals prediction games do not match exactly.")
    for column in metadata:
        spread_values = result[column]
        totals_values = result[f"{column}_totals"]
        if column == "gameday":
            spread_values = pd.to_datetime(spread_values)
            totals_values = pd.to_datetime(totals_values)
        elif column == "gametime":
            spread_values = spread_values.map(
                lambda value: str(value).strip()[:5] if pd.notna(value) else None
            )
            totals_values = totals_values.map(
                lambda value: str(value).strip()[:5] if pd.notna(value) else None
            )
        if not spread_values.equals(totals_values):
            raise RuntimeError(
                f"Spread and Totals prediction metadata differ: {column}."
            )
    margin = result["predicted_home_margin"].to_numpy(dtype=float)
    total = result["predicted_total_points"].to_numpy(dtype=float)
    if not np.isfinite(margin).all() or not np.isfinite(total).all():
        raise ValueError("Spread and Totals predictions must be finite.")
    result["implied_home_score"] = (total + margin) / 2.0
    result["implied_away_score"] = (total - margin) / 2.0
    if (result[["implied_home_score", "implied_away_score"]] < 0.0).any().any():
        raise RuntimeError("Model-implied team scores must not be negative.")
    result["implied_score_winner"] = np.where(
        result["implied_home_score"] >= result["implied_away_score"],
        result["home_team"], result["away_team"],
    )
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    result["score_prediction_generated_at"] = generated_at
    return result.loc[:, SCORE_PREDICTION_COLUMNS].sort_values(
        ["season", "week", "gameday", "gametime", "game_id"]
    ).reset_index(drop=True)
