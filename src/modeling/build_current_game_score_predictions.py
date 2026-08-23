"""Build and persist current model-implied home and away scores."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.current_game_score_predictions import (
    SCORE_PREDICTION_COLUMNS,
    create_current_game_score_predictions,
)
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

logger = logging.getLogger(__name__)
TARGET_FULL_NAME = "analytics.current_game_score_predictions"


def load_score_prediction_sources(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two production components required by the score identity."""
    spread = connection.execute(
        """
        SELECT game_id, season, game_type, week, gameday, gametime, home_team,
               away_team, model_name, model_version, prediction_mode,
               predicted_home_margin, prediction_generated_at
        FROM analytics.current_game_spread_predictions ORDER BY game_id
        """
    ).fetchdf()
    totals = connection.execute(
        """
        SELECT game_id, season, game_type, week, gameday, gametime, home_team,
               away_team, model_name, model_version, prediction_mode,
               predicted_total_points, prediction_generated_at
        FROM analytics.current_game_total_predictions ORDER BY game_id
        """
    ).fetchdf()
    if spread.empty or totals.empty:
        raise RuntimeError("Current Spread or Totals score source is empty.")
    return spread, totals


def create_current_game_score_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> None:
    missing = sorted(set(SCORE_PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError("Current score predictions are missing columns: " + ", ".join(missing))
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.register("_current_score_predictions", predictions.loc[:, SCORE_PREDICTION_COLUMNS])
    try:
        connection.execute(f"CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS SELECT * FROM _current_score_predictions")
    finally:
        connection.unregister("_current_score_predictions")


def validate_current_game_score_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    row_count = connection.execute(f"SELECT COUNT(*) FROM {TARGET_FULL_NAME}").fetchone()[0]
    if row_count != expected_row_count:
        raise RuntimeError("Current score prediction row count does not match.")
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM {TARGET_FULL_NAME}
        WHERE implied_home_score < 0.0 OR implied_away_score < 0.0
           OR ABS(implied_home_score + implied_away_score - predicted_total_points) > 0.000001
           OR ABS(implied_home_score - implied_away_score - predicted_home_margin) > 0.000001
           OR implied_score_winner NOT IN (home_team, away_team)
           OR (predicted_home_margin >= 0.0 AND implied_score_winner <> home_team)
           OR (predicted_home_margin < 0.0 AND implied_score_winner <> away_team)
           OR spread_model_name IS NULL OR totals_model_name IS NULL
           OR spread_prediction_generated_at IS NULL
           OR totals_prediction_generated_at IS NULL
           OR score_prediction_generated_at IS NULL
        """
    ).fetchone()[0]
    if invalid:
        raise RuntimeError("Invalid current model-implied score predictions found.")


def build_current_game_score_predictions(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build the current implied-score product transactionally."""
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        spread, totals = load_score_prediction_sources(connection)
        predictions = create_current_game_score_predictions(spread, totals)
        connection.execute("BEGIN TRANSACTION")
        try:
            create_current_game_score_predictions_table(connection, predictions)
            validate_current_game_score_predictions_table(connection, len(predictions))
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    logger.info("Current model-implied score build completed: %s games.", len(predictions))
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_current_game_score_predictions()
