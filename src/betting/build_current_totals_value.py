"""Build and persist current Totals expected-value offers."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.betting.calibrate_totals_probabilities import (
    create_totals_calibration_residuals,
    load_totals_calibration_data,
)
from src.betting.current_totals_value import TOTALS_VALUE_COLUMNS, create_current_totals_value
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

logger = logging.getLogger(__name__)
TARGET_FULL_NAME = "analytics.current_totals_value"


def validate_source_tables(connection: duckdb.DuckDBPyConnection) -> None:
    required = {
        ("analytics", "current_market_board"),
        ("analytics", "current_game_total_predictions"),
        ("analytics", "game_modeling_dataset"),
        ("analytics", "modeling_game_splits"),
    }
    existing = set(connection.execute(
        "SELECT table_schema, table_name FROM information_schema.tables"
    ).fetchall())
    missing = required - existing
    if missing:
        raise RuntimeError("Missing Totals value source tables: " + ", ".join(
            f"{schema}.{table}" for schema, table in sorted(missing)
        ))


def load_current_totals_market(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    market = connection.execute(
        """
        SELECT snapshot_id, fetched_at, game_id, season, game_type, week, gameday,
               commence_time, home_team, away_team, market_key, market_name,
               outcome_name, outcome_type, point, market_line, best_bookmaker_key,
               best_bookmaker_title, best_american_price, best_decimal_odds,
               best_implied_probability, consensus_no_vig_probability, bookmaker_count
        FROM analytics.current_market_board
        WHERE market_key = 'totals'
        ORDER BY commence_time, game_id, market_line, outcome_type
        """
    ).fetchdf()
    if market.empty:
        raise RuntimeError("Current market board contains no Totals offers.")
    return market


def load_current_totals_predictions(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    predictions = connection.execute(
        """
        SELECT game_id, model_name, model_version, prediction_mode,
               predicted_total_points, prediction_generated_at
        FROM analytics.current_game_total_predictions ORDER BY game_id
        """
    ).fetchdf()
    if predictions.empty:
        raise RuntimeError("Current Totals prediction source is empty.")
    return predictions


def create_current_totals_value_table(
    connection: duckdb.DuckDBPyConnection, value: pd.DataFrame
) -> None:
    missing = sorted(set(TOTALS_VALUE_COLUMNS) - set(value.columns))
    if missing:
        raise ValueError("Current Totals value output is missing columns: " + ", ".join(missing))
    connection.register("_current_totals_value", value.loc[:, TOTALS_VALUE_COLUMNS])
    try:
        connection.execute(f"CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS SELECT * FROM _current_totals_value")
    finally:
        connection.unregister("_current_totals_value")


def validate_current_totals_value_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
    expected_game_count: int,
) -> None:
    row_count, game_count = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT game_id) FROM {TARGET_FULL_NAME}"
    ).fetchone()
    if row_count != expected_row_count or game_count != expected_game_count:
        raise RuntimeError("Current Totals value row or game count does not match.")
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM {TARGET_FULL_NAME}
        WHERE market_key <> 'totals' OR outcome_type NOT IN ('over', 'under')
           OR best_decimal_odds <= 1.0 OR calibration_sample_count <= 0
           OR win_probability NOT BETWEEN 0.0 AND 1.0
           OR push_probability NOT BETWEEN 0.0 AND 1.0
           OR loss_probability NOT BETWEEN 0.0 AND 1.0
           OR ABS(win_probability + push_probability + loss_probability - 1.0) > 0.000001
           OR ABS(expected_value_percent - 100.0 * expected_value_per_unit) > 0.000001
           OR positive_expected_value <> (expected_value_per_unit > 0.0)
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]
    if invalid:
        raise RuntimeError("Invalid current Totals value rows found.")


def build_current_totals_value(database_file: Path = DATABASE_FILE) -> pd.DataFrame:
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)
        market = load_current_totals_market(connection)
        predictions = load_current_totals_predictions(connection)
        residuals = create_totals_calibration_residuals(load_totals_calibration_data(connection))
        value = create_current_totals_value(market, predictions, residuals)
        connection.execute("BEGIN TRANSACTION")
        try:
            create_current_totals_value_table(connection, value)
            validate_current_totals_value_table(
                connection, len(value), value["game_id"].nunique()
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    logger.info(
        "Current Totals value build completed: %s offers, %s positive-EV offers.",
        len(value), int(value["positive_expected_value"].sum()),
    )
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_current_totals_value()
