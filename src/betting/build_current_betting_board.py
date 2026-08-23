"""Create one decision-ready board across Moneyline, Spread and Totals."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

logger = logging.getLogger(__name__)
TARGET_FULL_NAME = "analytics.current_betting_board"

BOARD_COLUMNS = (
    "snapshot_id", "fetched_at", "game_id", "season", "game_type", "week",
    "gameday", "commence_time", "home_team", "away_team", "market_key",
    "market_name", "market_line", "outcome_name", "outcome_type", "point",
    "best_bookmaker_key", "best_bookmaker_title", "best_american_price",
    "best_decimal_odds", "bookmaker_count", "model_name", "model_version",
    "prediction_mode", "model_probability", "push_probability", "loss_probability",
    "probability_edge", "probability_edge_percentage_points", "fair_decimal_odds",
    "expected_value_per_unit", "expected_value_percent", "full_kelly_fraction",
    "positive_expected_value", "prediction_generated_at", "betting_board_generated_at",
)


def build_current_betting_board(database_file: Path = DATABASE_FILE) -> pd.DataFrame:
    """Union the three EV products and persist a ranked board."""
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        required = {"current_moneyline_value", "current_spread_value", "current_totals_value"}
        existing = {row[0] for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='analytics'"
        ).fetchall()}
        missing = required - existing
        if missing:
            raise RuntimeError("Missing betting board source tables: " + ", ".join(sorted(missing)))
        board = connection.execute(
            """
            SELECT snapshot_id, fetched_at, game_id, season, game_type, week, gameday,
                   commence_time, home_team, away_team, market_key, market_name,
                   CAST(NULL AS DOUBLE) AS market_line, outcome_name, outcome_type,
                   CAST(NULL AS DOUBLE) AS point, best_bookmaker_key, best_bookmaker_title,
                   best_american_price, best_decimal_odds, bookmaker_count, model_name,
                   model_version, prediction_mode, model_probability,
                   0.0 AS push_probability, 1.0 - model_probability AS loss_probability,
                   probability_edge, probability_edge_percentage_points, fair_decimal_odds,
                   expected_value_per_unit, expected_value_percent, full_kelly_fraction,
                   positive_expected_value, prediction_generated_at
            FROM analytics.current_moneyline_value
            UNION ALL
            SELECT snapshot_id, fetched_at, game_id, season, game_type, week, gameday,
                   commence_time, home_team, away_team, market_key, market_name,
                   market_line, outcome_name, outcome_type, point, best_bookmaker_key,
                   best_bookmaker_title, best_american_price, best_decimal_odds,
                   bookmaker_count, model_name, model_version, prediction_mode,
                   cover_probability, push_probability, loss_probability,
                   probability_edge, probability_edge_percentage_points, fair_decimal_odds,
                   expected_value_per_unit, expected_value_percent, full_kelly_fraction,
                   positive_expected_value, prediction_generated_at
            FROM analytics.current_spread_value
            UNION ALL
            SELECT snapshot_id, fetched_at, game_id, season, game_type, week, gameday,
                   commence_time, home_team, away_team, market_key, market_name,
                   market_line, outcome_name, outcome_type, point, best_bookmaker_key,
                   best_bookmaker_title, best_american_price, best_decimal_odds,
                   bookmaker_count, model_name, model_version, prediction_mode,
                   win_probability, push_probability, loss_probability,
                   probability_edge, probability_edge_percentage_points, fair_decimal_odds,
                   expected_value_per_unit, expected_value_percent, full_kelly_fraction,
                   positive_expected_value, prediction_generated_at
            FROM analytics.current_totals_value
            """
        ).fetchdf()
        board["betting_board_generated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        board = board.loc[:, BOARD_COLUMNS].sort_values(
            ["positive_expected_value", "expected_value_percent", "commence_time"],
            ascending=[False, False, True], kind="stable"
        ).reset_index(drop=True)
        connection.register("_current_betting_board", board)
        try:
            connection.execute(f"CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS SELECT * FROM _current_betting_board")
        finally:
            connection.unregister("_current_betting_board")
        invalid = connection.execute(
            f"""
            SELECT COUNT(*) FROM {TARGET_FULL_NAME}
            WHERE market_key NOT IN ('h2h','spreads','totals')
               OR model_probability NOT BETWEEN 0.0 AND 1.0
               OR push_probability NOT BETWEEN 0.0 AND 1.0
               OR loss_probability NOT BETWEEN 0.0 AND 1.0
               OR ABS(model_probability + push_probability + loss_probability - 1.0) > 0.000001
               OR positive_expected_value <> (expected_value_per_unit > 0.0)
               OR betting_board_generated_at IS NULL
            """
        ).fetchone()[0]
        if invalid:
            raise RuntimeError("Invalid combined betting board rows found.")
    logger.info("Combined betting board completed: %s offers.", len(board))
    return board


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_current_betting_board()
