"""
NFL Analytics Platform
Current Moneyline Value Builder

Purpose:
    Load current Moneyline offers and production win
    probabilities, calculate expected value and persist
    the validated result in DuckDB.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.betting.current_moneyline_value import (
    MONEYLINE_VALUE_COLUMNS,
    create_current_moneyline_value,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

MARKET_BOARD_SCHEMA = "analytics"
MARKET_BOARD_TABLE = "current_market_board"
MARKET_BOARD_FULL_NAME = (
    f"{MARKET_BOARD_SCHEMA}.{MARKET_BOARD_TABLE}"
)

PREDICTION_SCHEMA = "analytics"
PREDICTION_TABLE = "current_game_predictions"
PREDICTION_FULL_NAME = (
    f"{PREDICTION_SCHEMA}.{PREDICTION_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_moneyline_value"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the required Moneyline value sources."""

    required_tables = {
        (
            MARKET_BOARD_SCHEMA,
            MARKET_BOARD_TABLE,
        ),
        (
            PREDICTION_SCHEMA,
            PREDICTION_TABLE,
        ),
    }

    existing_tables = {
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT
                table_schema,
                table_name
            FROM information_schema.tables
            """
        ).fetchall()
    }

    missing_tables = (
        required_tables - existing_tables
    )

    if missing_tables:
        missing_names = ", ".join(
            f"{schema}.{table}"
            for schema, table
            in sorted(missing_tables)
        )

        raise RuntimeError(
            "Missing Moneyline value source tables: "
            + missing_names
        )

    logger.info(
        "Moneyline value sources validated: "
        "%s and %s.",
        MARKET_BOARD_FULL_NAME,
        PREDICTION_FULL_NAME,
    )


def load_current_moneyline_market(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load current two-way Moneyline offers."""

    market_board = connection.execute(
        f"""
        SELECT
            snapshot_id,
            fetched_at,
            game_id,
            season,
            game_type,
            week,
            gameday,
            commence_time,
            home_team,
            away_team,
            market_key,
            market_name,
            outcome_name,
            outcome_type,
            best_bookmaker_key,
            best_bookmaker_title,
            best_american_price,
            best_decimal_odds,
            best_implied_probability,
            bookmaker_count,
            consensus_no_vig_probability

        FROM {MARKET_BOARD_FULL_NAME}

        WHERE market_key = 'h2h'

        ORDER BY
            commence_time,
            game_id,
            outcome_type
        """
    ).fetchdf()

    if market_board.empty:
        raise RuntimeError(
            "Current market board contains no "
            "Moneyline offers."
        )

    logger.info(
        "Current Moneyline market loaded: "
        "%s offers across %s games.",
        len(market_board),
        market_board["game_id"].nunique(),
    )

    return market_board


def load_current_game_predictions(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load current production win probabilities."""

    predictions = connection.execute(
        f"""
        SELECT
            game_id,
            model_name,
            model_version,
            prediction_mode,
            home_win_probability,
            away_win_probability,
            prediction_generated_at

        FROM {PREDICTION_FULL_NAME}

        ORDER BY game_id
        """
    ).fetchdf()

    if predictions.empty:
        raise RuntimeError(
            "Current game prediction source is empty."
        )

    logger.info(
        "Current game predictions loaded: %s games.",
        len(predictions),
    )

    return predictions


def create_current_moneyline_value_table(
    connection: duckdb.DuckDBPyConnection,
    value: pd.DataFrame,
) -> None:
    """Persist the current Moneyline value table."""

    missing_columns = sorted(
        set(MONEYLINE_VALUE_COLUMNS)
        - set(value.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current Moneyline value output is "
            "missing columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "_current_moneyline_value",
        value.loc[
            :,
            MONEYLINE_VALUE_COLUMNS,
        ],
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
                {TARGET_FULL_NAME}
            AS
            SELECT *
            FROM _current_moneyline_value
            """
        )
    finally:
        connection.unregister(
            "_current_moneyline_value"
        )


def validate_current_moneyline_value_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
    expected_game_count: int,
) -> None:
    """Validate the persisted Moneyline value table."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current Moneyline value row count does "
            f"not match: expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    actual_game_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT game_id)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_game_count != expected_game_count:
        raise RuntimeError(
            "Current Moneyline value game count does "
            f"not match: expected {expected_game_count}, "
            f"found {actual_game_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                outcome_type
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                outcome_type
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current Moneyline "
            "game-outcome rows found."
        )

    incomplete_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) <> 2
                OR COUNT(
                    DISTINCT outcome_type
                ) <> 2
        )
        """
    ).fetchone()[0]

    if incomplete_game_count > 0:
        raise RuntimeError(
            "Incomplete current Moneyline games found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE model_probability <= 0.0
           OR model_probability >= 1.0
           OR consensus_no_vig_probability <= 0.0
           OR consensus_no_vig_probability >= 1.0
           OR best_implied_probability <= 0.0
           OR best_implied_probability >= 1.0
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid current Moneyline "
            "probabilities found."
        )

    invalid_price_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE best_decimal_odds <= 1.0
           OR fair_decimal_odds <= 1.0
           OR best_bookmaker_key IS NULL
           OR bookmaker_count <= 0
        """
    ).fetchone()[0]

    if invalid_price_count > 0:
        raise RuntimeError(
            "Invalid current Moneyline prices found."
        )

    invalid_calculation_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT isfinite(probability_edge)
           OR NOT isfinite(
                probability_edge_percentage_points
           )
           OR NOT isfinite(fair_decimal_odds)
           OR NOT isfinite(expected_value_per_unit)
           OR NOT isfinite(expected_value_percent)
           OR NOT isfinite(full_kelly_fraction)
           OR ABS(
                expected_value_per_unit
                - (
                    model_probability
                    * best_decimal_odds
                    - 1.0
                )
           ) > 0.000001
           OR ABS(
                probability_edge
                - (
                    model_probability
                    - consensus_no_vig_probability
                )
           ) > 0.000001
           OR ABS(
                fair_decimal_odds
                - (
                    1.0 / model_probability
                )
           ) > 0.000001
           OR full_kelly_fraction < 0.0
        """
    ).fetchone()[0]

    if invalid_calculation_count > 0:
        raise RuntimeError(
            "Invalid current Moneyline "
            "value calculations found."
        )

    invalid_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE positive_expected_value
            <> (
                expected_value_per_unit > 0.0
            )
        """
    ).fetchone()[0]

    if invalid_flag_count > 0:
        raise RuntimeError(
            "Invalid current Moneyline "
            "positive-EV flags found."
        )

    invalid_probability_sum_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    game_id,
                    SUM(
                        model_probability
                    ) AS model_probability_sum,
                    SUM(
                        consensus_no_vig_probability
                    ) AS market_probability_sum
                FROM {TARGET_FULL_NAME}
                GROUP BY game_id
                HAVING ABS(
                    model_probability_sum - 1.0
                ) > 0.000001
                   OR ABS(
                    market_probability_sum - 1.0
                ) > 0.000001
            )
            """
        ).fetchone()[0]
    )

    if invalid_probability_sum_count > 0:
        raise RuntimeError(
            "Current Moneyline probabilities do not "
            "sum to one by game."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE snapshot_id IS NULL
           OR fetched_at IS NULL
           OR game_id IS NULL
           OR season IS NULL
           OR week IS NULL
           OR commence_time IS NULL
           OR home_team IS NULL
           OR away_team IS NULL
           OR home_team = away_team
           OR market_key <> 'h2h'
           OR market_name <> 'Moneyline'
           OR outcome_type NOT IN (
                'home',
                'away'
           )
           OR model_name IS NULL
           OR model_version IS NULL
           OR prediction_mode IS NULL
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current Moneyline metadata found."
        )

    logger.info(
        "Current Moneyline value table validated: "
        "%s offers across %s games.",
        actual_row_count,
        actual_game_count,
    )


def build_current_moneyline_value(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build and persist current Moneyline value."""

    validate_database_file(database_file)

    logger.info(
        "Starting current Moneyline value build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_source_tables(connection)

        market_board = (
            load_current_moneyline_market(
                connection
            )
        )

        predictions = (
            load_current_game_predictions(
                connection
            )
        )

        value = create_current_moneyline_value(
            market_board=market_board,
            predictions=predictions,
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_current_moneyline_value_table(
                connection=connection,
                value=value,
            )

            validate_current_moneyline_value_table(
                connection=connection,
                expected_row_count=len(value),
                expected_game_count=(
                    value["game_id"].nunique()
                ),
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    positive_ev_count = int(
        value["positive_expected_value"].sum()
    )

    logger.info(
        "Current Moneyline value build completed: "
        "%s offers across %s games, "
        "%s positive-EV offers.",
        len(value),
        value["game_id"].nunique(),
        positive_ev_count,
    )

    return value


def main() -> None:
    """Run the current Moneyline value builder."""

    try:
        build_current_moneyline_value()
    except Exception:
        logger.exception(
            "Current Moneyline value build failed."
        )
        raise


if __name__ == "__main__":
    main()