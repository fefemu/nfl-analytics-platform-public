"""
NFL Analytics Platform
Current NFL Market Board Builder

Purpose:
    Combine the latest best bookmaker offers with
    nflverse schedule game identifiers.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

BEST_ODDS_SCHEMA = "analytics"
BEST_ODDS_TABLE = "best_odds_by_line"

BRIDGE_SCHEMA = "analytics"
BRIDGE_TABLE = "odds_event_schedule_bridge"

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_market_board"

BEST_ODDS_FULL_NAME = (
    f"{BEST_ODDS_SCHEMA}.{BEST_ODDS_TABLE}"
)
BRIDGE_FULL_NAME = (
    f"{BRIDGE_SCHEMA}.{BRIDGE_TABLE}"
)
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the best odds and event bridge tables."""

    required_tables = {
        (BEST_ODDS_SCHEMA, BEST_ODDS_TABLE),
        (BRIDGE_SCHEMA, BRIDGE_TABLE),
    }

    existing_tables = {
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            """
        ).fetchall()
    }

    missing_tables = required_tables - existing_tables

    if missing_tables:
        missing_names = ", ".join(
            f"{schema}.{table}"
            for schema, table in sorted(missing_tables)
        )
        raise RuntimeError(
            f"Missing market board source tables: {missing_names}"
        )

    logger.info(
        "Market board sources validated: %s and %s.",
        BEST_ODDS_FULL_NAME,
        BRIDGE_FULL_NAME,
    )


def create_current_market_board(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the latest schedule-linked market board."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH latest_snapshot AS (
            SELECT snapshot_id
            FROM {BEST_ODDS_FULL_NAME}
            ORDER BY fetched_at DESC, snapshot_id DESC
            LIMIT 1
        )

        SELECT
            best.snapshot_id,
            best.fetched_at,
            bridge.game_id,
            best.event_id AS odds_event_id,
            bridge.season,
            bridge.game_type,
            bridge.week,
            best.commence_time,
            bridge.gameday,
            bridge.gametime,
            bridge.home_team_code AS home_team,
            bridge.away_team_code AS away_team,
            best.home_team AS odds_home_team,
            best.away_team AS odds_away_team,
            best.market_key,
            CASE
                WHEN best.market_key = 'h2h'
                    THEN 'Moneyline'
                WHEN best.market_key = 'spreads'
                    THEN 'Spread'
                WHEN best.market_key = 'totals'
                    THEN 'Totals'
                ELSE best.market_key
            END AS market_name,
            best.outcome_name,
            best.outcome_type,
            best.point,
            best.market_line,
            best.best_bookmaker_key,
            best.best_bookmaker_title,
            best.best_american_price,
            best.best_decimal_odds,
            best.best_implied_probability,
            best.best_bookmaker_margin,
            best.best_bookmaker_no_vig_probability,
            best.bookmaker_count,
            best.consensus_no_vig_probability,
            best.minimum_no_vig_probability,
            best.maximum_no_vig_probability,
            best.probability_dispersion,
            best.average_decimal_odds,
            best.decimal_price_improvement
        FROM {BEST_ODDS_FULL_NAME} AS best
        INNER JOIN latest_snapshot AS latest
            ON best.snapshot_id = latest.snapshot_id
        INNER JOIN {BRIDGE_FULL_NAME} AS bridge
            ON best.snapshot_id = bridge.snapshot_id
           AND best.event_id = bridge.odds_event_id
        WHERE bridge.match_status = 'MATCHED'
        """
    )

    logger.info(
        "Current NFL market board created: %s.",
        TARGET_FULL_NAME,
    )


def validate_current_market_board(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the latest NFL market board."""

    expected_count = connection.execute(
        f"""
        WITH latest_snapshot AS (
            SELECT snapshot_id
            FROM {BEST_ODDS_FULL_NAME}
            ORDER BY fetched_at DESC, snapshot_id DESC
            LIMIT 1
        )
        SELECT COUNT(*)
        FROM {BEST_ODDS_FULL_NAME} AS best
        INNER JOIN latest_snapshot AS latest
            ON best.snapshot_id = latest.snapshot_id
        INNER JOIN {BRIDGE_FULL_NAME} AS bridge
            ON best.snapshot_id = bridge.snapshot_id
           AND best.event_id = bridge.odds_event_id
        WHERE bridge.match_status = 'MATCHED'
        """
    ).fetchone()[0]

    target_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if target_count != expected_count:
        raise RuntimeError(
            "Current market board row count does not match "
            "the latest matched best-odds records."
        )

    snapshot_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT snapshot_id)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if target_count > 0 and snapshot_count != 1:
        raise RuntimeError(
            "Current market board must contain "
            "exactly one snapshot."
        )

    invalid_identifier_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR odds_event_id IS NULL
           OR snapshot_id IS NULL
        """
    ).fetchone()[0]

    if invalid_identifier_count > 0:
        raise RuntimeError(
            "Current market board contains "
            "missing identifiers."
        )

    duplicate_offer_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                snapshot_id,
                game_id,
                market_key,
                outcome_type,
                point
            FROM {TARGET_FULL_NAME}
            GROUP BY
                snapshot_id,
                game_id,
                market_key,
                outcome_type,
                point
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_offer_count > 0:
        raise RuntimeError(
            "Current market board contains "
            "duplicate equivalent offers."
        )

    invalid_price_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE best_decimal_odds <= 1.0
           OR best_bookmaker_key IS NULL
           OR consensus_no_vig_probability <= 0.0
           OR consensus_no_vig_probability >= 1.0
        """
    ).fetchone()[0]

    if invalid_price_count > 0:
        raise RuntimeError(
            "Current market board contains "
            "invalid price or probability values."
        )

    logger.info(
        "Current NFL market board validated: %s rows.",
        target_count,
    )


def validate_database_file(
    database_file: Path,
) -> None:
    """Validate that the DuckDB database exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {database_file}"
        )

    if not database_file.is_file():
        raise ValueError(
            f"DuckDB path is not a file: {database_file}"
        )


def build_current_market_board(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the latest NFL market board."""

    validate_database_file(database_file)

    logger.info("Starting current NFL market board build...")

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)

        try:
            connection.execute("BEGIN TRANSACTION")

            create_current_market_board(connection)
            validate_current_market_board(connection)

            connection.execute("COMMIT")

            logger.info(
                "Current market board transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")
            logger.exception(
                "Current market board transaction rolled back."
            )
            raise

    logger.info(
        "Current NFL market board build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the current NFL market board workflow."""

    try:
        build_current_market_board()
    except Exception:
        logger.exception(
            "Current NFL market board build failed."
        )
        raise


if __name__ == "__main__":
    main()