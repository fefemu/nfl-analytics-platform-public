"""
NFL Analytics Platform
Processed Odds Builder

Purpose:
    Build analytics-ready NFL odds outcomes from
    the normalized raw Odds API tables.

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

SOURCE_SCHEMA = "raw"
SNAPSHOTS_TABLE = "odds_snapshots"
EVENTS_TABLE = "odds_events"
MARKETS_TABLE = "odds_markets"

TARGET_SCHEMA = "processed"
TARGET_TABLE = "odds_market_outcomes"

SNAPSHOTS_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SNAPSHOTS_TABLE}"
)
EVENTS_FULL_NAME = f"{SOURCE_SCHEMA}.{EVENTS_TABLE}"
MARKETS_FULL_NAME = f"{SOURCE_SCHEMA}.{MARKETS_TABLE}"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate that all raw odds source tables exist."""

    required_tables = {
        SNAPSHOTS_TABLE,
        EVENTS_TABLE,
        MARKETS_TABLE,
    }

    existing_tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = ?
            """,
            [SOURCE_SCHEMA],
        ).fetchall()
    }

    missing_tables = required_tables - existing_tables

    if missing_tables:
        missing_names = ", ".join(
            sorted(missing_tables)
        )
        raise RuntimeError(
            f"Missing raw odds source tables: {missing_names}"
        )

    logger.info(
        "Raw odds source tables validated: %s, %s and %s.",
        SNAPSHOTS_FULL_NAME,
        EVENTS_FULL_NAME,
        MARKETS_FULL_NAME,
    )


def create_processed_odds_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create analytics-ready odds outcomes."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH base_outcomes AS (
            SELECT
                market.snapshot_id,
                snapshot.fetched_at,
                market.event_id,
                event.commence_time,
                event.home_team,
                event.away_team,
                market.bookmaker_key,
                market.bookmaker_title,
                market.bookmaker_last_update,
                market.market_key,
                market.outcome_name,
                CASE
                    WHEN market.outcome_name = event.home_team
                        THEN 'home'
                    WHEN market.outcome_name = event.away_team
                        THEN 'away'
                    WHEN LOWER(market.outcome_name) = 'over'
                        THEN 'over'
                    WHEN LOWER(market.outcome_name) = 'under'
                        THEN 'under'
                    ELSE 'other'
                END AS outcome_type,
                market.price AS american_price,
                market.point,
                CASE
                    WHEN market.market_key = 'spreads'
                        THEN ABS(market.point)
                    WHEN market.market_key = 'totals'
                        THEN market.point
                    ELSE NULL
                END AS market_line,
                CASE
                    WHEN market.price >= 100
                        THEN 1.0 + market.price / 100.0
                    WHEN market.price <= -100
                        THEN 1.0
                            + 100.0 / ABS(market.price)
                    ELSE NULL
                END AS decimal_odds,
                CASE
                    WHEN market.price >= 100
                        THEN 100.0
                            / (market.price + 100.0)
                    WHEN market.price <= -100
                        THEN ABS(market.price)
                            / (ABS(market.price) + 100.0)
                    ELSE NULL
                END AS implied_probability
            FROM {MARKETS_FULL_NAME} AS market
            INNER JOIN {EVENTS_FULL_NAME} AS event
                ON market.snapshot_id = event.snapshot_id
               AND market.event_id = event.event_id
            INNER JOIN {SNAPSHOTS_FULL_NAME} AS snapshot
                ON market.snapshot_id = snapshot.snapshot_id
        ),

        market_probabilities AS (
            SELECT
                *,
                SUM(implied_probability) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        bookmaker_key,
                        market_key,
                        market_line
                ) AS market_probability_sum
            FROM base_outcomes
        )

        SELECT
            snapshot_id,
            fetched_at,
            event_id,
            commence_time,
            home_team,
            away_team,
            bookmaker_key,
            bookmaker_title,
            bookmaker_last_update,
            market_key,
            outcome_name,
            outcome_type,
            american_price,
            point,
            market_line,
            decimal_odds,
            implied_probability,
            market_probability_sum - 1.0
                AS bookmaker_margin,
            implied_probability
                / NULLIF(market_probability_sum, 0)
                AS no_vig_probability
        FROM market_probabilities
        """
    )

    logger.info(
        "Processed odds table created: %s.",
        TARGET_FULL_NAME,
    )


def validate_processed_odds_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate processed odds calculations."""

    source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {MARKETS_FULL_NAME}
        """
    ).fetchone()[0]

    target_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if target_count != source_count:
        raise RuntimeError(
            "Processed odds row count does not match "
            "the raw market outcome count."
        )

    invalid_calculation_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE decimal_odds IS NULL
           OR implied_probability IS NULL
           OR bookmaker_margin IS NULL
           OR no_vig_probability IS NULL
           OR decimal_odds <= 1.0
           OR implied_probability <= 0.0
           OR implied_probability >= 1.0
           OR no_vig_probability <= 0.0
           OR no_vig_probability >= 1.0
        """
    ).fetchone()[0]

    if invalid_calculation_count > 0:
        raise RuntimeError(
            "Processed odds contains invalid probability "
            "or price calculations."
        )

    invalid_no_vig_group_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                snapshot_id,
                event_id,
                bookmaker_key,
                market_key,
                market_line,
                SUM(no_vig_probability) AS probability_sum
            FROM {TARGET_FULL_NAME}
            GROUP BY
                snapshot_id,
                event_id,
                bookmaker_key,
                market_key,
                market_line
            HAVING ABS(probability_sum - 1.0) > 0.000001
        )
        """
    ).fetchone()[0]

    if invalid_no_vig_group_count > 0:
        raise RuntimeError(
            "No-vig probabilities do not sum to one "
            "within one or more markets."
        )

    logger.info(
        "Processed odds calculations validated: "
        "%s rows.",
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


def build_processed_odds(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the processed NFL odds table."""

    validate_database_file(database_file)

    logger.info("Starting processed odds build...")

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)

        try:
            connection.execute("BEGIN TRANSACTION")

            create_processed_odds_table(connection)
            validate_processed_odds_table(connection)

            connection.execute("COMMIT")

            logger.info(
                "Processed odds transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")
            logger.exception(
                "Processed odds transaction rolled back."
            )
            raise

    logger.info(
        "Processed odds build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the processed odds build workflow."""

    try:
        build_processed_odds()
    except Exception:
        logger.exception(
            "Processed odds build failed."
        )
        raise


if __name__ == "__main__":
    main()