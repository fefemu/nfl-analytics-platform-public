"""
NFL Analytics Platform
Best Available Odds Builder

Purpose:
    Compare bookmakers for equivalent NFL betting
    outcomes and select the best available price.

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

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "odds_market_outcomes"

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "best_odds_by_line"

SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate that the processed odds table exists."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [SOURCE_SCHEMA, SOURCE_TABLE],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )

    logger.info(
        "Processed odds source validated: %s.",
        SOURCE_FULL_NAME,
    )


def create_best_odds_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the best bookmaker price for each line."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH ranked_offers AS (
            SELECT
                *,
                COUNT(*) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                ) AS bookmaker_count,
                AVG(no_vig_probability) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                ) AS consensus_no_vig_probability,
                MIN(no_vig_probability) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                ) AS minimum_no_vig_probability,
                MAX(no_vig_probability) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                ) AS maximum_no_vig_probability,
                AVG(decimal_odds) OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                ) AS average_decimal_odds,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        snapshot_id,
                        event_id,
                        market_key,
                        outcome_type,
                        point
                    ORDER BY
                        decimal_odds DESC,
                        bookmaker_key
                ) AS price_rank
            FROM {SOURCE_FULL_NAME}
        )

        SELECT
            snapshot_id,
            fetched_at,
            event_id,
            commence_time,
            home_team,
            away_team,
            market_key,
            outcome_name,
            outcome_type,
            point,
            market_line,
            bookmaker_key AS best_bookmaker_key,
            bookmaker_title AS best_bookmaker_title,
            american_price AS best_american_price,
            decimal_odds AS best_decimal_odds,
            implied_probability
                AS best_implied_probability,
            bookmaker_margin
                AS best_bookmaker_margin,
            no_vig_probability
                AS best_bookmaker_no_vig_probability,
            bookmaker_count,
            consensus_no_vig_probability,
            minimum_no_vig_probability,
            maximum_no_vig_probability,
            maximum_no_vig_probability
                - minimum_no_vig_probability
                AS probability_dispersion,
            average_decimal_odds,
            decimal_odds - average_decimal_odds
                AS decimal_price_improvement
        FROM ranked_offers
        WHERE price_rank = 1
        """
    )

    logger.info(
        "Best available odds table created: %s.",
        TARGET_FULL_NAME,
    )


def validate_best_odds_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate best-price selection and consensus metrics."""

    expected_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                snapshot_id,
                event_id,
                market_key,
                outcome_type,
                point
            FROM {SOURCE_FULL_NAME}
        )
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
            "Best odds row count does not match "
            "the distinct source offer groups."
        )

    incorrect_best_price_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME} AS target
        WHERE target.best_decimal_odds IS DISTINCT FROM (
            SELECT MAX(source.decimal_odds)
            FROM {SOURCE_FULL_NAME} AS source
            WHERE source.snapshot_id = target.snapshot_id
              AND source.event_id = target.event_id
              AND source.market_key = target.market_key
              AND source.outcome_type = target.outcome_type
              AND source.point IS NOT DISTINCT FROM target.point
        )
        """
    ).fetchone()[0]

    if incorrect_best_price_count > 0:
        raise RuntimeError(
            "One or more best odds rows do not contain "
            "the maximum available decimal price."
        )

    incorrect_bookmaker_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME} AS target
        WHERE target.bookmaker_count <> (
            SELECT COUNT(*)
            FROM {SOURCE_FULL_NAME} AS source
            WHERE source.snapshot_id = target.snapshot_id
              AND source.event_id = target.event_id
              AND source.market_key = target.market_key
              AND source.outcome_type = target.outcome_type
              AND source.point IS NOT DISTINCT FROM target.point
        )
        """
    ).fetchone()[0]

    if incorrect_bookmaker_count > 0:
        raise RuntimeError(
            "One or more bookmaker counts are incorrect."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE consensus_no_vig_probability <= 0.0
           OR consensus_no_vig_probability >= 1.0
           OR minimum_no_vig_probability <= 0.0
           OR maximum_no_vig_probability >= 1.0
           OR probability_dispersion < 0.0
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Best odds table contains invalid "
            "consensus probability metrics."
        )

    logger.info(
        "Best available odds validated: %s rows.",
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


def build_best_odds(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the best available NFL odds table."""

    validate_database_file(database_file)

    logger.info("Starting best available odds build...")

    with duckdb.connect(str(database_file)) as connection:
        validate_source_table(connection)

        try:
            connection.execute("BEGIN TRANSACTION")

            create_best_odds_table(connection)
            validate_best_odds_table(connection)

            connection.execute("COMMIT")

            logger.info(
                "Best available odds transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")
            logger.exception(
                "Best available odds transaction rolled back."
            )
            raise

    logger.info(
        "Best available odds build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the best available odds build workflow."""

    try:
        build_best_odds()
    except Exception:
        logger.exception(
            "Best available odds build failed."
        )
        raise


if __name__ == "__main__":
    main()
