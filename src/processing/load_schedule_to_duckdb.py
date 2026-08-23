"""
NFL Analytics Platform
Schedule DuckDB Loader

Purpose:
    Load the raw NFL schedule dataset into the local
    DuckDB analytical database.

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

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_FILE = PROJECT_ROOT / "data" / "raw" / "schedules.parquet"
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

SCHEMA_NAME = "raw"
TABLE_NAME = "schedule"
FULL_TABLE_NAME = f"{SCHEMA_NAME}.{TABLE_NAME}"


def validate_source_file() -> None:
    """Check whether the raw schedule file exists."""

    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(
            f"Schedule file does not exist: {SCHEDULE_FILE}"
        )


def load_schedule_to_duckdb() -> None:
    """Create or replace the raw schedule table in DuckDB."""

    validate_source_file()

    logger.info("Loading NFL schedule data into DuckDB...")

    try:
        with duckdb.connect(str(DATABASE_FILE)) as connection:
            connection.execute(
                f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"
            )

            connection.execute(
                f"""
                CREATE OR REPLACE TABLE {FULL_TABLE_NAME} AS
                SELECT *
                FROM read_parquet(?)
                """,
                [str(SCHEDULE_FILE)],
            )

            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {FULL_TABLE_NAME}"
            ).fetchone()[0]

            column_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                """,
                [SCHEMA_NAME, TABLE_NAME],
            ).fetchone()[0]

    except Exception:
        logger.exception("Failed to load schedule data into DuckDB.")
        raise

    logger.info(
        "DuckDB load completed: %s rows and %s columns.",
        row_count,
        column_count,
    )
    logger.info("Database saved to: %s", DATABASE_FILE)
    logger.info("Table created: %s", FULL_TABLE_NAME)


def main() -> None:
    """Run the DuckDB loading workflow."""

    load_schedule_to_duckdb()


if __name__ == "__main__":
    main()
