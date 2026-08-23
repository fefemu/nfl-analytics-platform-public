"""
NFL Analytics Platform
Odds Event Schedule Bridge Builder

Purpose:
    Match Odds API NFL events to nflverse schedule
    games using team mappings and local game dates.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb

from src.config.nfl_team_mappings import (
    ODDS_TEAM_TO_NFLVERSE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

ODDS_SCHEMA = "raw"
ODDS_TABLE = "odds_events"

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "odds_event_schedule_bridge"

ODDS_FULL_NAME = f"{ODDS_SCHEMA}.{ODDS_TABLE}"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the odds events and schedule tables."""

    required_tables = {
        (ODDS_SCHEMA, ODDS_TABLE),
        (SCHEDULE_SCHEMA, SCHEDULE_TABLE),
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
            f"Missing event bridge source tables: {missing_names}"
        )

    logger.info(
        "Event bridge sources validated: %s and %s.",
        ODDS_FULL_NAME,
        SCHEDULE_FULL_NAME,
    )


def create_team_mapping_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a temporary Odds API team mapping table."""

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE odds_team_mapping (
            odds_team_name VARCHAR PRIMARY KEY,
            nflverse_team_code VARCHAR NOT NULL
        )
        """
    )

    connection.executemany(
        """
        INSERT INTO odds_team_mapping
        VALUES (?, ?)
        """,
        list(ODDS_TEAM_TO_NFLVERSE.items()),
    )


def create_event_bridge_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Match Odds API events to nflverse games."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH mapped_events AS (
            SELECT
                event.snapshot_id,
                event.event_id AS odds_event_id,
                event.commence_time,
                CAST(
                    event.commence_time
                        AT TIME ZONE 'America/New_York'
                    AS DATE
                ) AS eastern_game_date,
                event.home_team AS odds_home_team,
                event.away_team AS odds_away_team,
                home_mapping.nflverse_team_code
                    AS home_team_code,
                away_mapping.nflverse_team_code
                    AS away_team_code
            FROM {ODDS_FULL_NAME} AS event
            LEFT JOIN odds_team_mapping AS home_mapping
                ON event.home_team
                    = home_mapping.odds_team_name
            LEFT JOIN odds_team_mapping AS away_mapping
                ON event.away_team
                    = away_mapping.odds_team_name
        )

        SELECT
            mapped.snapshot_id,
            mapped.odds_event_id,
            mapped.commence_time,
            mapped.eastern_game_date,
            mapped.odds_home_team,
            mapped.odds_away_team,
            mapped.home_team_code,
            mapped.away_team_code,
            schedule.game_id,
            schedule.season,
            schedule.game_type,
            schedule.week,
            schedule.gameday,
            schedule.gametime,
            CASE
                WHEN mapped.home_team_code IS NULL
                  OR mapped.away_team_code IS NULL
                    THEN 'UNMAPPED_TEAM'
                WHEN schedule.game_id IS NULL
                    THEN 'UNMATCHED_GAME'
                ELSE 'MATCHED'
            END AS match_status
        FROM mapped_events AS mapped
        LEFT JOIN {SCHEDULE_FULL_NAME} AS schedule
            ON mapped.home_team_code = schedule.home_team
           AND mapped.away_team_code = schedule.away_team
           AND mapped.eastern_game_date = schedule.gameday
        """
    )

    logger.info(
        "Odds event schedule bridge created: %s.",
        TARGET_FULL_NAME,
    )


def validate_event_bridge(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate Odds API to schedule event matching."""

    source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {ODDS_FULL_NAME}
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
            "Event bridge row count does not match "
            "the raw odds event count."
        )

    unmapped_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE match_status = 'UNMAPPED_TEAM'
        """
    ).fetchone()[0]

    if unmapped_team_count > 0:
        raise RuntimeError(
            "Event bridge contains unknown Odds API teams."
        )

    unmatched_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE match_status = 'UNMATCHED_GAME'
        """
    ).fetchone()[0]

    if unmatched_game_count > 0:
        raise RuntimeError(
            "Event bridge contains Odds API events "
            "without a schedule match."
        )

    duplicate_odds_event_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                snapshot_id,
                odds_event_id
            FROM {TARGET_FULL_NAME}
            GROUP BY
                snapshot_id,
                odds_event_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_odds_event_count > 0:
        raise RuntimeError(
            "Event bridge contains duplicate Odds API events."
        )

    duplicate_schedule_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                snapshot_id,
                game_id
            FROM {TARGET_FULL_NAME}
            WHERE game_id IS NOT NULL
            GROUP BY
                snapshot_id,
                game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_schedule_game_count > 0:
        raise RuntimeError(
            "Multiple Odds API events matched the same "
            "schedule game within one snapshot."
        )

    logger.info(
        "Odds event schedule bridge validated: "
        "%s matched events.",
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


def build_odds_event_bridge(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the Odds API to schedule event bridge."""

    validate_database_file(database_file)

    logger.info(
        "Starting Odds API event bridge build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)
        create_team_mapping_table(connection)

        try:
            connection.execute("BEGIN TRANSACTION")

            create_event_bridge_table(connection)
            validate_event_bridge(connection)

            connection.execute("COMMIT")

            logger.info(
                "Odds event bridge transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")
            logger.exception(
                "Odds event bridge transaction rolled back."
            )
            raise

    logger.info(
        "Odds event bridge build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the Odds API event bridge workflow."""

    try:
        build_odds_event_bridge()
    except Exception:
        logger.exception(
            "Odds event bridge build failed."
        )
        raise


if __name__ == "__main__":
    main()
