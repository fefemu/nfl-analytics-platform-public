"""
NFL Analytics Platform
Game Schedule Feature Builder

Purpose:
    Build leakage-safe pregame rest and schedule
    context features for each NFL game.

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
SOURCE_TABLE = "schedule"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "game_schedule_features"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SHORT_WEEK_MAX_REST_DAYS = 6
EXTENDED_REST_MIN_DAYS = 9
POST_BYE_MIN_REST_DAYS = 13

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_rest",
    "away_rest",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the processed schedule source."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [
            SOURCE_SCHEMA,
            SOURCE_TABLE,
        ],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [
                SOURCE_SCHEMA,
                SOURCE_TABLE,
            ],
        ).fetchall()
    }

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing columns in {SOURCE_FULL_NAME}: "
            + ", ".join(missing_columns)
        )

    source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if source_count == 0:
        raise RuntimeError(
            f"Source table is empty: {SOURCE_FULL_NAME}"
        )

    invalid_rest_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        WHERE home_rest IS NULL
           OR away_rest IS NULL
           OR home_rest < 0
           OR away_rest < 0
        """
    ).fetchone()[0]

    if invalid_rest_count > 0:
        raise RuntimeError(
            "Invalid rest values found in processed schedule: "
            f"{invalid_rest_count}"
        )

    logger.info(
        "Schedule feature source validated: %s rows.",
        source_count,
    )


def create_game_schedule_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create pregame rest and schedule features."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        SELECT
            game_id,
            season,
            game_type,
            week,
            CAST(gameday AS DATE) AS game_date,
            home_team,
            away_team,

            CAST(home_rest AS INTEGER)
                AS home_rest_days,

            CAST(away_rest AS INTEGER)
                AS away_rest_days,

            CAST(home_rest AS INTEGER)
                - CAST(away_rest AS INTEGER)
                AS rest_days_difference,

            home_rest <= {SHORT_WEEK_MAX_REST_DAYS}
                AS home_short_week,

            away_rest <= {SHORT_WEEK_MAX_REST_DAYS}
                AS away_short_week,

            CAST(
                home_rest <= {SHORT_WEEK_MAX_REST_DAYS}
                AS INTEGER
            )
            - CAST(
                away_rest <= {SHORT_WEEK_MAX_REST_DAYS}
                AS INTEGER
            ) AS short_week_difference,

            home_rest >= {EXTENDED_REST_MIN_DAYS}
                AS home_extended_rest,

            away_rest >= {EXTENDED_REST_MIN_DAYS}
                AS away_extended_rest,

            CAST(
                home_rest >= {EXTENDED_REST_MIN_DAYS}
                AS INTEGER
            )
            - CAST(
                away_rest >= {EXTENDED_REST_MIN_DAYS}
                AS INTEGER
            ) AS extended_rest_difference,

            home_rest >= {POST_BYE_MIN_REST_DAYS}
                AS home_post_bye,

            away_rest >= {POST_BYE_MIN_REST_DAYS}
                AS away_post_bye,

            CAST(
                home_rest >= {POST_BYE_MIN_REST_DAYS}
                AS INTEGER
            )
            - CAST(
                away_rest >= {POST_BYE_MIN_REST_DAYS}
                AS INTEGER
            ) AS post_bye_difference

        FROM {SOURCE_FULL_NAME}

        ORDER BY
            game_date,
            game_id
        """
    )


def validate_game_schedule_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate schedule-feature output."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    expected_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Schedule feature row count does not match: "
            f"expected {expected_row_count}, "
            f"found {row_count}."
        )

    duplicate_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate schedule-feature games found: "
            f"{duplicate_game_count}"
        )

    invalid_feature_count = connection.execute(
        f"""
        SELECT COUNT(*)

        FROM {TARGET_FULL_NAME} AS features

        INNER JOIN {SOURCE_FULL_NAME} AS schedule
            ON features.game_id = schedule.game_id

        WHERE features.home_rest_days
                IS DISTINCT FROM schedule.home_rest

           OR features.away_rest_days
                IS DISTINCT FROM schedule.away_rest

           OR features.rest_days_difference
                IS DISTINCT FROM (
                    schedule.home_rest
                    - schedule.away_rest
                )

           OR features.home_short_week
                IS DISTINCT FROM (
                    schedule.home_rest
                    <= {SHORT_WEEK_MAX_REST_DAYS}
                )

           OR features.away_short_week
                IS DISTINCT FROM (
                    schedule.away_rest
                    <= {SHORT_WEEK_MAX_REST_DAYS}
                )

           OR features.home_extended_rest
                IS DISTINCT FROM (
                    schedule.home_rest
                    >= {EXTENDED_REST_MIN_DAYS}
                )

           OR features.away_extended_rest
                IS DISTINCT FROM (
                    schedule.away_rest
                    >= {EXTENDED_REST_MIN_DAYS}
                )

           OR features.home_post_bye
                IS DISTINCT FROM (
                    schedule.home_rest
                    >= {POST_BYE_MIN_REST_DAYS}
                )

           OR features.away_post_bye
                IS DISTINCT FROM (
                    schedule.away_rest
                    >= {POST_BYE_MIN_REST_DAYS}
                )

           OR features.short_week_difference
                IS DISTINCT FROM (
                    CAST(
                        schedule.home_rest
                            <= {SHORT_WEEK_MAX_REST_DAYS}
                        AS INTEGER
                    )
                    - CAST(
                        schedule.away_rest
                            <= {SHORT_WEEK_MAX_REST_DAYS}
                        AS INTEGER
                    )
                )

           OR features.extended_rest_difference
                IS DISTINCT FROM (
                    CAST(
                        schedule.home_rest
                            >= {EXTENDED_REST_MIN_DAYS}
                        AS INTEGER
                    )
                    - CAST(
                        schedule.away_rest
                            >= {EXTENDED_REST_MIN_DAYS}
                        AS INTEGER
                    )
                )

           OR features.post_bye_difference
                IS DISTINCT FROM (
                    CAST(
                        schedule.home_rest
                            >= {POST_BYE_MIN_REST_DAYS}
                        AS INTEGER
                    )
                    - CAST(
                        schedule.away_rest
                            >= {POST_BYE_MIN_REST_DAYS}
                        AS INTEGER
                    )
                )
        """
    ).fetchone()[0]

    if invalid_feature_count > 0:
        raise RuntimeError(
            "Invalid schedule features found: "
            f"{invalid_feature_count}"
        )

    logger.info(
        "Game schedule features validated: %s rows.",
        row_count,
    )


def build_game_schedule_features(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build leakage-safe game schedule features."""

    validate_database_file(database_file)

    logger.info(
        "Starting game schedule feature build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_source_table(connection)

        connection.execute("BEGIN TRANSACTION")

        try:
            create_game_schedule_features_table(
                connection
            )

            validate_game_schedule_features_table(
                connection
            )

            connection.execute("COMMIT")

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Game schedule feature build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Game schedule feature build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the game schedule feature builder."""

    try:
        build_game_schedule_features()

    except Exception:
        logger.exception(
            "Game schedule feature builder failed."
        )
        raise


if __name__ == "__main__":
    main()