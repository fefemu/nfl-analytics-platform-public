"""
NFL Analytics Platform
Processed Schedule Builder

Purpose:
    Build the processed NFL schedule dataset
    from the raw schedule table.

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
SOURCE_TABLE = "schedule"

TARGET_SCHEMA = "processed"
TARGET_TABLE = "schedule"

EXPECTED_TARGET_COLUMN_COUNT = 46

SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


def validate_database_file() -> None:
    """Validate that the DuckDB database file exists."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"DuckDB database does not exist: {DATABASE_FILE}"
        )

    logger.info("Database file validated: %s", DATABASE_FILE)


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate that the raw schedule table exists."""

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

    logger.info("Source table validated: %s", SOURCE_FULL_NAME)


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate required fields and the primary business key."""

    missing_required_fields = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR game_type IS NULL
           OR week IS NULL
           OR gameday IS NULL
           OR weekday IS NULL
           OR away_team IS NULL
           OR home_team IS NULL
           OR away_rest IS NULL
           OR home_rest IS NULL
        """
    ).fetchone()[0]

    if missing_required_fields > 0:
        raise RuntimeError(
            f"Found {missing_required_fields} records with "
            f"missing required fields."
        )

    duplicate_game_ids = connection.execute(
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

    if duplicate_game_ids > 0:
        raise RuntimeError(
            f"Found {duplicate_game_ids} duplicate game_id values."
        )

    logger.info(
        "Required fields and business keys validated successfully."
    )


def validate_game_results(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate fields derived from the final scores."""

    invalid_game_results = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE is_completed IS DISTINCT FROM (
                  home_score IS NOT NULL
                  AND away_score IS NOT NULL
              )
           OR home_win IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN NULL
                      ELSE home_score > away_score
                  END
              )
           OR away_win IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN NULL
                      ELSE away_score > home_score
                  END
              )
           OR is_tie IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN NULL
                      ELSE home_score = away_score
                  END
              )
           OR point_differential IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN NULL
                      ELSE home_score - away_score
                  END
              )
           OR total_points IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN NULL
                      ELSE home_score + away_score
                  END
              )
           OR game_result IS DISTINCT FROM (
                  CASE
                      WHEN home_score IS NULL OR away_score IS NULL
                          THEN 'NOT_PLAYED'
                      WHEN home_score > away_score THEN 'HOME_WIN'
                      WHEN away_score > home_score THEN 'AWAY_WIN'
                      ELSE 'TIE'
                  END
              )
        """
    ).fetchone()[0]

    if invalid_game_results > 0:
        raise RuntimeError(
            f"Found {invalid_game_results} records with "
            f"inconsistent derived game results."
        )

    logger.info("Derived game result fields validated successfully.")


def validate_additional_derived_fields(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate rest advantage and season classification fields."""

    invalid_derived_fields = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_rest_advantage IS DISTINCT FROM (
                  CASE
                      WHEN home_rest IS NULL OR away_rest IS NULL
                          THEN NULL
                      ELSE home_rest - away_rest
                  END
              )
           OR is_regular_season IS DISTINCT FROM (
                  game_type = 'REG'
              )
           OR is_playoff IS DISTINCT FROM (
                  game_type IN ('WC', 'DIV', 'CON', 'SB')
              )
        """
    ).fetchone()[0]

    if invalid_derived_fields > 0:
        raise RuntimeError(
            f"Found {invalid_derived_fields} records with "
            f"inconsistent additional derived fields."
        )

    logger.info(
        "Additional derived fields validated successfully."
    )


def build_processed_schedule() -> None:
    """Build the processed schedule dataset."""

    validate_database_file()

    logger.info("Starting processed schedule build...")

    with duckdb.connect(str(DATABASE_FILE)) as connection:
        validate_source_table(connection)
        connection.execute("BEGIN TRANSACTION")

        try:
            connection.execute(
                f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
            )

            connection.execute(
                f"""
                CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
                SELECT
                    CAST(game_id AS VARCHAR) AS game_id,
                    CAST(season AS INTEGER) AS season,
                    CAST(game_type AS VARCHAR) AS game_type,
                    CAST(week AS INTEGER) AS week,
                    CAST(gameday AS DATE) AS gameday,
                    CAST(gametime AS TIME) AS gametime,
                    CAST(weekday AS VARCHAR) AS weekday,
                    CAST(away_team AS VARCHAR) AS away_team,
                    CAST(home_team AS VARCHAR) AS home_team,
                    CAST(location AS VARCHAR) AS location,
                    CAST(away_score AS INTEGER) AS away_score,
                    CAST(home_score AS INTEGER) AS home_score,
                    CAST(overtime AS INTEGER) AS overtime,
                    CAST(away_rest AS INTEGER) AS away_rest,
                    CAST(home_rest AS INTEGER) AS home_rest,
                    CAST(away_moneyline AS INTEGER) AS away_moneyline,
                    CAST(home_moneyline AS INTEGER) AS home_moneyline,
                    CAST(spread_line AS DOUBLE) AS spread_line,
                    CAST(away_spread_odds AS INTEGER) AS away_spread_odds,
                    CAST(home_spread_odds AS INTEGER) AS home_spread_odds,
                    CAST(total_line AS DOUBLE) AS total_line,
                    CAST(under_odds AS INTEGER) AS under_odds,
                    CAST(over_odds AS INTEGER) AS over_odds,
                    CAST(roof AS VARCHAR) AS roof,
                    CAST(surface AS VARCHAR) AS surface,
                    CAST(temp AS INTEGER) AS temp,
                    CAST(wind AS INTEGER) AS wind,
                    CAST(away_qb_id AS VARCHAR) AS away_qb_id,
                    CAST(home_qb_id AS VARCHAR) AS home_qb_id,
                    CAST(away_qb_name AS VARCHAR) AS away_qb_name,
                    CAST(home_qb_name AS VARCHAR) AS home_qb_name,
                    CAST(away_coach AS VARCHAR) AS away_coach,
                    CAST(home_coach AS VARCHAR) AS home_coach,
                    CAST(referee AS VARCHAR) AS referee,
                    CAST(stadium_id AS VARCHAR) AS stadium_id,
                    CAST(stadium AS VARCHAR) AS stadium,
                    home_score IS NOT NULL
                        AND away_score IS NOT NULL AS is_completed,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL THEN NULL
                        ELSE home_score > away_score
                    END AS home_win,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL THEN NULL
                        ELSE away_score > home_score
                    END AS away_win,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL THEN NULL
                        ELSE home_score = away_score
                    END AS is_tie,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL THEN NULL
                        ELSE home_score - away_score
                    END AS point_differential,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL
                            THEN 'NOT_PLAYED'
                        WHEN home_score > away_score THEN 'HOME_WIN'
                        WHEN away_score > home_score THEN 'AWAY_WIN'
                        ELSE 'TIE'
                    END AS game_result,
                    CASE
                        WHEN home_rest IS NULL OR away_rest IS NULL THEN NULL
                        ELSE home_rest - away_rest
                    END AS home_rest_advantage,
                    CASE
                        WHEN home_score IS NULL OR away_score IS NULL THEN NULL
                        ELSE home_score + away_score
                    END AS total_points,
                    game_type = 'REG' AS is_regular_season,
                    game_type IN ('WC', 'DIV', 'CON', 'SB') AS is_playoff
                FROM {SOURCE_FULL_NAME}
                """
            )

            validate_target_table(connection)
            validate_game_results(connection)
            validate_additional_derived_fields(connection)

            source_row_count = connection.execute(
                f"SELECT COUNT(*) FROM {SOURCE_FULL_NAME}"
            ).fetchone()[0]

            target_row_count = connection.execute(
                f"SELECT COUNT(*) FROM {TARGET_FULL_NAME}"
            ).fetchone()[0]

            target_column_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = ?
                AND table_name = ?
                """,
                [TARGET_SCHEMA, TARGET_TABLE],
            ).fetchone()[0]

            if target_row_count != source_row_count:
                raise RuntimeError(
                    f"Row count mismatch: source={source_row_count}, "
                    f"target={target_row_count}"
                )

            if target_column_count != EXPECTED_TARGET_COLUMN_COUNT:
                raise RuntimeError(
                    f"Unexpected target column count: "
                    f"expected={EXPECTED_TARGET_COLUMN_COUNT}, "
                    f"actual={target_column_count}"
                )

        except Exception:
            connection.execute("ROLLBACK")
            logger.error(
                "Processed schedule transaction rolled back."
            )
            raise
        else:
            connection.execute("COMMIT")
            logger.info(
                "Processed schedule transaction committed."
            )

    logger.info(
        "Processed schedule created: %s rows and %s columns in %s.",
        target_row_count,
        target_column_count,
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the processed schedule build workflow."""

    try:
        build_processed_schedule()
    except Exception:
        logger.exception("Processed schedule build failed.")
        raise


if __name__ == "__main__":
    main()
