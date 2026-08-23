"""
NFL Analytics Platform
Game Quarterback Features Builder

Purpose:
    Build leakage-safe pregame quarterback features
    and a separate postgame quarterback audit table.

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


# ---------------------------------------------------------
# Source tables
# ---------------------------------------------------------

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

QB_PERFORMANCE_SCHEMA = "processed"
QB_PERFORMANCE_TABLE = "qb_game_performance"
QB_PERFORMANCE_FULL_NAME = (
    f"{QB_PERFORMANCE_SCHEMA}.{QB_PERFORMANCE_TABLE}"
)

QB_RATING_SCHEMA = "analytics"
QB_RATING_TABLE = "qb_rating_history"
QB_RATING_FULL_NAME = (
    f"{QB_RATING_SCHEMA}.{QB_RATING_TABLE}"
)


# ---------------------------------------------------------
# Target tables
# ---------------------------------------------------------

TARGET_SCHEMA = "analytics"

FEATURE_TABLE = "game_qb_features"
FEATURE_FULL_NAME = (
    f"{TARGET_SCHEMA}.{FEATURE_TABLE}"
)

AUDIT_TABLE = "game_qb_audit"
AUDIT_FULL_NAME = (
    f"{TARGET_SCHEMA}.{AUDIT_TABLE}"
)


# ---------------------------------------------------------
# Required source columns
# ---------------------------------------------------------

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_qb_id",
    "home_qb_name",
    "away_qb_id",
    "away_qb_name",
    "is_completed",
}

REQUIRED_QB_PERFORMANCE_COLUMNS = {
    "game_id",
    "game_date",
    "team",
    "opponent",
    "qb_id",
    "qb_name",
    "is_primary_qb",
    "is_listed_starter",
    "dropbacks",
}

REQUIRED_QB_RATING_COLUMNS = {
    "game_id",
    "game_date",
    "team",
    "opponent",
    "qb_id",
    "qb_name",
    "pregame_effective_dropbacks",
    "pregame_qb_rating",
    "pregame_prior_weight",
    "pregame_rating_standard_error",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

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


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the game QB feature source tables."""

    required_tables = {
        (
            SCHEDULE_SCHEMA,
            SCHEDULE_TABLE,
        ): REQUIRED_SCHEDULE_COLUMNS,
        (
            QB_PERFORMANCE_SCHEMA,
            QB_PERFORMANCE_TABLE,
        ): REQUIRED_QB_PERFORMANCE_COLUMNS,
        (
            QB_RATING_SCHEMA,
            QB_RATING_TABLE,
        ): REQUIRED_QB_RATING_COLUMNS,
    }

    for (
        schema_name,
        table_name,
    ), required_columns in required_tables.items():
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [schema_name, table_name],
        ).fetchone()[0]

        full_name = f"{schema_name}.{table_name}"

        if table_exists == 0:
            raise RuntimeError(
                f"Source table does not exist: {full_name}"
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
                [schema_name, table_name],
            ).fetchall()
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            missing_names = ", ".join(missing_columns)

            raise RuntimeError(
                f"Missing columns in {full_name}: "
                f"{missing_names}"
            )

    logger.info(
        "Game QB feature sources validated: %s, %s and %s.",
        SCHEDULE_FULL_NAME,
        QB_PERFORMANCE_FULL_NAME,
        QB_RATING_FULL_NAME,
    )


def create_game_qb_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create leakage-safe pregame quarterback features."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {FEATURE_FULL_NAME} AS

        WITH eligible_games AS (
            SELECT
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                CAST(schedule.gameday AS DATE) AS game_date,

                schedule.home_team,
                schedule.away_team,

                CASE schedule.home_team
                    WHEN 'OAK' THEN 'LV'
                    WHEN 'SD' THEN 'LAC'
                    WHEN 'STL' THEN 'LA'
                    ELSE schedule.home_team
                END AS home_team_key,

                CASE schedule.away_team
                    WHEN 'OAK' THEN 'LV'
                    WHEN 'SD' THEN 'LAC'
                    WHEN 'STL' THEN 'LA'
                    ELSE schedule.away_team
                END AS away_team_key,

                schedule.home_qb_id
                    AS home_listed_qb_id,

                schedule.home_qb_name
                    AS home_listed_qb_name,

                schedule.away_qb_id
                    AS away_listed_qb_id,

                schedule.away_qb_name
                    AS away_listed_qb_name

            FROM {SCHEDULE_FULL_NAME} AS schedule

            WHERE schedule.is_completed = TRUE

              AND EXISTS (
                    SELECT 1
                    FROM {QB_PERFORMANCE_FULL_NAME}
                        AS qb_performance
                    WHERE qb_performance.game_id
                        = schedule.game_id
              )
        )

        SELECT
            games.game_id,
            games.season,
            games.game_type,
            games.week,
            games.game_date,

            games.home_team,
            games.away_team,

            games.home_listed_qb_id,
            games.home_listed_qb_name,

            home_rating.pregame_qb_rating
                AS home_listed_qb_rating,

            home_rating.pregame_effective_dropbacks
                AS home_listed_qb_effective_dropbacks,

            home_rating.pregame_prior_weight
                AS home_listed_qb_prior_weight,

            home_rating.pregame_rating_standard_error
                AS home_listed_qb_rating_standard_error,

            home_rating.qb_id IS NOT NULL
                AS home_listed_qb_rating_available,

            games.away_listed_qb_id,
            games.away_listed_qb_name,

            away_rating.pregame_qb_rating
                AS away_listed_qb_rating,

            away_rating.pregame_effective_dropbacks
                AS away_listed_qb_effective_dropbacks,

            away_rating.pregame_prior_weight
                AS away_listed_qb_prior_weight,

            away_rating.pregame_rating_standard_error
                AS away_listed_qb_rating_standard_error,

            away_rating.qb_id IS NOT NULL
                AS away_listed_qb_rating_available,

            (
                home_rating.qb_id IS NOT NULL
                AND away_rating.qb_id IS NOT NULL
            ) AS both_listed_qb_ratings_available,

            CASE
                WHEN home_rating.qb_id IS NOT NULL
                 AND away_rating.qb_id IS NOT NULL
                THEN (
                    home_rating.pregame_qb_rating
                    - away_rating.pregame_qb_rating
                )
                ELSE NULL
            END AS listed_qb_rating_difference,

            CASE
                WHEN home_rating.qb_id IS NOT NULL
                 AND away_rating.qb_id IS NOT NULL
                THEN SQRT(
                    POWER(
                        home_rating
                            .pregame_rating_standard_error,
                        2
                    )
                    + POWER(
                        away_rating
                            .pregame_rating_standard_error,
                        2
                    )
                )
                ELSE NULL
            END AS listed_qb_rating_difference_standard_error

        FROM eligible_games AS games

        LEFT JOIN {QB_RATING_FULL_NAME}
            AS home_rating
            ON games.game_id = home_rating.game_id
           AND games.home_team_key = home_rating.team
           AND games.home_listed_qb_id
                = home_rating.qb_id

        LEFT JOIN {QB_RATING_FULL_NAME}
            AS away_rating
            ON games.game_id = away_rating.game_id
           AND games.away_team_key = away_rating.team
           AND games.away_listed_qb_id
                = away_rating.qb_id

        ORDER BY
            games.game_date,
            games.game_id
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Pregame QB features created: %s rows in %s.",
        row_count,
        FEATURE_FULL_NAME,
    )


def create_game_qb_audit_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the postgame quarterback audit table."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {AUDIT_FULL_NAME} AS

        WITH primary_quarterbacks AS (
            SELECT
                performance.game_id,
                performance.game_date,
                performance.team,
                performance.opponent,
                performance.qb_id
                    AS actual_primary_qb_id,
                performance.qb_name
                    AS actual_primary_qb_name,
                performance.dropbacks
                    AS actual_primary_qb_dropbacks

            FROM {QB_PERFORMANCE_FULL_NAME}
                AS performance

            WHERE performance.is_primary_qb = TRUE
        )

        SELECT
            features.game_id,
            features.season,
            features.game_type,
            features.week,
            features.game_date,

            features.home_team,
            features.away_team,

            features.home_listed_qb_id,
            features.home_listed_qb_name,

            home_primary.actual_primary_qb_id
                AS home_actual_primary_qb_id,

            home_primary.actual_primary_qb_name
                AS home_actual_primary_qb_name,

            home_primary.actual_primary_qb_dropbacks
                AS home_actual_primary_qb_dropbacks,

            (
                features.home_listed_qb_id
                = home_primary.actual_primary_qb_id
            ) AS home_listed_qb_matches_actual_primary,

            home_primary_rating.pregame_qb_rating
                AS home_actual_primary_qb_pregame_rating,

            home_primary_rating
                .pregame_effective_dropbacks
                AS home_actual_primary_qb_effective_dropbacks,

            features.away_listed_qb_id,
            features.away_listed_qb_name,

            away_primary.actual_primary_qb_id
                AS away_actual_primary_qb_id,

            away_primary.actual_primary_qb_name
                AS away_actual_primary_qb_name,

            away_primary.actual_primary_qb_dropbacks
                AS away_actual_primary_qb_dropbacks,

            (
                features.away_listed_qb_id
                = away_primary.actual_primary_qb_id
            ) AS away_listed_qb_matches_actual_primary,

            away_primary_rating.pregame_qb_rating
                AS away_actual_primary_qb_pregame_rating,

            away_primary_rating
                .pregame_effective_dropbacks
                AS away_actual_primary_qb_effective_dropbacks,

            (
                features.home_listed_qb_id
                    = home_primary.actual_primary_qb_id
                AND features.away_listed_qb_id
                    = away_primary.actual_primary_qb_id
            ) AS both_listed_qbs_match_actual_primary,

            CASE
                WHEN home_primary_rating.qb_id IS NOT NULL
                 AND away_primary_rating.qb_id IS NOT NULL
                THEN (
                    home_primary_rating.pregame_qb_rating
                    - away_primary_rating.pregame_qb_rating
                )
                ELSE NULL
            END AS actual_primary_qb_rating_difference

        FROM {FEATURE_FULL_NAME} AS features

        LEFT JOIN primary_quarterbacks
            AS home_primary
            ON features.game_id = home_primary.game_id
           AND (
                CASE features.home_team
                    WHEN 'OAK' THEN 'LV'
                    WHEN 'SD' THEN 'LAC'
                    WHEN 'STL' THEN 'LA'
                    ELSE features.home_team
                END
           ) = home_primary.team

        LEFT JOIN primary_quarterbacks
            AS away_primary
            ON features.game_id = away_primary.game_id
           AND (
                CASE features.away_team
                    WHEN 'OAK' THEN 'LV'
                    WHEN 'SD' THEN 'LAC'
                    WHEN 'STL' THEN 'LA'
                    ELSE features.away_team
                END
           ) = away_primary.team

        LEFT JOIN {QB_RATING_FULL_NAME}
            AS home_primary_rating
            ON features.game_id
                = home_primary_rating.game_id
           AND home_primary.team
                = home_primary_rating.team
           AND home_primary.actual_primary_qb_id
                = home_primary_rating.qb_id

        LEFT JOIN {QB_RATING_FULL_NAME}
            AS away_primary_rating
            ON features.game_id
                = away_primary_rating.game_id
           AND away_primary.team
                = away_primary_rating.team
           AND away_primary.actual_primary_qb_id
                = away_primary_rating.qb_id

        ORDER BY
            features.game_date,
            features.game_id
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {AUDIT_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Postgame QB audit created: %s rows in %s.",
        row_count,
        AUDIT_FULL_NAME,
    )


def validate_game_qb_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the leakage-safe game QB feature table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "The game QB feature table is empty."
        )

    expected_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SCHEDULE_FULL_NAME} AS schedule
        WHERE schedule.is_completed = TRUE
          AND EXISTS (
                SELECT 1
                FROM {QB_PERFORMANCE_FULL_NAME}
                    AS performance
                WHERE performance.game_id
                    = schedule.game_id
          )
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Game QB feature row count does not match: "
            f"expected {expected_row_count}, "
            f"found {row_count}."
        )

    duplicate_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {FEATURE_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate games found in the game QB features: "
            f"{duplicate_game_count}"
        )

    invalid_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        WHERE home_team IS NULL
           OR away_team IS NULL
           OR home_team = away_team
        """
    ).fetchone()[0]

    if invalid_team_count > 0:
        raise RuntimeError(
            "Invalid team assignments found in game QB features: "
            f"{invalid_team_count}"
        )

    invalid_availability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        WHERE
            home_listed_qb_rating_available
                <> (home_listed_qb_rating IS NOT NULL)

           OR away_listed_qb_rating_available
                <> (away_listed_qb_rating IS NOT NULL)

           OR both_listed_qb_ratings_available
                <> (
                    home_listed_qb_rating IS NOT NULL
                    AND away_listed_qb_rating IS NOT NULL
                )
        """
    ).fetchone()[0]

    if invalid_availability_count > 0:
        raise RuntimeError(
            "Invalid QB rating availability flags found: "
            f"{invalid_availability_count}"
        )

    invalid_difference_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        WHERE
            (
                both_listed_qb_ratings_available = TRUE
                AND (
                    listed_qb_rating_difference IS NULL
                    OR ABS(
                        listed_qb_rating_difference
                        - (
                            home_listed_qb_rating
                            - away_listed_qb_rating
                        )
                    ) > 0.000000001
                )
            )

            OR (
                both_listed_qb_ratings_available = FALSE
                AND (
                    listed_qb_rating_difference IS NOT NULL
                    OR listed_qb_rating_difference_standard_error
                        IS NOT NULL
                )
            )

            OR (
                listed_qb_rating_difference_standard_error
                    < 0
            )
        """
    ).fetchone()[0]

    if invalid_difference_count > 0:
        raise RuntimeError(
            "Invalid QB rating differences found: "
            f"{invalid_difference_count}"
        )

    forbidden_column_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
          AND (
                column_name LIKE '%actual_primary%'
                OR column_name LIKE '%postgame%'
                OR column_name LIKE '%final_score%'
          )
        """,
        [TARGET_SCHEMA, FEATURE_TABLE],
    ).fetchone()[0]

    if forbidden_column_count > 0:
        raise RuntimeError(
            "Postgame information was found in the "
            "pregame QB feature table."
        )

    logger.info(
        "Pregame QB features validated successfully: %s rows.",
        row_count,
    )


def validate_game_qb_audit_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the postgame quarterback audit table."""

    feature_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FEATURE_FULL_NAME}
        """
    ).fetchone()[0]

    audit_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {AUDIT_FULL_NAME}
        """
    ).fetchone()[0]

    if audit_count != feature_count:
        raise RuntimeError(
            "Game QB audit row count does not match: "
            f"expected {feature_count}, "
            f"found {audit_count}."
        )

    duplicate_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {AUDIT_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate games found in the game QB audit: "
            f"{duplicate_game_count}"
        )

    missing_primary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {AUDIT_FULL_NAME}
        WHERE home_actual_primary_qb_id IS NULL
           OR away_actual_primary_qb_id IS NULL
        """
    ).fetchone()[0]

    if missing_primary_count > 0:
        raise RuntimeError(
            "Games without both actual primary QBs found: "
            f"{missing_primary_count}"
        )

    invalid_match_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {AUDIT_FULL_NAME}
        WHERE
            home_listed_qb_matches_actual_primary
                <> (
                    home_listed_qb_id
                    = home_actual_primary_qb_id
                )

           OR away_listed_qb_matches_actual_primary
                <> (
                    away_listed_qb_id
                    = away_actual_primary_qb_id
                )

           OR both_listed_qbs_match_actual_primary
                <> (
                    home_listed_qb_id
                        = home_actual_primary_qb_id
                    AND away_listed_qb_id
                        = away_actual_primary_qb_id
                )
        """
    ).fetchone()[0]

    if invalid_match_flag_count > 0:
        raise RuntimeError(
            "Invalid listed-versus-actual QB match flags found: "
            f"{invalid_match_flag_count}"
        )

    logger.info(
        "Postgame QB audit validated successfully: %s rows.",
        audit_count,
    )


def build_game_qb_features(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build pregame QB features and postgame QB audit data."""

    validate_database_file(database_file)

    logger.info(
        "Starting game QB features build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)

        connection.execute("BEGIN TRANSACTION")

        try:
            create_game_qb_features_table(connection)
            create_game_qb_audit_table(connection)

            validate_game_qb_features_table(connection)
            validate_game_qb_audit_table(connection)

            connection.execute("COMMIT")

            logger.info(
                "Game QB features transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Game QB features build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Game QB features build completed: %s and %s.",
        FEATURE_FULL_NAME,
        AUDIT_FULL_NAME,
    )


def main() -> None:
    """Run the game quarterback features builder."""

    try:
        build_game_qb_features()

    except Exception:
        logger.exception(
            "Game QB features builder failed."
        )
        raise


if __name__ == "__main__":
    main()