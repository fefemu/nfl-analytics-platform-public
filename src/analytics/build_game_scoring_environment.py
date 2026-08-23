"""
NFL Analytics Platform
Game Scoring Environment Builder

Purpose:
    Create leakage-safe league scoring-environment
    features from completed games before each game date.

Games played on the same date never use each other's
results.

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

DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "nfl_analytics.duckdb"
)

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "schedule"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = (
    "game_scoring_environment_features"
)
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SCORING_WINDOWS = (
    32,
    64,
    128,
)

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "is_completed",
    "total_points",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate the DuckDB database path."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: "
            f"{database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: "
            f"{database_file}"
        )


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> set[str]:
    """Return available table columns."""

    return {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [
                schema_name,
                table_name,
            ],
        ).fetchall()
    }


def validate_scoring_environment_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the processed schedule source."""

    available_columns = get_table_columns(
        connection=connection,
        schema_name=SOURCE_SCHEMA,
        table_name=SOURCE_TABLE,
    )

    if not available_columns:
        raise RuntimeError(
            "Scoring environment source table "
            f"does not exist: {SOURCE_FULL_NAME}"
        )

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Scoring environment source is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if source_row_count == 0:
        raise RuntimeError(
            "Scoring environment source is empty."
        )

    logger.info(
        "Scoring environment source validated: "
        "%s rows in %s.",
        source_row_count,
        SOURCE_FULL_NAME,
    )


def create_window_lateral_sql(
    window_size: int,
) -> str:
    """Create one correlated historical window."""

    if window_size <= 0:
        raise ValueError(
            "Scoring window must be positive."
        )

    return f"""
        LEFT JOIN LATERAL (

            SELECT
                COUNT(*) AS
                    league_game_count_last_{window_size},

                AVG(total_points) AS
                    league_average_total_last_{window_size},

                STDDEV_SAMP(total_points) AS
                    league_total_standard_deviation_last_{window_size}

            FROM (
                SELECT
                    history.total_points

                FROM {SOURCE_FULL_NAME}
                    AS history

                WHERE history.is_completed = TRUE
                  AND history.total_points IS NOT NULL
                  AND history.game_type IN (
                        'REG',
                        'POST'
                      )
                  AND CAST(history.gameday AS DATE)
                        < CAST(
                            target_game.gameday
                            AS DATE
                          )

                ORDER BY
                    CAST(history.gameday AS DATE)
                        DESC,
                    history.game_id DESC

                LIMIT {window_size}
            ) AS recent_games

        ) AS environment_{window_size}
            ON TRUE
    """.strip()


def create_game_scoring_environment_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create league scoring-environment features."""

    lateral_sql = "\n\n".join(
        create_window_lateral_sql(
            window_size
        )
        for window_size in SCORING_WINDOWS
    )

    environment_columns = ",\n            ".join(
        (
            f"environment_{window_size}."
            f"league_game_count_last_{window_size},\n"
            f"            environment_{window_size}."
            f"league_average_total_last_{window_size},\n"
            f"            environment_{window_size}."
            f"league_total_standard_deviation_last_"
            f"{window_size}"
        )
        for window_size in SCORING_WINDOWS
    )

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        SELECT
            target_game.game_id,
            target_game.season,
            target_game.game_type,
            CAST(target_game.gameday AS DATE)
                AS game_date,
            target_game.home_team,
            target_game.away_team,
            {environment_columns}

        FROM {SOURCE_FULL_NAME} AS target_game

        {lateral_sql}

        ORDER BY
            target_game.gameday,
            target_game.game_id
        """
    )

    logger.info(
        "Game scoring environment table created: %s.",
        TARGET_FULL_NAME,
    )


def validate_game_scoring_environment(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate league scoring-environment features."""

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    target_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if target_row_count != source_row_count:
        raise RuntimeError(
            "Scoring environment row count does not "
            f"match schedule: source={source_row_count}, "
            f"target={target_row_count}."
        )

    duplicate_count = connection.execute(
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

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate scoring environment game "
            "identifiers found."
        )

    invalid_window_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE league_game_count_last_32
                NOT BETWEEN 0 AND 32
           OR league_game_count_last_64
                NOT BETWEEN 0 AND 64
           OR league_game_count_last_128
                NOT BETWEEN 0 AND 128
           OR league_game_count_last_32
                > league_game_count_last_64
           OR league_game_count_last_64
                > league_game_count_last_128
        """
    ).fetchone()[0]

    if invalid_window_count > 0:
        raise RuntimeError(
            "Invalid scoring environment window "
            "counts found."
        )

    invalid_value_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                league_game_count_last_32 = 0
                AND league_average_total_last_32
                    IS NOT NULL
              )
           OR (
                league_game_count_last_32 > 0
                AND league_average_total_last_32
                    IS NULL
              )
           OR (
                league_game_count_last_64 = 0
                AND league_average_total_last_64
                    IS NOT NULL
              )
           OR (
                league_game_count_last_64 > 0
                AND league_average_total_last_64
                    IS NULL
              )
           OR (
                league_game_count_last_128 = 0
                AND league_average_total_last_128
                    IS NOT NULL
              )
           OR (
                league_game_count_last_128 > 0
                AND league_average_total_last_128
                    IS NULL
              )
           OR league_average_total_last_32
                NOT BETWEEN 0.0 AND 150.0
           OR league_average_total_last_64
                NOT BETWEEN 0.0 AND 150.0
           OR league_average_total_last_128
                NOT BETWEEN 0.0 AND 150.0
           OR league_total_standard_deviation_last_32
                < 0.0
           OR league_total_standard_deviation_last_64
                < 0.0
           OR league_total_standard_deviation_last_128
                < 0.0
        """
    ).fetchone()[0]

    if invalid_value_count > 0:
        raise RuntimeError(
            "Invalid scoring environment values found."
        )

    logger.info(
        "Game scoring environment validated: "
        "%s rows.",
        target_row_count,
    )


def build_game_scoring_environment(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build league scoring-environment features."""

    validate_database_file(database_file)

    logger.info(
        "Starting game scoring environment build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_scoring_environment_source(
            connection
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            create_game_scoring_environment_table(
                connection
            )

            validate_game_scoring_environment(
                connection
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "Game scoring environment build completed: "
        "%s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the scoring-environment builder."""

    try:
        build_game_scoring_environment()
    except Exception:
        logger.exception(
            "Game scoring environment build failed."
        )
        raise


if __name__ == "__main__":
    main()